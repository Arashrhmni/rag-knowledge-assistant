import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import hashlib
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)


class VectorStore:
    """
    ChromaDB-backed vector store with local sentence-transformers embeddings.
    No OpenAI API key required for indexing — only for generation.
    """

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        self.embedder = SentenceTransformer(settings.embedding_model)
        logger.info("Vector store ready.")

    def _embed(self, texts: List[str]) -> List[List[float]]:
        return self.embedder.encode(texts, show_progress_bar=False).tolist()

    def _doc_id(self, content: str, source: str, chunk_index: int) -> str:
        h = hashlib.md5(f"{source}:{chunk_index}:{content[:64]}".encode()).hexdigest()
        return h

    def add_documents(
        self,
        chunks: List[str],
        source: str,
        metadata_extra: Optional[Dict[str, Any]] = None,
    ) -> int:
        if not chunks:
            return 0

        ids = [self._doc_id(c, source, i) for i, c in enumerate(chunks)]
        embeddings = self._embed(chunks)
        metadatas = [
            {
                "source": source,
                "chunk_index": i,
                **(metadata_extra or {}),
            }
            for i in range(len(chunks))
        ]

        # Upsert to avoid duplicates on re-ingest
        self.collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info(f"Upserted {len(chunks)} chunks from '{source}'")
        return len(chunks)

    def query(self, question: str, top_k: int = None) -> List[Dict[str, Any]]:
        k = top_k or settings.top_k
        embedding = self._embed([question])[0]
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(k, self.collection.count() or 1),
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            score = 1 - dist  # cosine distance → similarity
            if score >= settings.similarity_threshold:
                hits.append({"content": doc, "metadata": meta, "score": round(score, 4)})
        return hits

    def list_sources(self) -> List[Dict[str, Any]]:
        if self.collection.count() == 0:
            return []
        results = self.collection.get(include=["metadatas"])
        seen: Dict[str, int] = {}
        for meta in results["metadatas"]:
            src = meta.get("source", "unknown")
            seen[src] = seen.get(src, 0) + 1
        return [{"source": s, "chunks": c} for s, c in seen.items()]

    def delete_source(self, source: str) -> int:
        results = self.collection.get(where={"source": source}, include=["metadatas"])
        ids = results["ids"]
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    def count(self) -> int:
        return self.collection.count()
