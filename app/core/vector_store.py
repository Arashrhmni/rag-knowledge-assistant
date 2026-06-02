import logging

import chromadb
from sentence_transformers import SentenceTransformer

from app.core.config import settings

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self) -> None:
        logger.info("Loading embedding model: %s", settings.embedding_model)
        self._model = SentenceTransformer(settings.embedding_model)
        self._client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "VectorStore ready — %d chunks indexed across %d sources",
            self._collection.count(),
            len(self.list_sources()),
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: list[str], source: str) -> int:
        """Embed and store chunks. Replaces any existing chunks for this source."""
        if not chunks:
            return 0

        # Remove old chunks before adding new ones (upsert semantics)
        self.delete_source(source)

        logger.info("Embedding %d chunks for source '%s'", len(chunks), source)
        embeddings = self._model.encode(chunks).tolist()
        ids = [f"{source}::{i}" for i in range(len(chunks))]
        metadatas = [{"source": source, "chunk_index": i} for i in range(len(chunks))]

        self._collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )
        logger.info("Stored %d chunks for source '%s'", len(chunks), source)
        return len(chunks)

    def delete_source(self, source: str) -> int:
        """Delete all chunks for a source. Returns the number of chunks deleted."""
        try:
            existing = self._collection.get(where={"source": source})
            count = len(existing["ids"])
            if count > 0:
                self._collection.delete(where={"source": source})
                logger.info("Deleted %d chunks for source '%s'", count, source)
            return count
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def query(self, question: str, top_k: int | None = None) -> list[dict]:
        """Return top-k relevant chunks for a question, filtered by similarity threshold."""
        k = top_k or settings.top_k
        total = self._collection.count()

        if total == 0:
            return []

        k = min(k, total)
        logger.debug("Querying for: '%s' (top_k=%d)", question[:60], k)

        query_embedding = self._model.encode(question).tolist()
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for doc, meta, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB returns cosine distance (0 = identical, 2 = opposite).
            # Convert to similarity in [0, 1] range.
            similarity = round(1 - (distance / 2), 4)
            if similarity >= settings.similarity_threshold:
                chunks.append({
                    "content": doc,
                    "source": meta.get("source", "unknown"),
                    "chunk_index": meta.get("chunk_index", 0),
                    "similarity": similarity,
                })

        logger.debug(
            "Found %d relevant chunks above threshold %.2f",
            len(chunks),
            settings.similarity_threshold,
        )
        return chunks

    def list_sources(self) -> list[str]:
        """Return a sorted list of all indexed source names."""
        try:
            result = self._collection.get(include=["metadatas"])
            sources = {m.get("source", "unknown") for m in result["metadatas"]}
            return sorted(sources)
        except Exception:
            return []

    def get_chunk_count(self) -> int:
        """Return the total number of chunks stored."""
        return self._collection.count()
