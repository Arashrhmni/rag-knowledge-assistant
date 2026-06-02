# RAG Knowledge Assistant

A production-grade **Retrieval-Augmented Generation** API and UI that lets you ingest documents (PDF, TXT, MD, CSV) and ask questions about them — grounded strictly in what you uploaded.

Built with **FastAPI**, **ChromaDB**, **sentence-transformers** (local embeddings, no API key needed for indexing), and **OpenAI** for generation. Fully containerized and Kubernetes-ready.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Client (browser / API)              │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP / SSE
┌────────────────────▼────────────────────────────────────┐
│                    FastAPI Application                   │
│                                                          │
│  POST /api/v1/ingest/file   POST /api/v1/ingest/text    │
│  POST /api/v1/query         GET  /api/v1/sources         │
│  DELETE /api/v1/sources/:s  GET  /health                 │
└──────┬──────────────────────────────┬───────────────────┘
       │                              │
┌──────▼──────────┐        ┌──────────▼──────────────────┐
│  Chunker        │        │  Vector Store (ChromaDB)     │
│  - sentence-    │        │  - cosine similarity search  │
│    aware split  │──────▶│  - persistent to disk        │
│  - PDF/TXT/CSV  │        │  - upsert (no duplicates)   │
└─────────────────┘        └──────────┬──────────────────┘
                                       │  top-k hits
                           ┌──────────▼──────────────────┐
                           │  Embeddings                  │
                           │  sentence-transformers       │
                           │  all-MiniLM-L6-v2 (local)   │
                           └─────────────────────────────┘
                                       │  context chunks
                           ┌──────────▼──────────────────┐
                           │  LLM Generation (OpenAI)     │
                           │  - streaming SSE support     │
                           │  - graceful fallback if      │
                           │    no API key configured     │
                           └─────────────────────────────┘
```

**Key design decisions:**
- Embeddings run **locally** via `sentence-transformers` — no external API needed for indexing
- ChromaDB persists to disk — survives container restarts
- Upsert-on-ingest — re-uploading a file doesn't create duplicates
- Answers stream token-by-token via SSE — no waiting for the full response
- Works **without** an OpenAI key — returns raw retrieved chunks as fallback (great for testing retrieval quality independently of generation)

---

## Quickstart

### Option 1 — Docker Compose (recommended)

```bash
git clone https://github.com/yourusername/rag-knowledge-assistant
cd rag-knowledge-assistant

cp .env.example .env
# Edit .env and add your OPENAI_API_KEY (optional)

docker compose up --build
```

Open [http://localhost:8000](http://localhost:8000) — the UI loads automatically.

---

### Option 2 — Local Python

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env  # add your OPENAI_API_KEY

uvicorn app.main:app --reload
```

API docs at [http://localhost:8000/docs](http://localhost:8000/docs)

---

### Option 3 — Kubernetes

```bash
# Build and push image
docker build -t your-registry/rag-assistant:latest .
docker push your-registry/rag-assistant:latest

# Create secret (optional — needed only for LLM generation)
kubectl create secret generic rag-secrets \
  --from-literal=openai-api-key=sk-...

# Deploy
kubectl apply -f k8s/deployment.yaml

# Port-forward for local access
kubectl port-forward svc/rag-assistant-svc 8000:80
```

---

## API Reference

### Ingest a file

```bash
curl -X POST http://localhost:8000/api/v1/ingest/file \
  -F "file=@my_document.pdf"
```

```json
{
  "source": "my_document.pdf",
  "chunks_added": 42,
  "message": "Successfully indexed 42 chunks from 'my_document.pdf'"
}
```

### Ingest raw text

```bash
curl -X POST http://localhost:8000/api/v1/ingest/text \
  -H "Content-Type: application/json" \
  -d '{"text": "FastAPI is a modern Python web framework...", "source": "fastapi-notes"}'
```

### Query (standard)

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is FastAPI used for?"}'
```

```json
{
  "question": "What is FastAPI used for?",
  "answer": "Based on the provided context, FastAPI is a modern Python web framework...",
  "context": [
    {
      "content": "FastAPI is a modern Python web framework...",
      "source": "fastapi-notes",
      "score": 0.91
    }
  ],
  "model": "gpt-4o-mini"
}
```

### Query (streaming)

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarize the document", "stream": true}'
```

Returns SSE stream:
```
data: {"type": "context", "context": [...]}
data: {"type": "token", "token": "Based"}
data: {"type": "token", "token": " on"}
...
data: {"type": "done"}
```

### List sources

```bash
curl http://localhost:8000/api/v1/sources
```

### Delete a source

```bash
curl -X DELETE http://localhost:8000/api/v1/sources/my_document.pdf
```

### Health check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "uptime_seconds": 42.1,
  "vector_store_chunks": 156,
  "embedding_model": "all-MiniLM-L6-v2",
  "llm_configured": true
}
```

---

## Configuration

All settings are controlled via environment variables (or `.env` file):

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | `""` | OpenAI key — optional, fallback mode if unset |
| `OPENAI_MODEL` | `gpt-4o-mini` | Any OpenAI-compatible model |
| `OPENAI_BASE_URL` | OpenAI | Override for Ollama or other providers |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model (local) |
| `CHUNK_SIZE` | `512` | Target chars per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between adjacent chunks |
| `TOP_K` | `5` | Retrieved chunks per query |
| `SIMILARITY_THRESHOLD` | `0.3` | Minimum cosine similarity to include a chunk |
| `MAX_FILE_SIZE_MB` | `50` | Upload size limit |

### Using Ollama (fully local, no OpenAI needed)

```env
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_MODEL=llama3.2
OPENAI_API_KEY=ollama
```

---

## Running Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Tests cover: health endpoint, file ingestion, text ingestion, source listing, deletion, query response structure, and edge cases (empty KB, unsupported formats). No OpenAI key required.

---

## Project Structure

```
rag-knowledge-assistant/
├── app/
│   ├── main.py              # FastAPI app + lifespan
│   ├── api/
│   │   ├── ingest.py        # /ingest/file, /ingest/text, /sources
│   │   ├── query.py         # /query (standard + SSE streaming)
│   │   └── health.py        # /health
│   ├── core/
│   │   ├── config.py        # Pydantic settings
│   │   └── vector_store.py  # ChromaDB abstraction
│   └── services/
│       ├── chunker.py       # Sentence-aware text splitting + file parsing
│       └── llm.py           # OpenAI generation + streaming + fallback
├── frontend/
│   └── index.html           # Single-file UI (drag-drop, chat, SSE)
├── tests/
│   ├── test_api.py          # API integration tests
│   └── test_chunker.py      # Chunker unit tests
├── k8s/
│   └── deployment.yaml      # Deployment + Service + PVC
├── .github/
│   └── workflows/ci.yml     # Test → lint → Docker build + smoke test
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
└── .env.example
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API framework | FastAPI |
| Vector database | ChromaDB (persistent) |
| Embeddings | sentence-transformers / all-MiniLM-L6-v2 |
| LLM generation | OpenAI API (gpt-4o-mini default) |
| PDF parsing | pypdf |
| Containerization | Docker + Docker Compose |
| Orchestration | Kubernetes |
| CI/CD | GitHub Actions |
| Testing | pytest |

---

## License

MIT
