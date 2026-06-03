"""
Integration tests for the RAG Knowledge Assistant API.
These tests use a temporary vector store and do NOT require
an OpenAI API key — they test the full pipeline except LLM generation.
"""
import io
import tempfile
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


class FakeEmbeddingResult:
    def __init__(self, count: int):
        self.vectors = [[1.0, 0.0, 0.0] for _ in range(count)]

    def tolist(self):
        return self.vectors


class FakeSentenceTransformer:
    def __init__(self, *_args, **_kwargs):
        pass

    def encode(self, texts, show_progress_bar=False):
        return FakeEmbeddingResult(len(texts))


@pytest.fixture(scope="module")
def client():
    # Use a fresh vector store and fake embeddings so CI never downloads a model.
    with tempfile.TemporaryDirectory() as chroma_dir:
        with patch("app.core.config.settings.chroma_persist_dir", chroma_dir):
            with patch("app.core.vector_store.SentenceTransformer", FakeSentenceTransformer):
                with TestClient(app) as c:
                    yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "vector_store_chunks" in data
    assert "embedding_model" in data


def test_ingest_text(client):
    response = client.post(
        "/api/v1/ingest/text",
        json={
            "text": "The Eiffel Tower is located in Paris, France. It was built in 1889 by Gustave Eiffel.",
            "source": "test_eiffel",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["chunks_added"] >= 1
    assert data["source"] == "test_eiffel"


def test_ingest_empty_text_rejected(client):
    response = client.post(
        "/api/v1/ingest/text",
        json={"text": "   ", "source": "empty"},
    )
    assert response.status_code == 400


def test_ingest_txt_file(client):
    content = b"Python is a high-level programming language. It was created by Guido van Rossum."
    response = client.post(
        "/api/v1/ingest/file",
        files={"file": ("test.txt", io.BytesIO(content), "text/plain")},
    )
    assert response.status_code == 200
    assert response.json()["chunks_added"] >= 1


def test_ingest_unsupported_format(client):
    response = client.post(
        "/api/v1/ingest/file",
        files={"file": ("test.exe", io.BytesIO(b"binary"), "application/octet-stream")},
    )
    assert response.status_code == 400


def test_list_sources(client):
    response = client.get("/api/v1/sources")
    assert response.status_code == 200
    data = response.json()
    assert "sources" in data
    assert "total_chunks" in data
    assert data["total_chunks"] >= 0


def test_query_no_llm(client):
    """Query should return fallback answer when no OpenAI key is set."""
    response = client.post(
        "/api/v1/query",
        json={"question": "Where is the Eiffel Tower?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "context" in data
    assert isinstance(data["context"], list)


def test_query_empty_kb(client):
    """Query on a fresh empty store should return a helpful message."""
    # Delete all sources first
    sources_resp = client.get("/api/v1/sources")
    for src in sources_resp.json()["sources"]:
        client.delete(f"/api/v1/sources/{src['source']}")

    response = client.post(
        "/api/v1/query",
        json={"question": "What is the meaning of life?"},
    )
    # Should return 422 when empty
    assert response.status_code == 422


def test_delete_source(client):
    # Ingest first
    client.post(
        "/api/v1/ingest/text",
        json={"text": "Temporary document to be deleted.", "source": "to_delete"},
    )
    # Delete it
    response = client.delete("/api/v1/sources/to_delete")
    assert response.status_code == 200
    assert response.json()["chunks_deleted"] >= 1


def test_delete_nonexistent_source(client):
    response = client.delete("/api/v1/sources/does_not_exist")
    assert response.status_code == 404
