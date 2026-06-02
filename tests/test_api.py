"""
API endpoint tests.
Run with:  python -m pytest tests/ -v
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app


# ------------------------------------------------------------------
# Fixture — wraps client in context manager so lifespan runs
# (this is what initialises app.state.vector_store)
# ------------------------------------------------------------------

@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


# ------------------------------------------------------------------
# Health
# ------------------------------------------------------------------

def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "llm_configured" in data
    assert "embedding_model" in data
    assert "chunks_indexed" in data
    assert "sources_indexed" in data


def test_health_llm_configured_is_bool(client):
    response = client.get("/health")
    assert isinstance(response.json()["llm_configured"], bool)


# ------------------------------------------------------------------
# Ingest — text
# ------------------------------------------------------------------

def test_ingest_text_success(client):
    response = client.post(
        "/api/v1/ingest/text",
        json={"text": "The railway switch failed at 10:00 AM.", "source": "test-doc"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["source"] == "test-doc"
    assert data["chunks_added"] >= 1
    assert "Successfully" in data["message"]


def test_ingest_text_empty_text_returns_422(client):
    response = client.post(
        "/api/v1/ingest/text",
        json={"text": "   ", "source": "test"},
    )
    assert response.status_code == 422


def test_ingest_text_empty_source_returns_422(client):
    response = client.post(
        "/api/v1/ingest/text",
        json={"text": "Some content here.", "source": ""},
    )
    assert response.status_code == 422


# ------------------------------------------------------------------
# Ingest — file
# ------------------------------------------------------------------

def test_ingest_txt_file_success(client):
    file_content = b"This is a plain text document about railway maintenance."
    response = client.post(
        "/api/v1/ingest/file",
        files={"file": ("test.txt", file_content, "text/plain")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["chunks_added"] >= 1


def test_ingest_unsupported_file_type_returns_400(client):
    response = client.post(
        "/api/v1/ingest/file",
        files={"file": ("test.exe", b"binary content", "application/octet-stream")},
    )
    assert response.status_code == 400
    assert "not supported" in response.json()["detail"].lower()


def test_ingest_empty_file_returns_422(client):
    response = client.post(
        "/api/v1/ingest/file",
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert response.status_code == 422


# ------------------------------------------------------------------
# Sources
# ------------------------------------------------------------------

def test_list_sources_returns_list(client):
    client.post(
        "/api/v1/ingest/text",
        json={"text": "Content for source listing test.", "source": "source-list-test"},
    )
    response = client.get("/api/v1/sources")
    assert response.status_code == 200
    data = response.json()
    assert "sources" in data
    assert isinstance(data["sources"], list)
    assert isinstance(data["total"], int)
    assert data["total"] == len(data["sources"])


def test_delete_nonexistent_source_returns_404(client):
    response = client.delete("/api/v1/sources/this-source-does-not-exist-xyz")
    assert response.status_code == 404


# ------------------------------------------------------------------
# Query
# ------------------------------------------------------------------

def test_query_empty_question_returns_422(client):
    response = client.post("/api/v1/query", json={"question": "  "})
    assert response.status_code == 422


def test_query_returns_answer_structure(client):
    client.post(
        "/api/v1/ingest/text",
        json={
            "text": "KONUX uses IoT sensors to monitor railway switches and predict failures.",
            "source": "konux-overview",
        },
    )
    response = client.post(
        "/api/v1/query",
        json={"question": "What does KONUX monitor?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "question" in data
    assert "answer" in data
    assert "context" in data
    assert "model" in data
    assert isinstance(data["context"], list)
