from fastapi import APIRouter, UploadFile, File, Request, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import logging

from app.services.chunker import parse_file, chunk_text
from app.core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)


class IngestTextRequest(BaseModel):
    text: str
    source: str = "manual_input"


class IngestResponse(BaseModel):
    source: str
    chunks_added: int
    message: str


class SourcesResponse(BaseModel):
    sources: List[dict]
    total_chunks: int


class DeleteResponse(BaseModel):
    source: str
    chunks_deleted: int


@router.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(request: Request, file: UploadFile = File(...)):
    """Upload a file (PDF, TXT, MD, CSV) and index it into the vector store."""
    ext = "." + file.filename.split(".")[-1].lower() if "." in file.filename else ""
    if ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{ext}' not supported. Allowed: {settings.allowed_extensions}",
        )

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.max_file_size_mb:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Max: {settings.max_file_size_mb} MB",
        )

    try:
        chunks, source_label = parse_file(file.filename, content)
    except Exception as e:
        logger.exception("File parsing failed")
        raise HTTPException(status_code=422, detail=str(e))

    vector_store = request.app.state.vector_store
    added = vector_store.add_documents(chunks, source=source_label)

    return IngestResponse(
        source=source_label,
        chunks_added=added,
        message=f"Successfully indexed {added} chunks from '{file.filename}'",
    )


@router.post("/ingest/text", response_model=IngestResponse)
async def ingest_text(request: Request, body: IngestTextRequest):
    """Ingest raw text directly (useful for testing or API integrations)."""
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    chunks = chunk_text(body.text)
    vector_store = request.app.state.vector_store
    added = vector_store.add_documents(chunks, source=body.source)

    return IngestResponse(
        source=body.source,
        chunks_added=added,
        message=f"Successfully indexed {added} chunks from text input.",
    )


@router.get("/sources", response_model=SourcesResponse)
async def list_sources(request: Request):
    """List all indexed document sources and their chunk counts."""
    vector_store = request.app.state.vector_store
    sources = vector_store.list_sources()
    return SourcesResponse(sources=sources, total_chunks=vector_store.count())


@router.delete("/sources/{source:path}", response_model=DeleteResponse)
async def delete_source(request: Request, source: str):
    """Remove all chunks for a given source document."""
    vector_store = request.app.state.vector_store
    deleted = vector_store.delete_source(source)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"Source '{source}' not found.")
    return DeleteResponse(source=source, chunks_deleted=deleted)
