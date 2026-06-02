import logging

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

from app.core.config import settings
from app.services.chunker import chunk_text, parse_file

logger = logging.getLogger(__name__)
router = APIRouter(tags=["ingest"])

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".csv"}


# ------------------------------------------------------------------
# Schemas
# ------------------------------------------------------------------

class TextIngestRequest(BaseModel):
    text: str
    source: str


class IngestResponse(BaseModel):
    source: str
    chunks_added: int
    message: str


class SourcesResponse(BaseModel):
    sources: list[str]
    total: int


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@router.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(request: Request, file: UploadFile = File(...)) -> IngestResponse:
    filename = file.filename or "upload"
    logger.info("Ingest file: '%s'", filename)

    ext = f".{filename.rsplit('.', 1)[-1].lower()}" if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{ext}' is not supported. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    content = await file.read()

    if len(content) == 0:
        raise HTTPException(status_code=422, detail="The uploaded file is empty.")

    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        size_mb = len(content) / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"File is too large ({size_mb:.1f} MB). Maximum allowed size is {settings.max_file_size_mb} MB.",
        )

    try:
        text = parse_file(content, filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not text.strip():
        raise HTTPException(status_code=422, detail="No text could be extracted from the file.")

    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=422, detail="File produced no usable text chunks.")

    request.app.state.vector_store.add_chunks(chunks, source=filename)
    logger.info("Indexed '%s': %d chunks added", filename, len(chunks))

    return IngestResponse(
        source=filename,
        chunks_added=len(chunks),
        message=f"Successfully indexed {len(chunks)} chunks from '{filename}'.",
    )


@router.post("/ingest/text", response_model=IngestResponse)
async def ingest_text(request: Request, body: TextIngestRequest) -> IngestResponse:
    if not body.text.strip():
        raise HTTPException(status_code=422, detail="Text content cannot be empty.")
    if not body.source.strip():
        raise HTTPException(status_code=422, detail="A source name is required.")

    logger.info("Ingest text: source='%s' (%d chars)", body.source, len(body.text))

    chunks = chunk_text(body.text)
    if not chunks:
        raise HTTPException(status_code=422, detail="Text produced no usable chunks.")

    request.app.state.vector_store.add_chunks(chunks, source=body.source)

    return IngestResponse(
        source=body.source,
        chunks_added=len(chunks),
        message=f"Successfully indexed {len(chunks)} chunks from '{body.source}'.",
    )


@router.get("/sources", response_model=SourcesResponse)
async def list_sources(request: Request) -> SourcesResponse:
    sources = request.app.state.vector_store.list_sources()
    return SourcesResponse(sources=sources, total=len(sources))


@router.delete("/sources/{source_name}")
async def delete_source(source_name: str, request: Request) -> dict:
    deleted = request.app.state.vector_store.delete_source(source_name)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"Source '{source_name}' not found.")
    logger.info("Deleted source '%s' (%d chunks)", source_name, deleted)
    return {"message": f"Deleted '{source_name}' ({deleted} chunks removed)."}
