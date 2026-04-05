from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import Optional
import time

router = APIRouter()
_start_time = time.time()


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    vector_store_chunks: int
    embedding_model: str
    llm_configured: bool


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    """Liveness + readiness probe."""
    from app.core.config import settings

    vs = getattr(request.app.state, "vector_store", None)
    chunk_count = vs.count() if vs else 0

    return HealthResponse(
        status="ok",
        uptime_seconds=round(time.time() - _start_time, 1),
        vector_store_chunks=chunk_count,
        embedding_model=settings.embedding_model,
        llm_configured=bool(settings.openai_api_key),
    )
