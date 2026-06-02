from fastapi import APIRouter, Request
from pydantic import BaseModel

from app.core.config import settings

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    llm_configured: bool
    llm_model: str
    embedding_model: str
    chunks_indexed: int
    sources_indexed: int


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    vector_store = request.app.state.vector_store
    return HealthResponse(
        status="ok",
        llm_configured=bool(settings.openai_api_key),
        llm_model=settings.openai_model if settings.openai_api_key else "none",
        embedding_model=settings.embedding_model,
        chunks_indexed=vector_store.get_chunk_count(),
        sources_indexed=len(vector_store.list_sources()),
    )
