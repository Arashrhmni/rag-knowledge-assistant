from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import json
import logging

from app.services.llm import generate_answer, stream_answer
from app.core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: Optional[int] = Field(None, ge=1, le=20)
    stream: bool = False


class ContextChunk(BaseModel):
    content: str
    source: str
    score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    context: List[ContextChunk]
    model: str


@router.post("/query", response_model=QueryResponse)
async def query(request: Request, body: QueryRequest):
    """
    Query the knowledge base. Returns an LLM-generated answer grounded
    in retrieved document chunks.

    Set `stream: true` to receive a Server-Sent Events stream instead.
    """
    vector_store = request.app.state.vector_store

    if vector_store.count() == 0:
        raise HTTPException(
            status_code=422,
            detail="Knowledge base is empty. Please ingest documents first.",
        )

    hits = vector_store.query(body.question, top_k=body.top_k)

    if not hits:
        return QueryResponse(
            question=body.question,
            answer="I couldn't find any relevant information in the knowledge base for your question.",
            context=[],
            model=settings.openai_model,
        )

    if body.stream:
        # SSE streaming response
        async def event_generator():
            # First emit the context chunks
            context_payload = [
                {
                    "content": h["content"],
                    "source": h["metadata"].get("source", "unknown"),
                    "score": h["score"],
                }
                for h in hits
            ]
            yield f"data: {json.dumps({'type': 'context', 'context': context_payload})}\n\n"

            # Stream the answer tokens
            async for token in stream_answer(body.question, hits):
                yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming
    answer = await generate_answer(body.question, hits)

    return QueryResponse(
        question=body.question,
        answer=answer,
        context=[
            ContextChunk(
                content=h["content"],
                source=h["metadata"].get("source", "unknown"),
                score=h["score"],
            )
            for h in hits
        ],
        model=settings.openai_model,
    )
