import json
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.config import settings
from app.services.llm import generate_answer, stream_answer

logger = logging.getLogger(__name__)
router = APIRouter(tags=["query"])

NO_CONTEXT_ANSWER = (
    "I could not find relevant information in the indexed documents to answer this question."
)


# ------------------------------------------------------------------
# Schemas
# ------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str
    stream: bool = False
    top_k: int = Field(default=0, ge=0, le=20)  # 0 = use settings default


class ContextChunk(BaseModel):
    content: str
    source: str
    similarity: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    context: list[ContextChunk]
    model: str


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@router.post("/query")
async def query(request: Request, body: QueryRequest):
    if not body.question.strip():
        raise HTTPException(status_code=422, detail="Question cannot be empty.")

    top_k = body.top_k or settings.top_k
    logger.info("Query: '%s' (top_k=%d, stream=%s)", body.question[:80], top_k, body.stream)

    vector_store = request.app.state.vector_store
    if vector_store.get_chunk_count() == 0:
        raise HTTPException(status_code=404, detail="No documents have been indexed yet.")

    chunks = vector_store.query(body.question, top_k=top_k)

    if not chunks:
        logger.info("No relevant chunks found for query")
        if body.stream:
            return StreamingResponse(
                _empty_context_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        return QueryResponse(
            question=body.question,
            answer=NO_CONTEXT_ANSWER,
            context=[],
            model="none",
        )

    if body.stream:
        return StreamingResponse(
            _stream_response(body.question, chunks),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    answer = await generate_answer(body.question, chunks)
    model_name = settings.openai_model if settings.openai_api_key else "fallback"

    return QueryResponse(
        question=body.question,
        answer=answer,
        context=[
            ContextChunk(content=c["content"], source=c["source"], similarity=c["similarity"])
            for c in chunks
        ],
        model=model_name,
    )


# ------------------------------------------------------------------
# Streaming helpers
# ------------------------------------------------------------------

def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


async def _empty_context_stream():
    yield _sse({"type": "context", "context": []})
    yield _sse({"type": "token", "token": NO_CONTEXT_ANSWER})
    yield _sse({"type": "done"})


async def _stream_response(question: str, chunks: list[dict]):
    # Send context first so the UI can show sources immediately
    context_preview = [
        {
            "content": c["content"][:200],  # truncate for the wire
            "source": c["source"],
            "similarity": c["similarity"],
        }
        for c in chunks
    ]
    yield _sse({"type": "context", "context": context_preview})

    async for token in stream_answer(question, chunks):
        yield _sse({"type": "token", "token": token})

    yield _sse({"type": "done"})
