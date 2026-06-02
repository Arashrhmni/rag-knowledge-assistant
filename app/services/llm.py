import logging
from typing import List, Dict, Any, AsyncGenerator
from openai import AsyncOpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a precise, helpful knowledge assistant.
Answer the user's question using ONLY the context provided below.
If the context does not contain enough information to answer confidently, say so clearly.
Do not hallucinate or use knowledge outside of the provided context.
Cite the source document when referencing specific information.
Be concise but complete."""


def _build_context_block(hits: List[Dict[str, Any]]) -> str:
    parts = []
    for i, hit in enumerate(hits, 1):
        source = hit["metadata"].get("source", "unknown")
        score = hit.get("score", 0)
        parts.append(
            f"[Context {i} | source: {source} | relevance: {score:.2f}]\n{hit['content']}"
        )
    return "\n\n---\n\n".join(parts)


async def generate_answer(
    question: str,
    hits: List[Dict[str, Any]],
) -> str:
    """Non-streaming answer generation."""
    if not settings.openai_api_key:
        return _fallback_answer(question, hits)

    client = AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )
    context = _build_context_block(hits)
    response = await client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n\n{context}\n\nQuestion: {question}",
            },
        ],
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content


async def stream_answer(
    question: str,
    hits: List[Dict[str, Any]],
) -> AsyncGenerator[str, None]:
    """Streaming answer generation (SSE-compatible)."""
    if not settings.openai_api_key:
        yield _fallback_answer(question, hits)
        return

    client = AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )
    context = _build_context_block(hits)
    stream = await client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n\n{context}\n\nQuestion: {question}",
            },
        ],
        temperature=0.2,
        max_tokens=1024,
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def _fallback_answer(question: str, hits: List[Dict[str, Any]]) -> str:
    """
    No-LLM fallback: returns the top retrieved chunks directly.
    Useful for testing the retrieval pipeline without an API key.
    """
    if not hits:
        return "No relevant documents found for your question."

    lines = [
        "⚠️  No OpenAI API key configured — showing raw retrieved context instead.\n",
        f"Top {len(hits)} relevant chunk(s) for: *{question}*\n",
    ]
    for i, hit in enumerate(hits, 1):
        src = hit["metadata"].get("source", "unknown")
        lines.append(f"\n**[{i}] {src}** (score: {hit['score']:.2f})\n{hit['content']}")
    return "\n".join(lines)
