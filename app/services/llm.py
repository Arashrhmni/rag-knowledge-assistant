import logging
from collections.abc import AsyncGenerator

from openai import APIConnectionError, APIStatusError, AsyncOpenAI, RateLimitError

from app.core.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based ONLY on the provided context. "
    "If the answer cannot be found in the context, say so clearly. "
    "Do not use outside knowledge or make assumptions beyond what is provided."
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _is_configured() -> bool:
    return bool(settings.openai_api_key)


def _get_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )


def _build_user_message(question: str, chunks: list[dict]) -> str:
    context_parts = [
        f"[Source: {c['source']}]\n{c['content']}"
        for c in chunks
    ]
    context = "\n\n---\n\n".join(context_parts)
    return f"Context:\n{context}\n\nQuestion: {question}"


def _fallback_response(chunks: list[dict]) -> str:
    """Return raw retrieved chunks when no LLM is configured."""
    parts = [f"[Source: {c['source']}]\n{c['content']}" for c in chunks]
    return "No LLM configured. Relevant passages:\n\n" + "\n\n---\n\n".join(parts)


# ------------------------------------------------------------------
# Generation
# ------------------------------------------------------------------

async def generate_answer(question: str, chunks: list[dict]) -> str:
    """Generate a single answer string for the given question and context chunks."""
    if not _is_configured():
        logger.debug("No OpenAI key configured — returning raw chunks")
        return _fallback_response(chunks)

    logger.debug("Generating answer for: '%s'", question[:60])
    try:
        response = await _get_client().chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_message(question, chunks)},
            ],
            temperature=0.1,
        )
        answer = response.choices[0].message.content or ""
        logger.debug("Answer generated (%d chars)", len(answer))
        return answer

    except RateLimitError:
        logger.warning("OpenAI rate limit reached")
        return "Rate limit reached. Please try again in a moment."
    except APIConnectionError:
        logger.error("Could not connect to OpenAI API")
        return "Could not connect to the language model. Check your API configuration."
    except APIStatusError as e:
        logger.error("OpenAI API error %s", e.status_code)
        return f"Language model returned an error (HTTP {e.status_code})."


async def stream_answer(question: str, chunks: list[dict]) -> AsyncGenerator[str, None]:
    """Async generator that yields answer tokens one by one."""
    if not _is_configured():
        yield _fallback_response(chunks)
        return

    try:
        stream = await _get_client().chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_message(question, chunks)},
            ],
            temperature=0.1,
            stream=True,
        )
        async for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                yield token

    except RateLimitError:
        yield "Rate limit reached. Please try again in a moment."
    except APIConnectionError:
        yield "Could not connect to the language model."
    except APIStatusError as e:
        yield f"Language model error (HTTP {e.status_code})."
