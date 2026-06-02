import io
import logging

from pypdf import PdfReader

from app.core.config import settings

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".csv"}


# ------------------------------------------------------------------
# File parsing
# ------------------------------------------------------------------

def parse_file(content: bytes, filename: str) -> str:
    """Parse uploaded file bytes into plain text."""
    ext = f".{filename.rsplit('.', 1)[-1].lower()}" if "." in filename else ""

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    logger.debug("Parsing '%s' (%d bytes)", filename, len(content))

    if ext == ".pdf":
        return _parse_pdf(content)
    return content.decode("utf-8", errors="replace")


def _parse_pdf(content: bytes) -> str:
    reader = PdfReader(io.BytesIO(content))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(p.strip() for p in pages if p.strip())
    logger.debug("Extracted %d chars from %d PDF pages", len(text), len(reader.pages))
    return text


# ------------------------------------------------------------------
# Text splitting
# ------------------------------------------------------------------

def _split_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
    """Split text into overlapping chunks of ~chunk_size characters."""
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap

    if not text or not text.strip():
        return []

    step = max(chunk_size - overlap, 1)
    chunks = []
    start = 0

    while start < len(text):
        chunk = text[start : start + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
        start += step

    return chunks


def chunk_text(text: str) -> list[str]:
    """Chunk text using default settings."""
    chunks = _split_text(text)
    logger.debug("Chunked %d chars → %d chunks", len(text), len(chunks))
    return chunks
