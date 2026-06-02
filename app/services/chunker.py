import re
import io
import csv
import logging
from typing import List, Tuple
from pathlib import Path

from app.core.config import settings

logger = logging.getLogger(__name__)


def _split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Sentence-aware chunker: splits on sentence boundaries,
    then groups into chunks of ~chunk_size characters with overlap.
    """
    # Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text.strip())

    # Split into sentences (simple heuristic)
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks: List[str] = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= chunk_size:
            current = (current + " " + sentence).strip()
        else:
            if current:
                chunks.append(current)
            # Handle sentences longer than chunk_size
            if len(sentence) > chunk_size:
                for i in range(0, len(sentence), chunk_size - overlap):
                    chunks.append(sentence[i : i + chunk_size])
            else:
                current = sentence

    if current:
        chunks.append(current)

    # Apply overlap by prepending tail of previous chunk
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-overlap:]
            overlapped.append((tail + " " + chunks[i]).strip())
        return overlapped

    return chunks


def chunk_text(text: str) -> List[str]:
    return _split_text(text, settings.chunk_size, settings.chunk_overlap)


def parse_file(filename: str, content: bytes) -> Tuple[List[str], str]:
    """
    Parse uploaded file bytes into chunks.
    Returns (chunks, detected_source_label).
    """
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        chunks = _parse_pdf(content, filename)
    elif ext in (".txt", ".md"):
        text = content.decode("utf-8", errors="replace")
        chunks = chunk_text(text)
    elif ext == ".csv":
        chunks = _parse_csv(content)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    logger.info(f"Parsed '{filename}' → {len(chunks)} chunks")
    return chunks, filename


def _parse_pdf(content: bytes, filename: str) -> List[str]:
    try:
        import pypdf

        reader = pypdf.PdfReader(io.BytesIO(content))
        pages_text = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                pages_text.append(t)
        full_text = "\n\n".join(pages_text)
        return chunk_text(full_text)
    except ImportError:
        raise ImportError("pypdf not installed. Add it to requirements.txt.")


def _parse_csv(content: bytes) -> List[str]:
    text = content.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    rows = []
    for row in reader:
        row_text = "; ".join(f"{k}: {v}" for k, v in row.items() if v)
        rows.append(row_text)
    # Group rows into chunks
    combined = "\n".join(rows)
    return chunk_text(combined)
