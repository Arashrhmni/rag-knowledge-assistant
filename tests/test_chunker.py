"""
Chunker unit tests.
Run with:  python -m pytest tests/ -v
"""
from app.services.chunker import _split_text, chunk_text


def test_chunk_text_short():
    chunks = chunk_text("Hello world. This is a test.")
    assert len(chunks) >= 1
    assert all(isinstance(c, str) for c in chunks)


def test_chunk_text_long():
    text = "This is a sentence about artificial intelligence. " * 100
    chunks = chunk_text(text)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 600  # tolerance for overlap


def test_chunk_overlap():
    text = "Sentence one. Sentence two. Sentence three. Sentence four. " * 20
    chunks = _split_text(text, chunk_size=100, overlap=20)
    if len(chunks) > 1:
        assert all(len(c) > 0 for c in chunks)


def test_chunk_empty_text():
    assert chunk_text("") == []


def test_chunk_whitespace_only():
    assert chunk_text("     \n\n\t  ") == []


def test_chunk_preserves_content():
    text = "The quick brown fox jumps over the lazy dog."
    chunks = chunk_text(text)
    combined = " ".join(chunks)
    for word in ["quick", "brown", "fox", "lazy", "dog"]:
        assert word in combined


def test_chunk_respects_chunk_size():
    text = "word " * 500  # 2500 chars
    chunks = _split_text(text, chunk_size=200, overlap=20)
    assert len(chunks) > 1
    # No chunk should exceed chunk_size (without tolerance — exact slicing)
    for chunk in chunks:
        assert len(chunk) <= 200
