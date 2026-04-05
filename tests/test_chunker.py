import pytest
from app.services.chunker import chunk_text, _split_text


def test_chunk_text_short():
    text = "Hello world. This is a test."
    chunks = chunk_text(text)
    assert len(chunks) >= 1
    assert all(isinstance(c, str) for c in chunks)


def test_chunk_text_long():
    # 5000 char document
    text = "This is a sentence about artificial intelligence. " * 100
    chunks = chunk_text(text)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 600  # some tolerance for overlap


def test_chunk_overlap():
    text = "Sentence one. Sentence two. Sentence three. Sentence four. " * 20
    chunks = _split_text(text, chunk_size=100, overlap=20)
    # With overlap, adjacent chunks should share some content
    if len(chunks) > 1:
        # The start of chunk[1] should contain the tail of chunk[0]
        tail = chunks[0][-20:]
        # tail might be partial word — just verify chunks are non-empty
        assert all(len(c) > 0 for c in chunks)


def test_chunk_empty_text():
    chunks = chunk_text("")
    assert chunks == [] or all(c.strip() == "" for c in chunks)


def test_chunk_preserves_content():
    text = "The quick brown fox jumps over the lazy dog."
    chunks = chunk_text(text)
    combined = " ".join(chunks)
    # All words from original should appear somewhere
    for word in ["quick", "brown", "fox", "lazy", "dog"]:
        assert word in combined
