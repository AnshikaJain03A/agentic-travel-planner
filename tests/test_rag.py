"""Tests for the RAG chunking and embedding fallback."""

from __future__ import annotations

from rag.embeddings import _hash_embedding
from rag.knowledge_base import chunk_text


def test_chunk_text_empty() -> None:
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_short() -> None:
    chunks = chunk_text("A short sentence.")
    assert len(chunks) == 1


def test_chunk_text_splits_long_input() -> None:
    text = ". ".join(f"This is sentence number {i}" for i in range(200))
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    assert len(chunks) > 1
    # No chunk should be drastically larger than the target size.
    assert all(len(c) <= 500 for c in chunks)


def test_hash_embedding_is_deterministic_and_normalised() -> None:
    a = _hash_embedding("goa beaches and seafood")
    b = _hash_embedding("goa beaches and seafood")
    assert a == b
    norm = sum(x * x for x in a) ** 0.5
    assert abs(norm - 1.0) < 1e-6
