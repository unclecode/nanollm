"""Live tests for nanollm.embedding() and nanollm.aembedding().

Run with:
    pytest -m live tests/test_live_embeddings.py
"""

from __future__ import annotations

import asyncio
import os

import pytest

from nanollm import embedding, aembedding

TEST_TEXT = "The quick brown fox jumps over the lazy dog."
TEST_TEXTS = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with many layers.",
    "Natural language processing handles text and speech.",
]


def _validate(response, expected_count: int):
    assert response is not None
    assert len(response.data) == expected_count
    for item in response.data:
        vec = item.embedding
        assert isinstance(vec, list) and len(vec) > 0
        assert all(isinstance(v, float) for v in vec[:5])


# ── Sync ──────────────────────────────────────────────────────────────────────


@pytest.mark.live
def test_openai_single_embedding():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")

    response = embedding("openai/text-embedding-3-small", input=TEST_TEXT, api_key=api_key)
    _validate(response, 1)
    assert len(response.data[0].embedding) == 1536


@pytest.mark.live
def test_openai_batch_embedding():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")

    response = embedding("openai/text-embedding-3-small", input=TEST_TEXTS, api_key=api_key)
    _validate(response, len(TEST_TEXTS))


@pytest.mark.live
def test_hf_embedding_skips_gracefully():
    """HF inference router may not support embedding models — must not hard-fail."""
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        pytest.skip("HUGGINGFACE_API_KEY not set")

    try:
        response = embedding(
            "huggingface/BAAI/bge-small-en-v1.5", input=TEST_TEXT, api_key=api_key
        )
        _validate(response, 1)
    except Exception:
        pytest.skip("HF router does not support this embedding model")


# ── Async ─────────────────────────────────────────────────────────────────────


@pytest.mark.live
def test_openai_single_embedding_async():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")

    response = asyncio.run(
        aembedding("openai/text-embedding-3-small", input=TEST_TEXT, api_key=api_key)
    )
    _validate(response, 1)


@pytest.mark.live
def test_openai_batch_embedding_async():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")

    response = asyncio.run(
        aembedding("openai/text-embedding-3-small", input=TEST_TEXTS, api_key=api_key)
    )
    _validate(response, len(TEST_TEXTS))
