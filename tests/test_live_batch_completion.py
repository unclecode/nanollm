"""Live tests for nanollm.batch_completion().

Marked with @pytest.mark.live — skipped by default, run with:
    pytest -m live tests/test_live_batch_completion.py

All tests that need API keys skip automatically when the key is absent.
"""

from __future__ import annotations

import os

import pytest

import nanollm
from nanollm import batch_completion


# ── No-network (always run) ───────────────────────────────────────────────────


def test_empty_batch_returns_empty_list():
    """batch_completion(model, []) must return [] without touching the network."""
    assert batch_completion("openai/gpt-4o-mini", []) == []


def test_empty_batch_with_large_max_workers():
    """max_workers cap: passing 1000 for 0 messages must not crash."""
    assert batch_completion("openai/gpt-4o-mini", [], max_workers=1000) == []


def test_empty_batch_does_not_invoke_logger():
    call_count = 0

    def counter(_r):
        nonlocal call_count
        call_count += 1

    batch_completion("openai/gpt-4o-mini", [], logger_fn=counter)
    assert call_count == 0


# ── Live tests ────────────────────────────────────────────────────────────────


@pytest.mark.live
def test_single_message_openai():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")

    results = batch_completion(
        "openai/gpt-4o-mini",
        [[{"role": "user", "content": "Reply with only the number 42"}]],
        api_key=api_key,
    )
    assert len(results) == 1
    assert results[0].choices[0].message.content


@pytest.mark.live
def test_parallel_batch_openai():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")

    messages_batch = [
        [{"role": "user", "content": f"Reply with only: item_{i}"}]
        for i in range(5)
    ]
    results = batch_completion("openai/gpt-4o-mini", messages_batch, api_key=api_key)

    assert len(results) == 5
    for r in results:
        assert r.choices[0].message.content


@pytest.mark.live
def test_max_workers_capped_to_batch_size():
    """max_workers=1000 with 3 messages should work fine (capped internally)."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")

    messages_batch = [
        [{"role": "user", "content": "Say yes"}],
        [{"role": "user", "content": "Say no"}],
        [{"role": "user", "content": "Say maybe"}],
    ]
    results = batch_completion(
        "openai/gpt-4o-mini", messages_batch, api_key=api_key, max_workers=1000
    )
    assert len(results) == 3


@pytest.mark.live
@pytest.mark.bedrock
def test_single_message_bedrock():
    if not os.getenv("AWS_ACCESS_KEY_ID"):
        pytest.skip("AWS_ACCESS_KEY_ID not set")

    results = batch_completion(
        "bedrock/apac.amazon.nova-micro-v1:0",
        [[{"role": "user", "content": "Reply with one word: hello"}]],
    )
    assert len(results) == 1
    assert results[0].choices[0].message.content
