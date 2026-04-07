"""NanoLLM — Minimal, zero-bloat LLM API wrapper.

Drop-in replacement for litellm. Supports all major LLM providers
through a unified OpenAI-compatible interface.

Usage:
    from nanollm import completion, acompletion

    response = completion(model="openai/gpt-4o", messages=[...])
    print(response.choices[0].message.content)
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ._config import get_provider_config
from ._http import async_post, async_stream, sync_post, sync_stream
from ._router import parse_model_string, resolve, get_adapter
from ._types import EmbeddingResponse, ModelResponse, StreamChunk
from .exceptions import (
    APIError,
    AuthenticationError,
    ContextWindowExceededError,
    InvalidRequestError,
    NanoLLMException,
    RateLimitError,
    ServiceUnavailableError,
)

__version__ = "0.1.0"

# Module-level configuration (litellm compatibility)
drop_params: bool = True
set_verbose: bool = False


def _resolve_api_key(
    api_key: str | None,
    api_key_env: str | None,
) -> str | None:
    """Resolve API key: explicit param > environment variable."""
    if api_key:
        return api_key
    if api_key_env:
        return os.environ.get(api_key_env)
    return None


class _SyncStreamIterator:
    """Wraps SSE stream into an iterator of StreamChunk objects."""

    def __init__(self, stream: Iterator[str], adapter: Any, model: str):
        self._stream = stream
        self._adapter = adapter
        self._model = model

    def __iter__(self) -> Iterator[StreamChunk]:
        return self

    def __next__(self) -> StreamChunk:
        while True:
            line = next(self._stream)  # Raises StopIteration when done
            chunk = self._adapter.parse_stream_chunk(line, self._model)
            if chunk is not None:
                return chunk


class _AsyncStreamIterator:
    """Wraps async SSE stream into an async iterator of StreamChunk objects."""

    def __init__(self, stream: AsyncIterator[str], adapter: Any, model: str):
        self._stream = stream
        self._adapter = adapter
        self._model = model

    def __aiter__(self) -> AsyncIterator[StreamChunk]:
        return self

    async def __anext__(self) -> StreamChunk:
        while True:
            try:
                line = await self._stream.__anext__()
            except StopAsyncIteration:
                raise
            chunk = self._adapter.parse_stream_chunk(line, self._model)
            if chunk is not None:
                return chunk


def completion(
    model: str,
    messages: list[dict] | None = None,
    *,
    stream: bool | None = None,
    timeout: float | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    api_base: str | None = None,
    **kwargs: Any,
) -> ModelResponse | _SyncStreamIterator:
    """Synchronous completion call.

    Args:
        model: Provider/model string (e.g., "openai/gpt-4o")
        messages: List of message dicts with "role" and "content"
        stream: Whether to stream the response
        timeout: Request timeout in seconds
        api_key: API key (overrides env var)
        base_url: Base URL (overrides provider default)
        api_base: Alias for base_url (litellm compat)
        **kwargs: Additional params (temperature, response_format, etc.)

    Returns:
        ModelResponse for non-streaming, iterator of chunk dicts for streaming
    """
    messages = messages or []
    effective_base_url = base_url or api_base
    effective_stream = bool(stream)
    effective_timeout = timeout or 600.0

    provider, model_id, adapter, config = resolve(model)
    effective_api_key = _resolve_api_key(api_key, config.api_key_env)

    # Filter params if drop_params is enabled
    if drop_params:
        kwargs = adapter.filter_params(kwargs, config.supported_params)

    url, headers, body = adapter.build_request(
        model=model_id,
        messages=messages,
        api_key=effective_api_key,
        base_url=effective_base_url or config.base_url,
        stream=effective_stream,
        provider_config=config,
        **kwargs,
    )

    if effective_stream:
        raw_stream = sync_stream(
            url, headers, body,
            timeout=effective_timeout,
            provider=provider,
            model=model,
        )
        return _SyncStreamIterator(raw_stream, adapter, model_id)

    data = sync_post(
        url, headers, body,
        timeout=effective_timeout,
        provider=provider,
        model=model,
    )
    return adapter.parse_response(data, model_id)


async def acompletion(
    model: str,
    messages: list[dict] | None = None,
    *,
    stream: bool | None = None,
    timeout: float | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    api_base: str | None = None,
    **kwargs: Any,
) -> ModelResponse | _AsyncStreamIterator:
    """Asynchronous completion call.

    Same parameters as completion(), but returns an awaitable.
    """
    messages = messages or []
    effective_base_url = base_url or api_base
    effective_stream = bool(stream)
    effective_timeout = timeout or 600.0

    provider, model_id, adapter, config = resolve(model)
    effective_api_key = _resolve_api_key(api_key, config.api_key_env)

    if drop_params:
        kwargs = adapter.filter_params(kwargs, config.supported_params)

    url, headers, body = adapter.build_request(
        model=model_id,
        messages=messages,
        api_key=effective_api_key,
        base_url=effective_base_url or config.base_url,
        stream=effective_stream,
        provider_config=config,
        **kwargs,
    )

    if effective_stream:
        raw_stream = async_stream(
            url, headers, body,
            timeout=effective_timeout,
            provider=provider,
            model=model,
        )
        return _AsyncStreamIterator(raw_stream, adapter, model_id)

    data = await async_post(
        url, headers, body,
        timeout=effective_timeout,
        provider=provider,
        model=model,
    )
    return adapter.parse_response(data, model_id)


def batch_completion(
    model: str,
    messages: list[list[dict]],
    *,
    timeout: float | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    max_workers: int = 100,
    logger_fn: Any = None,
    **kwargs: Any,
) -> list[ModelResponse]:
    """Parallel batch completion using thread pool.

    Args:
        model: Provider/model string
        messages: List of message lists (each is a separate completion call)
        timeout: Request timeout per call
        max_workers: Max concurrent threads (default 100)
        logger_fn: Unused, kept for litellm compat
        **kwargs: Additional params passed to each completion call

    Returns:
        List of ModelResponse objects
    """
    def _single_completion(msgs: list[dict]) -> ModelResponse:
        return completion(
            model=model,
            messages=msgs,
            timeout=timeout,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

    with ThreadPoolExecutor(max_workers=min(max_workers, len(messages))) as executor:
        results = list(executor.map(_single_completion, messages))

    return results


async def aembedding(
    model: str,
    input: list[str] | str,
    *,
    timeout: float | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> EmbeddingResponse:
    """Asynchronous embedding call.

    Args:
        model: Provider/model string (e.g., "openai/text-embedding-3-small")
        input: Text or list of texts to embed
        timeout: Request timeout
        api_key: API key (overrides env var)
        api_base: Base URL (litellm compat alias)
        base_url: Base URL (overrides provider default)
        **kwargs: Additional params (dimensions, etc.)

    Returns:
        EmbeddingResponse with .data list of embedding dicts
    """
    if isinstance(input, str):
        input = [input]

    effective_base_url = base_url or api_base
    effective_timeout = timeout or 600.0

    provider, model_id, adapter, config = resolve(model)
    effective_api_key = _resolve_api_key(api_key, config.api_key_env)

    url, headers, body = adapter.build_embedding_request(
        model=model_id,
        input=input,
        api_key=effective_api_key,
        base_url=effective_base_url or config.base_url,
        provider_config=config,
        **kwargs,
    )

    data = await async_post(
        url, headers, body,
        timeout=effective_timeout,
        provider=provider,
        model=model,
    )
    return adapter.parse_embedding_response(data, model_id)
