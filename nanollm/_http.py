"""HTTP transport layer for NanoLLM.

Thin wrapper around httpx for sync/async POST requests and SSE streaming.
All HTTP and connection errors are mapped to NanoLLM exception types so
callers never need to catch httpx-specific exceptions.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator, Iterator
from typing import Any, NoReturn

import httpx

from .exceptions import APIConnectionError, Timeout, raise_for_status

# 10-minute default -- long enough for large model responses
_DEFAULT_TIMEOUT = 600.0

# Module-level sync client with connection pooling (thread-safe per httpx docs)
_sync_client: httpx.Client | None = None
_sync_client_lock = threading.Lock()


def _get_sync_client() -> httpx.Client:
    global _sync_client
    if _sync_client is None or _sync_client.is_closed:
        with _sync_client_lock:
            if _sync_client is None or _sync_client.is_closed:
                _sync_client = httpx.Client(http2=True)
    return _sync_client


# ── Helpers ──────────────────────────────────────────────────────────


def _parse_error_body(response: httpx.Response) -> dict | str:
    """Try to parse the error response as JSON; fall back to raw text."""
    try:
        return response.json()
    except (json.JSONDecodeError, ValueError):
        return response.text


def _wrap_connection_error(
    exc: Exception,
    provider: str | None,
    model: str | None,
) -> NoReturn:
    """Re-raise an httpx error as the appropriate NanoLLM exception."""
    if isinstance(exc, httpx.TimeoutException):
        raise Timeout(
            message=f"Request timed out: {exc}",
            llm_provider=provider,
            model=model,
        ) from exc
    if isinstance(exc, httpx.ConnectError):
        raise APIConnectionError(
            message=f"Connection error: {exc}",
            llm_provider=provider,
            model=model,
        ) from exc
    raise APIConnectionError(
        message=f"HTTP error: {exc}",
        llm_provider=provider,
        model=model,
    ) from exc


def _check_response(
    response: httpx.Response,
    provider: str | None,
    model: str | None,
) -> None:
    """Raise for non-2xx status codes."""
    if response.status_code >= 400:
        raise_for_status(
            response.status_code,
            _parse_error_body(response),
            provider=provider,
            model=model,
        )


# ── Sync transport ───────────────────────────────────────────────────


def sync_post(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any] | bytes,
    timeout: float = _DEFAULT_TIMEOUT,
    provider: str | None = None,
    model: str | None = None,
) -> dict:
    """Synchronous JSON POST request.  Returns the parsed JSON body.

    If *body* is bytes, send it as raw content (preserves exact bytes for
    request signing).  If it is a dict, let httpx serialize it.
    """
    try:
        client = _get_sync_client()
        if isinstance(body, bytes):
            response = client.post(url, headers=headers, content=body, timeout=timeout)
        else:
            response = client.post(url, headers=headers, json=body, timeout=timeout)
        _check_response(response, provider, model)
        return response.json()
    except (httpx.HTTPError, httpx.StreamError) as exc:
        _wrap_connection_error(exc, provider, model)


# Module-level async client (no lock needed — async is single-threaded per loop)
_async_client: httpx.AsyncClient | None = None
_async_client_loop_id: int | None = None


def _get_async_client() -> httpx.AsyncClient:
    global _async_client, _async_client_loop_id
    try:
        current_loop_id = id(asyncio.get_running_loop())
    except RuntimeError:
        current_loop_id = None
    if _async_client is None or _async_client.is_closed or _async_client_loop_id != current_loop_id:
        _async_client = httpx.AsyncClient(http2=True)
        _async_client_loop_id = current_loop_id
    return _async_client


async def async_post(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any] | bytes,
    timeout: float = _DEFAULT_TIMEOUT,
    provider: str | None = None,
    model: str | None = None,
) -> dict:
    """Asynchronous JSON POST request.  Returns the parsed JSON body.

    If *body* is bytes, send it as raw content (preserves exact bytes for
    request signing).  If it is a dict, let httpx serialize it.
    """
    try:
        client = _get_async_client()
        if isinstance(body, bytes):
            response = await client.post(url, headers=headers, content=body, timeout=timeout)
        else:
            response = await client.post(url, headers=headers, json=body, timeout=timeout)
        _check_response(response, provider, model)
        return response.json()
    except (httpx.HTTPError, httpx.StreamError) as exc:
        _wrap_connection_error(exc, provider, model)


# ── Sync streaming ───────────────────────────────────────────────────


def sync_stream(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any] | bytes,
    timeout: float = _DEFAULT_TIMEOUT,
    provider: str | None = None,
    model: str | None = None,
) -> Iterator[str]:
    """Synchronous SSE streaming POST.  Yields ``data:`` payloads."""
    try:
        client = _get_sync_client()
        kwargs = {"content": body} if isinstance(body, bytes) else {"json": body}
        with client.stream("POST", url, headers=headers, timeout=timeout, **kwargs) as response:
            if response.status_code >= 400:
                response.read()
                raise_for_status(
                    response.status_code,
                    _parse_error_body(response),
                    provider=provider,
                    model=model,
                )
            for line in response.iter_lines():
                line = line.strip()
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        return
                    yield data
    except (httpx.HTTPError, httpx.StreamError) as exc:
        _wrap_connection_error(exc, provider, model)


async def async_stream(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any] | bytes,
    timeout: float = _DEFAULT_TIMEOUT,
    provider: str | None = None,
    model: str | None = None,
) -> AsyncIterator[str]:
    """Asynchronous SSE streaming POST.  Yields ``data:`` payloads."""
    try:
        client = _get_async_client()
        kwargs = {"content": body} if isinstance(body, bytes) else {"json": body}
        async with client.stream(
            "POST", url, headers=headers, timeout=timeout, **kwargs
        ) as response:
            if response.status_code >= 400:
                await response.aread()
                raise_for_status(
                    response.status_code,
                    _parse_error_body(response),
                    provider=provider,
                    model=model,
                )
            async for line in response.aiter_lines():
                line = line.strip()
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        return
                    yield data
    except (httpx.HTTPError, httpx.StreamError) as exc:
        _wrap_connection_error(exc, provider, model)
