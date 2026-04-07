"""HTTP transport layer for NanoLLM.

Thin wrapper around httpx for sync/async POST and SSE streaming.
Maps HTTP error status codes to NanoLLM exceptions.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx

from .exceptions import raise_for_status

_DEFAULT_TIMEOUT = 600.0  # 10 minutes


def _parse_error_body(response: httpx.Response) -> dict | str:
    """Try to parse error response as JSON, fall back to text."""
    try:
        return response.json()
    except Exception:
        return response.text


def sync_post(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    timeout: float = _DEFAULT_TIMEOUT,
    provider: str | None = None,
    model: str | None = None,
) -> dict:
    """Synchronous JSON POST request."""
    with httpx.Client(timeout=timeout, http2=True) as client:
        response = client.post(url, headers=headers, json=body)
        if response.status_code >= 400:
            raise_for_status(
                response.status_code,
                _parse_error_body(response),
                provider=provider,
                model=model,
            )
        return response.json()


async def async_post(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    timeout: float = _DEFAULT_TIMEOUT,
    provider: str | None = None,
    model: str | None = None,
) -> dict:
    """Asynchronous JSON POST request."""
    async with httpx.AsyncClient(timeout=timeout, http2=True) as client:
        response = await client.post(url, headers=headers, json=body)
        if response.status_code >= 400:
            raise_for_status(
                response.status_code,
                _parse_error_body(response),
                provider=provider,
                model=model,
            )
        return response.json()


def sync_stream(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    timeout: float = _DEFAULT_TIMEOUT,
    provider: str | None = None,
    model: str | None = None,
) -> Iterator[str]:
    """Synchronous SSE streaming POST. Yields data lines (without 'data: ' prefix)."""
    with httpx.Client(timeout=timeout, http2=True) as client:
        with client.stream("POST", url, headers=headers, json=body) as response:
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


async def async_stream(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    timeout: float = _DEFAULT_TIMEOUT,
    provider: str | None = None,
    model: str | None = None,
) -> AsyncIterator[str]:
    """Asynchronous SSE streaming POST. Yields data lines (without 'data: ' prefix)."""
    async with httpx.AsyncClient(timeout=timeout, http2=True) as client:
        async with client.stream("POST", url, headers=headers, json=body) as response:
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
