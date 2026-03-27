"""NanoLLM client -- the primary way to use NanoLLM.

Wraps provider registry + HTTP transport into a clean client object with
built-in retry (exponential backoff), streaming support, and convenience
methods for common tasks (JSON extraction, vision).
"""

from __future__ import annotations

import json
import random
import time
import asyncio
from collections.abc import AsyncIterator, Iterator
from typing import Any, Generator, AsyncGenerator, Optional

from .exceptions import NanoLLMException, RetryableError, Timeout
from ._http import sync_post, async_post, sync_stream, async_stream
from ._image import (
    parse_data_uri,
    guess_mime_from_url,
    download_image_as_base64,
)
from .providers import get_provider
from .providers.base import BaseProvider
from ._structured import extract_json
from ._types import (
    ModelResponse,
    EmbeddingResponse,
    stream_chunk_builder,
)


def _parse_model_string(model: str) -> tuple[str, str]:
    """Parse 'provider/model' into (provider_name, model_name).

    Examples:
        'openai/gpt-4o'       -> ('openai', 'gpt-4o')
        'anthropic/claude-3'  -> ('anthropic', 'claude-3')
        'gpt-4o'              -> ('openai', 'gpt-4o')  (default)
        'claude-3-opus'       -> ('anthropic', 'claude-3-opus')
        'gemini-pro'          -> ('gemini', 'gemini-pro')
    """
    if "/" in model:
        provider, model_name = model.split("/", 1)
        return provider, model_name

    # Auto-detect provider from model name
    model_lower = model.lower()
    if model_lower.startswith("claude"):
        return "anthropic", model
    if model_lower.startswith("gemini"):
        return "gemini", model
    if (
        model_lower.startswith("gpt")
        or model_lower.startswith("o1")
        or model_lower.startswith("o3")
        or model_lower.startswith("o4")
    ):
        return "openai", model
    if "llama" in model_lower or "mixtral" in model_lower:
        return "groq", model
    if model_lower.startswith("mistral"):
        return "mistral", model
    if model_lower.startswith("deepseek"):
        return "deepseek", model
    if model_lower.startswith("grok"):
        return "xai", model

    # Default to openai
    return "openai", model


class NanoLLM:
    """Main NanoLLM client.

    Usage::

        client = NanoLLM()

        # Simple completion
        response = client.complete("openai/gpt-4o", messages=[
            {"role": "user", "content": "Hello!"}
        ])
        print(response.choices[0].message.content)

        # Streaming
        for chunk in client.stream("anthropic/claude-3-5-sonnet-20241022",
                                   messages=[{"role": "user", "content": "Hi"}]):
            print(chunk.choices[0].delta.content, end="")

        # JSON extraction
        data = client.json("openai/gpt-4o", messages=[...], schema={...})

        # Vision
        response = client.vision("openai/gpt-4o", "Describe this image",
                                 images=["https://example.com/cat.jpg"])
    """

    def __init__(
        self,
        default_model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 600,
        drop_params: bool = True,
    ):
        self.default_model = default_model
        self.api_key = api_key
        self.base_url = base_url
        self.max_retries = max_retries
        self.timeout = timeout
        self.drop_params = drop_params

    def _resolve_model(self, model: Optional[str]) -> str:
        m = model or self.default_model
        if not m:
            raise ValueError(
                "No model specified. Pass model= or set default_model on the client."
            )
        return m

    def _get_provider(self, provider_name: str) -> BaseProvider:
        return get_provider(provider_name)

    # -- Sync completion -----------------------------------------------------

    def complete(
        self,
        model: Optional[str] = None,
        messages: Optional[list] = None,
        *,
        stream: bool = False,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Synchronous chat completion with built-in retry.

        Args:
            model: Model string, e.g. "openai/gpt-4o" or "claude-3-5-sonnet".
            messages: List of message dicts (OpenAI format).
            stream: Whether to stream (if True, use ``stream()`` instead).
            timeout: Request timeout in seconds.
            api_key: API key (overrides client default and env var).
            base_url: Base URL (overrides client default).
            **kwargs: Additional params (temperature, max_tokens, tools, etc.)

        Returns:
            ModelResponse with choices, usage, etc.
        """
        if stream:
            # Collect all chunks and merge
            chunks = list(self.stream(
                model=model, messages=messages,
                timeout=timeout, api_key=api_key, base_url=base_url,
                **kwargs,
            ))
            return stream_chunk_builder(chunks)

        model_str = self._resolve_model(model)
        messages = messages or []
        provider_name, model_name = _parse_model_string(model_str)
        provider = self._get_provider(provider_name)

        effective_api_key = provider.get_api_key(api_key or self.api_key)
        effective_base_url = base_url or self.base_url
        effective_timeout = timeout or self.timeout

        # Build request
        url = provider.build_url(
            model_name, base_url=effective_base_url, stream=False,
            api_key=effective_api_key, **kwargs,
        )
        body = provider.build_body(model_name, messages, stream=False, **kwargs)

        if self.drop_params:
            body = {k: v for k, v in body.items() if v is not None}

        # Handle Bedrock SigV4 signing
        if hasattr(provider, "build_signed_headers"):
            body_bytes = json.dumps(body).encode("utf-8")
            headers = provider.build_signed_headers(url, body_bytes, **kwargs)
        else:
            headers = provider.build_headers(effective_api_key)

        # Request with retry
        data = self._retry_sync(
            lambda: sync_post(
                url, headers, body,
                timeout=effective_timeout,
                provider=provider_name,
                model=model_str,
            ),
            provider_name=provider_name,
            model_name=model_name,
        )

        return provider.parse_response(data, model=model_name)

    # -- Async completion ----------------------------------------------------

    async def acomplete(
        self,
        model: Optional[str] = None,
        messages: Optional[list] = None,
        *,
        stream: bool = False,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Async chat completion with built-in retry."""
        if stream:
            chunks = []
            async for chunk in self.astream(
                model=model, messages=messages,
                timeout=timeout, api_key=api_key, base_url=base_url,
                **kwargs,
            ):
                chunks.append(chunk)
            return stream_chunk_builder(chunks)

        model_str = self._resolve_model(model)
        messages = messages or []
        provider_name, model_name = _parse_model_string(model_str)
        provider = self._get_provider(provider_name)

        effective_api_key = provider.get_api_key(api_key or self.api_key)
        effective_base_url = base_url or self.base_url
        effective_timeout = timeout or self.timeout

        url = provider.build_url(
            model_name, base_url=effective_base_url, stream=False,
            api_key=effective_api_key, **kwargs,
        )
        body = provider.build_body(model_name, messages, stream=False, **kwargs)

        if self.drop_params:
            body = {k: v for k, v in body.items() if v is not None}

        if hasattr(provider, "build_signed_headers"):
            body_bytes = json.dumps(body).encode("utf-8")
            headers = provider.build_signed_headers(url, body_bytes, **kwargs)
        else:
            headers = provider.build_headers(effective_api_key)

        data = await self._retry_async(
            lambda: async_post(
                url, headers, body,
                timeout=effective_timeout,
                provider=provider_name,
                model=model_str,
            ),
            provider_name=provider_name,
            model_name=model_name,
        )

        return provider.parse_response(data, model=model_name)

    # -- Sync streaming ------------------------------------------------------

    def stream(
        self,
        model: Optional[str] = None,
        messages: Optional[list] = None,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> Generator[ModelResponse, None, None]:
        """Synchronous streaming completion.

        Yields ModelResponse chunks with delta content.
        """
        model_str = self._resolve_model(model)
        messages = messages or []
        provider_name, model_name = _parse_model_string(model_str)
        provider = self._get_provider(provider_name)

        effective_api_key = provider.get_api_key(api_key or self.api_key)
        effective_base_url = base_url or self.base_url
        effective_timeout = timeout or self.timeout

        url = provider.build_url(
            model_name, base_url=effective_base_url, stream=True,
            api_key=effective_api_key, **kwargs,
        )
        body = provider.build_body(model_name, messages, stream=True, **kwargs)

        if self.drop_params:
            body = {k: v for k, v in body.items() if v is not None}

        if hasattr(provider, "build_signed_headers"):
            body_bytes = json.dumps(body).encode("utf-8")
            headers = provider.build_signed_headers(url, body_bytes, **kwargs)
        else:
            headers = provider.build_headers(effective_api_key)

        for line in sync_stream(
            url, headers, body,
            timeout=effective_timeout,
            provider=provider_name,
            model=model_str,
        ):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            chunk = provider.parse_stream_line(data, model=model_name)
            if chunk is not None:
                yield chunk

    # -- Async streaming -----------------------------------------------------

    async def astream(
        self,
        model: Optional[str] = None,
        messages: Optional[list] = None,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[ModelResponse, None]:
        """Async streaming completion.

        Yields ModelResponse chunks with delta content.
        """
        model_str = self._resolve_model(model)
        messages = messages or []
        provider_name, model_name = _parse_model_string(model_str)
        provider = self._get_provider(provider_name)

        effective_api_key = provider.get_api_key(api_key or self.api_key)
        effective_base_url = base_url or self.base_url
        effective_timeout = timeout or self.timeout

        url = provider.build_url(
            model_name, base_url=effective_base_url, stream=True,
            api_key=effective_api_key, **kwargs,
        )
        body = provider.build_body(model_name, messages, stream=True, **kwargs)

        if self.drop_params:
            body = {k: v for k, v in body.items() if v is not None}

        if hasattr(provider, "build_signed_headers"):
            body_bytes = json.dumps(body).encode("utf-8")
            headers = provider.build_signed_headers(url, body_bytes, **kwargs)
        else:
            headers = provider.build_headers(effective_api_key)

        async for line in async_stream(
            url, headers, body,
            timeout=effective_timeout,
            provider=provider_name,
            model=model_str,
        ):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            chunk = provider.parse_stream_line(data, model=model_name)
            if chunk is not None:
                yield chunk

    # -- Embeddings ----------------------------------------------------------

    def embed(
        self,
        model: Optional[str] = None,
        input: Optional[list | str] = None,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """Synchronous embedding request."""
        model_str = self._resolve_model(model)
        provider_name, model_name = _parse_model_string(model_str)
        provider = self._get_provider(provider_name)

        effective_api_key = provider.get_api_key(api_key or self.api_key)
        effective_base_url = base_url or self.base_url
        effective_timeout = timeout or self.timeout

        url = provider.build_embedding_url(model_name, base_url=effective_base_url)
        headers = provider.build_headers(effective_api_key)
        body = provider.build_embedding_body(model_name, input or [], **kwargs)

        data = self._retry_sync(
            lambda: sync_post(
                url, headers, body,
                timeout=effective_timeout,
                provider=provider_name,
                model=model_str,
            ),
            provider_name=provider_name,
            model_name=model_name,
        )

        return provider.parse_embedding_response(data, model=model_name)

    async def aembed(
        self,
        model: Optional[str] = None,
        input: Optional[list | str] = None,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """Async embedding request."""
        model_str = self._resolve_model(model)
        provider_name, model_name = _parse_model_string(model_str)
        provider = self._get_provider(provider_name)

        effective_api_key = provider.get_api_key(api_key or self.api_key)
        effective_base_url = base_url or self.base_url
        effective_timeout = timeout or self.timeout

        url = provider.build_embedding_url(model_name, base_url=effective_base_url)
        headers = provider.build_headers(effective_api_key)
        body = provider.build_embedding_body(model_name, input or [], **kwargs)

        data = await self._retry_async(
            lambda: async_post(
                url, headers, body,
                timeout=effective_timeout,
                provider=provider_name,
                model=model_str,
            ),
            provider_name=provider_name,
            model_name=model_name,
        )

        return provider.parse_embedding_response(data, model=model_name)

    # -- Convenience: structured JSON ----------------------------------------

    def json(
        self,
        model: Optional[str] = None,
        messages: Optional[list] = None,
        schema: Optional[dict] = None,
        **kwargs: Any,
    ) -> dict:
        """Complete and extract structured JSON from the response.

        If *schema* is provided, passes it to the provider's structured
        output mechanism.  Falls back to parsing JSON from the response text.
        """
        if schema:
            kwargs["json_schema"] = schema

        response = self.complete(model=model, messages=messages, **kwargs)
        content = response.choices[0].message.content

        # If model used tool_use for structured output (Anthropic pattern)
        if response.choices[0].message.tool_calls:
            tc = response.choices[0].message.tool_calls[0]
            try:
                args = tc.get("function", {}).get("arguments", "")
                if isinstance(args, str):
                    return json.loads(args)
                return args
            except (json.JSONDecodeError, AttributeError, TypeError):
                pass

        if content:
            return extract_json(content)

        return {}

    async def ajson(
        self,
        model: Optional[str] = None,
        messages: Optional[list] = None,
        schema: Optional[dict] = None,
        **kwargs: Any,
    ) -> dict:
        """Async version of json()."""
        if schema:
            kwargs["json_schema"] = schema

        response = await self.acomplete(model=model, messages=messages, **kwargs)
        content = response.choices[0].message.content

        if response.choices[0].message.tool_calls:
            tc = response.choices[0].message.tool_calls[0]
            try:
                args = tc.get("function", {}).get("arguments", "")
                if isinstance(args, str):
                    return json.loads(args)
                return args
            except (json.JSONDecodeError, AttributeError, TypeError):
                pass

        if content:
            return extract_json(content)

        return {}

    # -- Convenience: vision -------------------------------------------------

    def vision(
        self,
        model: Optional[str] = None,
        prompt: str = "Describe this image.",
        images: Optional[list[str]] = None,
        detail: str = "auto",
        **kwargs: Any,
    ) -> ModelResponse:
        """Convenience method for vision requests.

        Builds content blocks with text + images automatically.

        Args:
            model: Model string.
            prompt: Text prompt.
            images: List of image sources (URLs, file paths, data URIs).
            detail: Image detail level ("auto", "low", "high").
            **kwargs: Additional completion params.
        """
        content: list[dict] = [{"type": "text", "text": prompt}]

        for img in images or []:
            content.append(_to_openai_image_block(img, detail=detail))

        messages = kwargs.pop("messages", None) or []
        messages = list(messages) + [{"role": "user", "content": content}]

        return self.complete(model=model, messages=messages, **kwargs)

    async def avision(
        self,
        model: Optional[str] = None,
        prompt: str = "Describe this image.",
        images: Optional[list[str]] = None,
        detail: str = "auto",
        **kwargs: Any,
    ) -> ModelResponse:
        """Async version of vision()."""
        content: list[dict] = [{"type": "text", "text": prompt}]

        for img in images or []:
            content.append(_to_openai_image_block(img, detail=detail))

        messages = kwargs.pop("messages", None) or []
        messages = list(messages) + [{"role": "user", "content": content}]

        return await self.acomplete(model=model, messages=messages, **kwargs)

    # -- Retry logic ---------------------------------------------------------

    def _retry_sync(
        self, fn, provider_name: str = "", model_name: str = "",
    ) -> Any:
        """Execute fn() with exponential backoff retry on retryable errors."""
        last_exc: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                return fn()
            except NanoLLMException as exc:
                last_exc = exc
                exc.llm_provider = exc.llm_provider or provider_name
                exc.model = exc.model or model_name

                if not isinstance(exc, RetryableError):
                    raise

                if attempt >= self.max_retries:
                    raise

                wait = _backoff_delay(attempt)
                time.sleep(wait)

        raise last_exc  # type: ignore[misc]

    async def _retry_async(
        self, fn, provider_name: str = "", model_name: str = "",
    ) -> Any:
        """Execute async fn() with exponential backoff retry."""
        last_exc: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                return await fn()
            except NanoLLMException as exc:
                last_exc = exc
                exc.llm_provider = exc.llm_provider or provider_name
                exc.model = exc.model or model_name

                if not isinstance(exc, RetryableError):
                    raise

                if attempt >= self.max_retries:
                    raise

                wait = _backoff_delay(attempt)
                await asyncio.sleep(wait)

        raise last_exc  # type: ignore[misc]


# -- Helpers -----------------------------------------------------------------


def _backoff_delay(
    attempt: int, base: float = 0.5, max_delay: float = 60.0,
) -> float:
    """Calculate exponential backoff delay with jitter.

    delay = min(base * 2^attempt + jitter, max_delay)
    """
    delay = base * (2 ** attempt)
    jitter = random.random() * delay * 0.5
    delay = min(delay + jitter, max_delay)
    return delay


def _to_openai_image_block(source: str, detail: str = "auto") -> dict:
    """Build an OpenAI-format image_url content block.

    Accepts HTTP(S) URLs, data URIs, and local file paths.
    Local files and non-HTTPS URLs are converted to data URIs.
    """
    if source.startswith(("http://", "https://")):
        url = source
    else:
        # data URI or local file -- convert via _image utilities
        parsed = parse_data_uri(source)
        if parsed:
            url = source  # already a data URI
        else:
            # Local file or raw base64 -- download/encode
            import base64
            import os

            if os.path.isfile(source):
                mime = guess_mime_from_url(source)
                with open(source, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("ascii")
                url = f"data:{mime};base64,{b64}"
            else:
                # Assume raw base64
                url = f"data:image/png;base64,{source}"

    return {
        "type": "image_url",
        "image_url": {"url": url, "detail": detail},
    }
