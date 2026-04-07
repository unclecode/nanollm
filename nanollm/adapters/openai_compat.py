"""OpenAI-compatible adapter for NanoLLM.

Handles all providers that use the OpenAI chat completions format:
OpenAI, Groq, Together, Mistral, Deepseek, Perplexity, Fireworks,
OpenRouter, DeepInfra, Anyscale, xAI, Cerebras, and any custom
OpenAI-compatible endpoint.
"""

from __future__ import annotations

import json
from typing import Any

from .._config import ProviderConfig, get_provider_config
from .._types import (
    _AttrDict,
    Choice,
    EmbeddingData,
    EmbeddingResponse,
    Message,
    ModelResponse,
    Usage,
    make_stream_chunk,
)
from ._base import BaseAdapter


class Adapter(BaseAdapter):
    """Adapter for OpenAI-compatible chat completions API."""

    def build_request(
        self,
        model: str,
        messages: list[dict],
        api_key: str | None = None,
        base_url: str | None = None,
        stream: bool = False,
        provider_config: ProviderConfig | None = None,
        **kwargs: Any,
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        base = base_url or (provider_config.base_url if provider_config else "https://api.openai.com/v1")
        url = f"{base.rstrip('/')}/chat/completions"

        headers = {"Content-Type": "application/json"}
        if api_key:
            auth_header = provider_config.auth_header if provider_config else "Authorization"
            auth_prefix = provider_config.auth_prefix if provider_config else "Bearer"
            if auth_prefix:
                headers[auth_header] = f"{auth_prefix} {api_key}"
            else:
                headers[auth_header] = api_key
        if provider_config and provider_config.extra_headers:
            headers.update(provider_config.extra_headers)

        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if stream:
            body["stream"] = True

        # Add supported params
        for key, value in kwargs.items():
            if value is not None:
                body[key] = value

        return url, headers, body

    def parse_response(self, data: dict, model: str = "") -> ModelResponse:
        choices = []
        for c in data.get("choices", []):
            msg = c.get("message", {})
            choices.append(
                Choice(
                    message=Message(
                        content=msg.get("content"),
                        role=msg.get("role", "assistant"),
                        tool_calls=msg.get("tool_calls"),
                        function_call=msg.get("function_call"),
                    ),
                    index=c.get("index", 0),
                    finish_reason=c.get("finish_reason"),
                )
            )

        usage_data = data.get("usage") or {}
        ctd = usage_data.get("completion_tokens_details")
        ptd = usage_data.get("prompt_tokens_details")
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
            completion_tokens_details=_AttrDict(ctd) if ctd else None,
            prompt_tokens_details=_AttrDict(ptd) if ptd else None,
        )

        return ModelResponse(
            id=data.get("id", ""),
            choices=choices,
            model=data.get("model", model),
            usage=usage,
            created=data.get("created", 0),
        )

    def parse_stream_chunk(self, line: str, model: str = "") -> dict | None:
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return None

        choices = data.get("choices", [])
        if not choices:
            return None

        delta = choices[0].get("delta", {})
        return make_stream_chunk(
            content=delta.get("content"),
            role=delta.get("role"),
            finish_reason=choices[0].get("finish_reason"),
            model=data.get("model", model),
        )

    def build_embedding_request(
        self,
        model: str,
        input: list[str],
        api_key: str | None = None,
        base_url: str | None = None,
        provider_config: ProviderConfig | None = None,
        **kwargs: Any,
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        base = base_url or (provider_config.base_url if provider_config else "https://api.openai.com/v1")
        url = f"{base.rstrip('/')}/embeddings"

        headers = {"Content-Type": "application/json"}
        if api_key:
            auth_header = provider_config.auth_header if provider_config else "Authorization"
            auth_prefix = provider_config.auth_prefix if provider_config else "Bearer"
            if auth_prefix:
                headers[auth_header] = f"{auth_prefix} {api_key}"
            else:
                headers[auth_header] = api_key

        body: dict[str, Any] = {
            "model": model,
            "input": input,
        }
        # Pass through optional params like dimensions
        for key in ("dimensions", "encoding_format", "user"):
            if key in kwargs and kwargs[key] is not None:
                body[key] = kwargs[key]

        return url, headers, body

    def parse_embedding_response(
        self, data: dict, model: str = ""
    ) -> EmbeddingResponse:
        items = []
        for item in data.get("data", []):
            items.append(
                EmbeddingData(
                    embedding=item.get("embedding", []),
                    index=item.get("index", 0),
                )
            )

        usage_data = data.get("usage") or {}
        return EmbeddingResponse(
            data=items,
            model=data.get("model", model),
            usage=Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            ),
        )
