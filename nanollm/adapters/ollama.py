"""Ollama adapter for NanoLLM.

Extends OpenAI-compatible adapter with Ollama-specific defaults:
- Local endpoint (http://localhost:11434/v1)
- No authentication required
"""

from __future__ import annotations

from typing import Any

from .._config import ProviderConfig
from .openai_compat import Adapter as OpenAICompatAdapter


class Adapter(OpenAICompatAdapter):
    """Adapter for Ollama (OpenAI-compatible local inference)."""

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
        # Default to localhost, no auth needed
        effective_base = base_url or (
            provider_config.base_url if provider_config
            else "http://localhost:11434/v1"
        )
        url, headers, body = super().build_request(
            model=model,
            messages=messages,
            api_key=None,  # Ollama doesn't use auth
            base_url=effective_base,
            stream=stream,
            provider_config=provider_config,
            **kwargs,
        )
        # Remove any auth headers
        headers.pop("Authorization", None)
        headers.pop("api-key", None)
        return url, headers, body
