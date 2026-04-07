"""Azure OpenAI adapter for NanoLLM.

Extends OpenAI-compatible adapter with Azure-specific URL construction
and authentication (api-key header instead of Bearer token).
"""

from __future__ import annotations

from typing import Any

from .._config import ProviderConfig
from .openai_compat import Adapter as OpenAICompatAdapter


class Adapter(OpenAICompatAdapter):
    """Adapter for Azure OpenAI deployments."""

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
        api_version = kwargs.pop("api_version", "2024-02-01")

        if base_url:
            # User provided full base URL — append completions path
            url = f"{base_url.rstrip('/')}/chat/completions?api-version={api_version}"
        else:
            # Construct from components
            # Model format: "azure/deployment-name" or just "deployment-name"
            # base_url should be: https://{resource}.openai.azure.com/openai/deployments/{deployment}
            raise ValueError(
                "Azure OpenAI requires 'base_url' parameter. "
                "Format: https://<resource>.openai.azure.com/openai/deployments/<deployment>"
            )

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["api-key"] = api_key

        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if stream:
            body["stream"] = True

        for key, value in kwargs.items():
            if value is not None:
                body[key] = value

        return url, headers, body

    def build_embedding_request(
        self,
        model: str,
        input: list[str],
        api_key: str | None = None,
        base_url: str | None = None,
        provider_config: ProviderConfig | None = None,
        **kwargs: Any,
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        api_version = kwargs.pop("api_version", "2024-02-01")

        if base_url:
            url = f"{base_url.rstrip('/')}/embeddings?api-version={api_version}"
        else:
            raise ValueError(
                "Azure OpenAI requires 'base_url' parameter for embeddings."
            )

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["api-key"] = api_key

        body: dict[str, Any] = {
            "model": model,
            "input": input,
        }

        return url, headers, body
