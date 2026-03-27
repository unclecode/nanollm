"""Azure OpenAI provider.

Extends the OpenAI provider with Azure-specific auth (api-key header instead
of Bearer token) and URL structure (deployment-based URLs with api-version).

URL: {base_url}/openai/deployments/{model}/chat/completions?api-version=2025-02-01-preview
Auth: api-key header (not Bearer)
Otherwise same as OpenAI.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from . import register
from .openai import OpenAIProvider


@register("azure")
class AzureOpenAIProvider(OpenAIProvider):
    """Azure OpenAI Service provider."""

    name = "azure"
    base_url = ""  # Built from AZURE_API_BASE
    api_key_env = "AZURE_API_KEY"

    _api_base_env = "AZURE_API_BASE"
    _api_version_env = "AZURE_API_VERSION"
    _default_api_version = "2025-02-01-preview"

    def _get_api_base(self, base_url: Optional[str] = None) -> str:
        return base_url or os.environ.get(self._api_base_env, "")

    def _get_api_version(self, api_version: Optional[str] = None) -> str:
        return api_version or os.environ.get(
            self._api_version_env, self._default_api_version
        )

    # -- Headers -------------------------------------------------------------

    def build_headers(self, api_key: str) -> dict[str, str]:
        """Azure uses api-key header instead of Authorization: Bearer."""
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["api-key"] = api_key
        return headers

    # -- URL -----------------------------------------------------------------

    def build_url(self, model: str, endpoint: str = "chat/completions",
                  base_url: Optional[str] = None, **kwargs: Any) -> str:
        """Build Azure URL: {base}/openai/deployments/{model}/{endpoint}?api-version={version}."""
        api_base = self._get_api_base(base_url).rstrip("/")
        api_version = self._get_api_version(kwargs.get("api_version"))

        if not api_base:
            raise ValueError(
                "Azure requires base_url or AZURE_API_BASE env var. "
                "Example: https://your-resource.openai.azure.com"
            )

        return (f"{api_base}/openai/deployments/{model}/{endpoint}"
                f"?api-version={api_version}")

    def build_embedding_url(self, model: str,
                            base_url: Optional[str] = None) -> str:
        return self.build_url(model, endpoint="embeddings", base_url=base_url)

    # -- Body ----------------------------------------------------------------

    def build_body(self, model: str, messages: list, stream: bool = False,
                   **kwargs: Any) -> dict:
        """Build body -- same as OpenAI but exclude model (it's in the URL)."""
        kwargs.pop("api_version", None)  # Not a body param
        body = super().build_body(model, messages, stream, **kwargs)
        # Azure doesn't need model in the body since it's in the URL,
        # but it doesn't hurt to include it
        return body


@register("azure_ai")
class AzureAIProvider(OpenAIProvider):
    """Azure AI (serverless / model-as-a-service) provider.

    Uses standard Bearer token auth but with Azure AI endpoints.
    Different from Azure OpenAI -- this is for models deployed via
    Azure AI Studio (Llama, Mistral, etc.).
    """

    name = "azure_ai"
    api_key_env = "AZURE_AI_API_KEY"

    def build_url(self, model: str, endpoint: str = "chat/completions",
                  base_url: Optional[str] = None, **kwargs: Any) -> str:
        base = (base_url or "").rstrip("/")
        if not base:
            raise ValueError(
                "Azure AI requires base_url. "
                "Example: https://your-endpoint.region.inference.ai.azure.com/v1"
            )
        return f"{base}/{endpoint}"
