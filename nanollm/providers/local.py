"""Local model providers: Ollama, LM Studio, vLLM, and Text Gen WebUI.

All use the OpenAI-compatible format with local defaults (localhost URLs,
no API key required).
"""

from __future__ import annotations

from typing import Any, Optional

from . import register
from .openai import OpenAIProvider


@register("ollama")
class OllamaProvider(OpenAIProvider):
    """Ollama local inference provider.

    Default: http://localhost:11434/v1, no auth required.
    """

    name = "ollama"
    base_url = "http://localhost:11434/v1"
    api_key_env = "OLLAMA_API_KEY"  # Usually not needed

    supported_params = frozenset({
        "temperature", "top_p", "top_k", "stop", "max_tokens",
        "max_completion_tokens", "response_format", "tools", "tool_choice",
        "seed", "num_predict", "num_ctx", "repeat_penalty",
    })

    def get_api_key(self, api_key: Optional[str] = None) -> str:
        """Ollama typically doesn't need an API key."""
        return api_key or super().get_api_key(api_key) or "ollama"

    def build_headers(self, api_key: str) -> dict[str, str]:
        """Minimal headers for local server."""
        return {"Content-Type": "application/json"}

    def map_params(self, **kwargs: Any) -> dict:
        """Map Ollama-specific parameters."""
        result = {}
        for k, v in kwargs.items():
            if k == "num_predict":
                result["num_predict"] = v
            elif k == "num_ctx":
                result["num_ctx"] = v
            elif k == "repeat_penalty":
                result["repeat_penalty"] = v
            else:
                result[k] = v
        return result


@register("ollama_chat")
class OllamaChatProvider(OllamaProvider):
    """Ollama native chat API (non-OpenAI-compatible endpoint)."""

    name = "ollama_chat"

    def build_url(self, model: str, endpoint: str = "chat/completions",
                  base_url: Optional[str] = None, **kwargs: Any) -> str:
        base = (base_url or "http://localhost:11434").rstrip("/")
        # Use the OpenAI-compatible endpoint
        return f"{base}/v1/{endpoint}"


@register("lm_studio")
class LMStudioProvider(OpenAIProvider):
    """LM Studio local inference provider.

    Default: http://localhost:1234/v1, no auth required.
    """

    name = "lm_studio"
    base_url = "http://localhost:1234/v1"
    api_key_env = "LM_STUDIO_API_KEY"

    def get_api_key(self, api_key: Optional[str] = None) -> str:
        """LM Studio typically doesn't need an API key."""
        return api_key or super().get_api_key(api_key) or "lm-studio"

    def build_headers(self, api_key: str) -> dict[str, str]:
        """Minimal headers for local server."""
        return {"Content-Type": "application/json"}


@register("vllm")
class VLLMProvider(OpenAIProvider):
    """vLLM OpenAI-compatible server."""

    name = "vllm"
    base_url = "http://localhost:8000/v1"
    api_key_env = "VLLM_API_KEY"

    supported_params = frozenset({
        "temperature", "top_p", "top_k", "stop", "max_tokens",
        "max_completion_tokens", "response_format", "tools", "tool_choice",
        "seed", "presence_penalty", "frequency_penalty", "best_of",
        "use_beam_search", "logprobs", "top_logprobs",
    })

    def get_api_key(self, api_key: Optional[str] = None) -> str:
        return api_key or super().get_api_key(api_key) or "vllm"

    def build_headers(self, api_key: str) -> dict[str, str]:
        return {"Content-Type": "application/json"}


@register("text_gen_webui")
class TextGenWebUIProvider(OpenAIProvider):
    """Text Generation WebUI (oobabooga) provider."""

    name = "text_gen_webui"
    base_url = "http://localhost:5000/v1"
    api_key_env = ""

    def get_api_key(self, api_key: Optional[str] = None) -> str:
        return api_key or "none"

    def build_headers(self, api_key: str) -> dict[str, str]:
        return {"Content-Type": "application/json"}
