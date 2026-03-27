"""OpenAI provider and all OpenAI-compatible providers.

The OpenAIProvider class handles standard OpenAI chat/completions.
Messages are passed through as-is since OpenAI format is the native format
(including multimodal image_url content blocks).

OpenAI-compatible providers (Groq, Together, Mistral, etc.) are created
via _make_openai_compat() which generates subclasses with different
base_url and api_key_env.
"""

from __future__ import annotations

from typing import Any, Optional

from . import register
from .base import BaseProvider


@register("openai")
class OpenAIProvider(BaseProvider):
    """OpenAI API provider -- the baseline format."""

    name = "openai"
    base_url = "https://api.openai.com/v1"
    api_key_env = "OPENAI_API_KEY"

    supported_params = frozenset({
        "temperature", "top_p", "n", "stop", "max_tokens", "max_completion_tokens",
        "presence_penalty", "frequency_penalty", "logit_bias", "logprobs",
        "top_logprobs", "response_format", "seed", "tools", "tool_choice",
        "parallel_tool_calls", "user", "stream_options", "reasoning_effort",
        "prediction", "store", "metadata", "service_tier",
    })

    def build_body(self, model: str, messages: list, stream: bool = False,
                   **kwargs: Any) -> dict:
        """Build request body, handling OpenAI-specific features.

        Messages are passed through as-is (OpenAI format is native).
        Image_url blocks are passed through (OpenAI supports natively).
        """
        # Extract special params before general processing
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        response_format = kwargs.pop("response_format", None)
        json_schema = kwargs.pop("json_schema", None)

        body = super().build_body(model, messages, stream, **kwargs)

        # Reasoning / thinking -- reasoning_effort passed through for o-series
        if reasoning_effort:
            body.update(self.map_thinking(reasoning_effort))

        # Structured output -- response_format with json_schema passed through
        if response_format:
            body_updates, _ = self.map_response_format(response_format)
            body.update(body_updates)
        elif json_schema:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": json_schema,
            }

        # stream_options for usage in streaming responses
        if stream and "stream_options" not in body:
            body["stream_options"] = {"include_usage": True}

        return body

    def map_thinking(self, reasoning_effort: str) -> dict:
        """OpenAI uses reasoning_effort directly for o-series models."""
        return {"reasoning_effort": reasoning_effort}


# -- Factory for OpenAI-compatible providers ---------------------------------

def _make_openai_compat(
    provider_name: str,
    base_url: str,
    api_key_env: str,
    extra_supported_params: frozenset[str] | None = None,
) -> type[OpenAIProvider]:
    """Create and register an OpenAI-compatible provider class.

    Generates a new class that inherits from OpenAIProvider but overrides
    base_url and api_key_env.
    """
    params = OpenAIProvider.supported_params
    if extra_supported_params:
        params = params | extra_supported_params

    cls = type(
        f"{provider_name.title().replace('_', '')}Provider",
        (OpenAIProvider,),
        {
            "name": provider_name,
            "base_url": base_url,
            "api_key_env": api_key_env,
            "supported_params": params,
        },
    )
    register(provider_name)(cls)
    return cls


# -- Register all OpenAI-compatible providers --------------------------------

_make_openai_compat(
    "groq",
    "https://api.groq.com/openai/v1",
    "GROQ_API_KEY",
)

_make_openai_compat(
    "together_ai",
    "https://api.together.xyz/v1",
    "TOGETHER_API_KEY",
)

_make_openai_compat(
    "mistral",
    "https://api.mistral.ai/v1",
    "MISTRAL_API_KEY",
)

_make_openai_compat(
    "deepseek",
    "https://api.deepseek.com/v1",
    "DEEPSEEK_API_KEY",
)

_make_openai_compat(
    "perplexity",
    "https://api.perplexity.ai",
    "PERPLEXITYAI_API_KEY",
)

_make_openai_compat(
    "fireworks_ai",
    "https://api.fireworks.ai/inference/v1",
    "FIREWORKS_API_KEY",
)

_make_openai_compat(
    "openrouter",
    "https://openrouter.ai/api/v1",
    "OPENROUTER_API_KEY",
)

_make_openai_compat(
    "deepinfra",
    "https://api.deepinfra.com/v1/openai",
    "DEEPINFRA_API_KEY",
)

_make_openai_compat(
    "xai",
    "https://api.x.ai/v1",
    "XAI_API_KEY",
)

_make_openai_compat(
    "cerebras",
    "https://api.cerebras.ai/v1",
    "CEREBRAS_API_KEY",
)

_make_openai_compat(
    "anyscale",
    "https://api.anyscale.com/v1",
    "ANYSCALE_API_KEY",
)

_make_openai_compat(
    "nvidia_nim",
    "https://integrate.api.nvidia.com/v1",
    "NVIDIA_API_KEY",
)

_make_openai_compat(
    "sambanova",
    "https://api.sambanova.ai/v1",
    "SAMBANOVA_API_KEY",
)
