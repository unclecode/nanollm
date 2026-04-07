"""Model string routing for NanoLLM.

Parses "provider/model-name" strings and resolves to the appropriate
adapter instance and provider configuration.
"""

from __future__ import annotations

import importlib
from functools import lru_cache

from ._config import ProviderConfig, get_provider_config
from .adapters._base import BaseAdapter


def parse_model_string(model: str) -> tuple[str, str]:
    """Parse 'provider/model-name' into (provider, model_id).

    Examples:
        >>> parse_model_string("openai/gpt-4o")
        ("openai", "gpt-4o")
        >>> parse_model_string("anthropic/claude-3-5-sonnet-20240620")
        ("anthropic", "claude-3-5-sonnet-20240620")
        >>> parse_model_string("ollama/llama3")
        ("ollama", "llama3")
        >>> parse_model_string("bedrock/anthropic.claude-3-sonnet")
        ("bedrock", "anthropic.claude-3-sonnet")
    """
    if "/" not in model:
        # Default to openai if no provider prefix
        return "openai", model

    # Split on first "/" only — model names can contain "/"
    provider, model_id = model.split("/", 1)
    return provider, model_id


@lru_cache(maxsize=32)
def get_adapter(provider: str) -> BaseAdapter:
    """Get the adapter instance for a provider (cached)."""
    config = get_provider_config(provider)
    module = importlib.import_module(f"nanollm.adapters.{config.adapter}")
    adapter_class = module.Adapter
    return adapter_class()


def resolve(
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    api_base: str | None = None,
) -> tuple[str, str, BaseAdapter, ProviderConfig]:
    """Full resolution: model string → (provider, model_id, adapter, config).

    Args:
        model: Model string like "openai/gpt-4o"
        api_key: Explicit API key (overrides env var)
        base_url: Explicit base URL (overrides provider default)
        api_base: Alias for base_url (litellm compat)

    Returns:
        Tuple of (provider_name, model_id, adapter_instance, provider_config)
    """
    provider, model_id = parse_model_string(model)
    adapter = get_adapter(provider)
    config = get_provider_config(provider)
    return provider, model_id, adapter, config
