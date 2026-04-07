"""Provider registry and configuration for NanoLLM.

Each provider is defined by its base URL, API key environment variable,
adapter type, and the set of parameters it supports.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Common OpenAI-compatible parameters
OPENAI_PARAMS = {
    "temperature", "top_p", "n", "stream", "stop", "max_tokens",
    "presence_penalty", "frequency_penalty", "logit_bias", "user",
    "response_format", "seed", "tools", "tool_choice", "logprobs",
    "top_logprobs", "extra_headers", "functions", "function_call",
    "max_completion_tokens",
}

ANTHROPIC_PARAMS = {
    "temperature", "top_p", "top_k", "max_tokens", "stop",
    "stream", "system", "tools", "tool_choice", "metadata",
}

GEMINI_PARAMS = {
    "temperature", "top_p", "top_k", "max_tokens", "max_output_tokens",
    "stop", "stream", "response_format", "tools", "tool_choice",
    "candidate_count", "safety_settings",
}


@dataclass(frozen=True)
class ProviderConfig:
    base_url: str
    api_key_env: str | None
    adapter: str  # adapter module name in nanollm.adapters
    supported_params: frozenset[str] = field(default_factory=frozenset)
    auth_header: str = "Authorization"  # header name for auth
    auth_prefix: str = "Bearer"  # e.g. "Bearer", "x-api-key"
    extra_headers: dict[str, str] = field(default_factory=dict)


# Provider registry — provider_name -> ProviderConfig
PROVIDER_REGISTRY: dict[str, ProviderConfig] = {
    # === OpenAI-compatible providers ===
    "openai": ProviderConfig(
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        adapter="openai_compat",
        supported_params=frozenset(OPENAI_PARAMS),
    ),
    "groq": ProviderConfig(
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        adapter="openai_compat",
        supported_params=frozenset(OPENAI_PARAMS),
    ),
    "together": ProviderConfig(
        base_url="https://api.together.xyz/v1",
        api_key_env="TOGETHER_API_KEY",
        adapter="openai_compat",
        supported_params=frozenset(OPENAI_PARAMS),
    ),
    "together_ai": ProviderConfig(
        base_url="https://api.together.xyz/v1",
        api_key_env="TOGETHER_AI_API_KEY",
        adapter="openai_compat",
        supported_params=frozenset(OPENAI_PARAMS),
    ),
    "mistral": ProviderConfig(
        base_url="https://api.mistral.ai/v1",
        api_key_env="MISTRAL_API_KEY",
        adapter="openai_compat",
        supported_params=frozenset(OPENAI_PARAMS),
    ),
    "deepseek": ProviderConfig(
        base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
        adapter="openai_compat",
        supported_params=frozenset(OPENAI_PARAMS),
    ),
    "perplexity": ProviderConfig(
        base_url="https://api.perplexity.ai",
        api_key_env="PERPLEXITYAI_API_KEY",
        adapter="openai_compat",
        supported_params=frozenset(OPENAI_PARAMS),
    ),
    "fireworks": ProviderConfig(
        base_url="https://api.fireworks.ai/inference/v1",
        api_key_env="FIREWORKS_API_KEY",
        adapter="openai_compat",
        supported_params=frozenset(OPENAI_PARAMS),
    ),
    "fireworks_ai": ProviderConfig(
        base_url="https://api.fireworks.ai/inference/v1",
        api_key_env="FIREWORKS_AI_API_KEY",
        adapter="openai_compat",
        supported_params=frozenset(OPENAI_PARAMS),
    ),
    "openrouter": ProviderConfig(
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        adapter="openai_compat",
        supported_params=frozenset(OPENAI_PARAMS),
    ),
    "deepinfra": ProviderConfig(
        base_url="https://api.deepinfra.com/v1/openai",
        api_key_env="DEEPINFRA_API_KEY",
        adapter="openai_compat",
        supported_params=frozenset(OPENAI_PARAMS),
    ),
    "anyscale": ProviderConfig(
        base_url="https://api.endpoints.anyscale.com/v1",
        api_key_env="ANYSCALE_API_KEY",
        adapter="openai_compat",
        supported_params=frozenset(OPENAI_PARAMS),
    ),
    "xai": ProviderConfig(
        base_url="https://api.x.ai/v1",
        api_key_env="XAI_API_KEY",
        adapter="openai_compat",
        supported_params=frozenset(OPENAI_PARAMS),
    ),
    "cerebras": ProviderConfig(
        base_url="https://api.cerebras.ai/v1",
        api_key_env="CEREBRAS_API_KEY",
        adapter="openai_compat",
        supported_params=frozenset(OPENAI_PARAMS),
    ),
    "custom_openai": ProviderConfig(
        base_url="",  # Must be provided via base_url param
        api_key_env=None,
        adapter="openai_compat",
        supported_params=frozenset(OPENAI_PARAMS),
    ),

    # === Non-OpenAI providers ===
    "anthropic": ProviderConfig(
        base_url="https://api.anthropic.com",
        api_key_env="ANTHROPIC_API_KEY",
        adapter="anthropic",
        supported_params=frozenset(ANTHROPIC_PARAMS),
        auth_header="x-api-key",
        auth_prefix="",  # No prefix — raw key
        extra_headers={"anthropic-version": "2023-06-01"},
    ),
    "gemini": ProviderConfig(
        base_url="https://generativelanguage.googleapis.com/v1beta",
        api_key_env="GEMINI_API_KEY",
        adapter="gemini",
        supported_params=frozenset(GEMINI_PARAMS),
    ),

    # === Local providers ===
    "ollama": ProviderConfig(
        base_url="http://localhost:11434/v1",
        api_key_env=None,
        adapter="ollama",
        supported_params=frozenset(OPENAI_PARAMS),
    ),
    "ollama_chat": ProviderConfig(
        base_url="http://localhost:11434/v1",
        api_key_env=None,
        adapter="ollama",
        supported_params=frozenset(OPENAI_PARAMS),
    ),
    "lm_studio": ProviderConfig(
        base_url="http://localhost:1234/v1",
        api_key_env=None,
        adapter="ollama",
        supported_params=frozenset(OPENAI_PARAMS),
    ),

    # === Cloud providers ===
    "azure": ProviderConfig(
        base_url="",  # Dynamic: https://{resource}.openai.azure.com
        api_key_env="AZURE_API_KEY",
        adapter="azure_openai",
        supported_params=frozenset(OPENAI_PARAMS),
        auth_header="api-key",
        auth_prefix="",
    ),
    "bedrock": ProviderConfig(
        base_url="",  # Dynamic: regional endpoint
        api_key_env=None,  # Uses AWS credentials
        adapter="bedrock",
        supported_params=frozenset(ANTHROPIC_PARAMS | {"response_format"}),
    ),
    "vertex_ai": ProviderConfig(
        base_url="",  # Dynamic: regional endpoint
        api_key_env=None,  # Uses Google credentials
        adapter="vertex",
        supported_params=frozenset(GEMINI_PARAMS),
    ),
}


def get_provider_config(provider: str) -> ProviderConfig:
    """Look up provider configuration. Raises ValueError for unknown providers."""
    config = PROVIDER_REGISTRY.get(provider)
    if config is None:
        available = ", ".join(sorted(PROVIDER_REGISTRY.keys()))
        raise ValueError(
            f"Unknown provider: '{provider}'. "
            f"Available providers: {available}"
        )
    return config
