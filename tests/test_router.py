"""Tests for model string parsing and provider routing."""

import pytest

from nanollm._router import parse_model_string, get_adapter, resolve
from nanollm._config import get_provider_config


class TestParseModelString:
    def test_openai(self):
        assert parse_model_string("openai/gpt-4o") == ("openai", "gpt-4o")

    def test_anthropic(self):
        assert parse_model_string("anthropic/claude-3-5-sonnet-20240620") == (
            "anthropic", "claude-3-5-sonnet-20240620"
        )

    def test_groq(self):
        assert parse_model_string("groq/llama3-70b-8192") == ("groq", "llama3-70b-8192")

    def test_ollama(self):
        assert parse_model_string("ollama/llama3") == ("ollama", "llama3")

    def test_gemini(self):
        assert parse_model_string("gemini/gemini-2.0-flash") == ("gemini", "gemini-2.0-flash")

    def test_deepseek(self):
        assert parse_model_string("deepseek/deepseek-chat") == ("deepseek", "deepseek-chat")

    def test_bedrock_with_dots(self):
        assert parse_model_string("bedrock/anthropic.claude-3-sonnet") == (
            "bedrock", "anthropic.claude-3-sonnet"
        )

    def test_no_provider_defaults_to_openai(self):
        assert parse_model_string("gpt-4o") == ("openai", "gpt-4o")

    def test_model_with_slashes(self):
        # e.g., "openrouter/meta-llama/llama-3-70b"
        assert parse_model_string("openrouter/meta-llama/llama-3-70b") == (
            "openrouter", "meta-llama/llama-3-70b"
        )


class TestGetProviderConfig:
    def test_known_provider(self):
        config = get_provider_config("openai")
        assert config.base_url == "https://api.openai.com/v1"
        assert config.api_key_env == "OPENAI_API_KEY"

    def test_groq(self):
        config = get_provider_config("groq")
        assert "groq.com" in config.base_url

    def test_anthropic(self):
        config = get_provider_config("anthropic")
        assert config.auth_header == "x-api-key"
        assert config.auth_prefix == ""

    def test_ollama(self):
        config = get_provider_config("ollama")
        assert "localhost" in config.base_url
        assert config.api_key_env is None

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider_config("nonexistent_provider")


class TestGetAdapter:
    def test_openai_adapter(self):
        adapter = get_adapter("openai")
        assert adapter.__class__.__name__ == "Adapter"

    def test_anthropic_adapter(self):
        adapter = get_adapter("anthropic")
        assert adapter.__class__.__module__ == "nanollm.adapters.anthropic"

    def test_gemini_adapter(self):
        adapter = get_adapter("gemini")
        assert adapter.__class__.__module__ == "nanollm.adapters.gemini"

    def test_ollama_adapter(self):
        adapter = get_adapter("ollama")
        assert adapter.__class__.__module__ == "nanollm.adapters.ollama"

    def test_groq_uses_openai_compat(self):
        adapter = get_adapter("groq")
        assert adapter.__class__.__module__ == "nanollm.adapters.openai_compat"


class TestResolve:
    def test_full_resolution(self):
        provider, model_id, adapter, config = resolve("openai/gpt-4o")
        assert provider == "openai"
        assert model_id == "gpt-4o"
        assert config.base_url == "https://api.openai.com/v1"
