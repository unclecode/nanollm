"""Exhaustive tests for nanollm.providers -- ~100+ tests."""
from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest

from nanollm.providers import get_provider, list_providers, _REGISTRY
from nanollm.providers.base import BaseProvider
from nanollm.providers.openai import OpenAIProvider
from nanollm.providers.anthropic import AnthropicProvider
from nanollm.providers.google import GeminiProvider, VertexProvider
from nanollm.providers.aws import BedrockProvider, sigv4_headers
from nanollm.providers.azure import AzureOpenAIProvider, AzureAIProvider
from nanollm.providers.local import (
    OllamaProvider, OllamaChatProvider, LMStudioProvider,
    VLLMProvider, TextGenWebUIProvider,
)
from nanollm._types import ModelResponse, Usage


# ── Registry ─────────────────────────────────────────────────────────


class TestRegistry:
    def test_get_provider_openai(self):
        p = get_provider("openai")
        assert isinstance(p, OpenAIProvider)

    def test_get_provider_anthropic(self):
        p = get_provider("anthropic")
        assert isinstance(p, AnthropicProvider)

    def test_get_provider_gemini(self):
        p = get_provider("gemini")
        assert isinstance(p, GeminiProvider)

    def test_get_provider_bedrock(self):
        p = get_provider("bedrock")
        assert isinstance(p, BedrockProvider)

    def test_get_provider_azure(self):
        p = get_provider("azure")
        assert isinstance(p, AzureOpenAIProvider)

    def test_get_provider_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("nonexistent_provider_xyz")

    def test_list_providers_returns_list(self):
        providers = list_providers()
        assert isinstance(providers, list)
        assert "openai" in providers
        assert "anthropic" in providers

    def test_list_providers_sorted(self):
        providers = list_providers()
        assert providers == sorted(providers)

    def test_groq_registered(self):
        assert "groq" in list_providers()

    def test_together_registered(self):
        assert "together_ai" in list_providers()

    def test_ollama_registered(self):
        assert "ollama" in list_providers()


# ── BaseProvider ─────────────────────────────────────────────────────


class TestBaseProvider:
    def test_get_api_key_explicit(self):
        bp = BaseProvider()
        assert bp.get_api_key("my-key") == "my-key"

    def test_get_api_key_env(self):
        bp = BaseProvider()
        bp.api_key_env = "TEST_API_KEY_XYZ"
        with patch.dict(os.environ, {"TEST_API_KEY_XYZ": "env-key"}):
            assert bp.get_api_key() == "env-key"

    def test_get_api_key_empty(self):
        bp = BaseProvider()
        assert bp.get_api_key() == ""

    def test_build_headers(self):
        bp = BaseProvider()
        h = bp.build_headers("my-key")
        assert h["Authorization"] == "Bearer my-key"
        assert h["Content-Type"] == "application/json"

    def test_build_url(self):
        bp = BaseProvider()
        bp.base_url = "https://api.example.com/v1"
        url = bp.build_url("gpt-4")
        assert url == "https://api.example.com/v1/chat/completions"

    def test_build_url_custom_base(self):
        bp = BaseProvider()
        bp.base_url = "https://api.example.com/v1"
        url = bp.build_url("gpt-4", base_url="https://custom.com/v1")
        assert url == "https://custom.com/v1/chat/completions"

    def test_build_body_basic(self):
        bp = BaseProvider()
        body = bp.build_body("gpt-4", [{"role": "user", "content": "hi"}])
        assert body["model"] == "gpt-4"
        assert body["messages"] == [{"role": "user", "content": "hi"}]
        assert body["stream"] is False

    def test_filter_params_drops_none(self):
        bp = BaseProvider()
        result = bp.filter_params({"a": 1, "b": None})
        assert result == {"a": 1}

    def test_filter_params_with_supported_params(self):
        bp = BaseProvider()
        bp.supported_params = frozenset({"temperature"})
        result = bp.filter_params({"temperature": 0.5, "custom": "val"})
        assert result == {"temperature": 0.5}

    def test_transform_messages_passthrough(self):
        bp = BaseProvider()
        msgs = [{"role": "user", "content": "hi"}]
        assert bp.transform_messages(msgs) is msgs

    def test_map_params_passthrough(self):
        bp = BaseProvider()
        assert bp.map_params(temperature=0.5) == {"temperature": 0.5}

    def test_map_thinking_default_empty(self):
        bp = BaseProvider()
        assert bp.map_thinking("high") == {}

    def test_map_response_format_default(self):
        bp = BaseProvider()
        body, headers = bp.map_response_format({"type": "json_object"})
        assert body["response_format"]["type"] == "json_object"

    def test_repr(self):
        bp = BaseProvider()
        bp.name = "test"
        assert "test" in repr(bp)


# ── BaseProvider parse_response ──────────────────────────────────────


class TestBaseProviderParseResponse:
    def test_basic(self):
        bp = BaseProvider()
        data = {
            "id": "abc",
            "model": "gpt-4",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "hello"},
                 "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        r = bp.parse_response(data)
        assert r.choices[0].message.content == "hello"
        assert r.usage.prompt_tokens == 10

    def test_with_tool_calls(self):
        bp = BaseProvider()
        data = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "fn", "arguments": "{}"},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
        }
        r = bp.parse_response(data)
        assert r.choices[0].message.tool_calls[0].function.name == "fn"

    def test_with_reasoning_content(self):
        bp = BaseProvider()
        data = {
            "choices": [{
                "message": {"role": "assistant", "content": "hi",
                            "reasoning_content": "think"},
                "finish_reason": "stop",
            }],
        }
        r = bp.parse_response(data)
        assert r.choices[0].message.reasoning_content == "think"

    def test_usage_with_details(self):
        bp = BaseProvider()
        data = {
            "choices": [{"message": {"content": "hi"}}],
            "usage": {
                "prompt_tokens": 10, "completion_tokens": 5,
                "prompt_tokens_details": {"cached_tokens": 3},
                "completion_tokens_details": {"reasoning_tokens": 2},
            },
        }
        r = bp.parse_response(data)
        assert r.usage.prompt_tokens_details.cached_tokens == 3
        assert r.usage.completion_tokens_details.reasoning_tokens == 2

    def test_empty_choices(self):
        bp = BaseProvider()
        data = {"choices": []}
        r = bp.parse_response(data)
        assert len(r.choices) == 1  # default choice

    def test_system_fingerprint(self):
        bp = BaseProvider()
        data = {"choices": [{"message": {"content": "hi"}}],
                "system_fingerprint": "fp_abc"}
        r = bp.parse_response(data)
        assert r.system_fingerprint == "fp_abc"


# ── BaseProvider parse_stream_line ───────────────────────────────────


class TestBaseProviderParseStreamLine:
    def test_basic(self):
        bp = BaseProvider()
        data = {
            "choices": [{"delta": {"content": "hello"}, "index": 0}],
        }
        r = bp.parse_stream_line(data)
        assert r.choices[0].delta.content == "hello"

    def test_empty_returns_none(self):
        bp = BaseProvider()
        assert bp.parse_stream_line({}) is None

    def test_usage_only_chunk(self):
        bp = BaseProvider()
        data = {"usage": {"prompt_tokens": 10, "completion_tokens": 5}}
        r = bp.parse_stream_line(data)
        assert r is not None
        assert r.usage.prompt_tokens == 10

    def test_tool_calls_delta(self):
        bp = BaseProvider()
        data = {
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "id": "tc1", "function": {"name": "fn", "arguments": "{}"},
                    }]
                },
                "index": 0,
            }],
        }
        r = bp.parse_stream_line(data)
        assert r.choices[0].delta.tool_calls[0].function.name == "fn"

    def test_finish_reason(self):
        bp = BaseProvider()
        data = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
        r = bp.parse_stream_line(data)
        assert r.choices[0].finish_reason == "stop"


# ── BaseProvider embeddings ──────────────────────────────────────────


class TestBaseProviderEmbeddings:
    def test_build_embedding_url(self):
        bp = BaseProvider()
        bp.base_url = "https://api.example.com/v1"
        url = bp.build_embedding_url("text-embedding-ada-002")
        assert url == "https://api.example.com/v1/embeddings"

    def test_build_embedding_body(self):
        bp = BaseProvider()
        body = bp.build_embedding_body("ada-002", ["hello"])
        assert body["model"] == "ada-002"
        assert body["input"] == ["hello"]

    def test_parse_embedding_response(self):
        bp = BaseProvider()
        data = {
            "data": [{"embedding": [0.1, 0.2], "index": 0}],
            "model": "ada-002",
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }
        r = bp.parse_embedding_response(data)
        assert r.data[0].embedding == [0.1, 0.2]
        assert r.usage.prompt_tokens == 5


# ── OpenAI Provider ──────────────────────────────────────────────────


class TestOpenAIProvider:
    def test_config(self):
        p = OpenAIProvider()
        assert p.name == "openai"
        assert p.base_url == "https://api.openai.com/v1"
        assert p.api_key_env == "OPENAI_API_KEY"

    def test_build_body_basic(self):
        p = OpenAIProvider()
        body = p.build_body("gpt-4", [{"role": "user", "content": "hi"}])
        assert body["model"] == "gpt-4"
        assert "stream_options" not in body

    def test_build_body_stream_adds_options(self):
        p = OpenAIProvider()
        body = p.build_body("gpt-4", [{"role": "user", "content": "hi"}], stream=True)
        assert body["stream_options"] == {"include_usage": True}

    def test_reasoning_effort_passthrough(self):
        p = OpenAIProvider()
        body = p.build_body("o1", [], reasoning_effort="high")
        assert body["reasoning_effort"] == "high"

    def test_response_format_passthrough(self):
        p = OpenAIProvider()
        body = p.build_body("gpt-4", [], response_format={"type": "json_object"})
        assert body["response_format"]["type"] == "json_object"

    def test_json_schema(self):
        p = OpenAIProvider()
        schema = {"name": "test", "schema": {"type": "object"}}
        body = p.build_body("gpt-4", [], json_schema=schema)
        assert body["response_format"]["type"] == "json_schema"

    def test_supported_params_filter(self):
        p = OpenAIProvider()
        body = p.build_body("gpt-4", [], temperature=0.5, custom_unsupported_param=1)
        assert "temperature" in body
        assert "custom_unsupported_param" not in body

    def test_map_thinking(self):
        p = OpenAIProvider()
        assert p.map_thinking("high") == {"reasoning_effort": "high"}


# ── Anthropic Provider ───────────────────────────────────────────────


class TestAnthropicProvider:
    def test_config(self):
        p = AnthropicProvider()
        assert p.name == "anthropic"
        assert p.api_key_env == "ANTHROPIC_API_KEY"

    def test_build_headers(self):
        p = AnthropicProvider()
        h = p.build_headers("my-key")
        assert h["x-api-key"] == "my-key"
        assert "anthropic-version" in h

    def test_build_url(self):
        p = AnthropicProvider()
        url = p.build_url("claude-3")
        assert url.endswith("/messages")

    def test_system_extraction(self):
        p = AnthropicProvider()
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        body = p.build_body("claude-3", msgs)
        assert body["system"] == "You are helpful."
        assert len(body["messages"]) == 1
        assert body["messages"][0]["role"] == "user"

    def test_multiple_system_messages(self):
        p = AnthropicProvider()
        msgs = [
            {"role": "system", "content": "A"},
            {"role": "system", "content": "B"},
            {"role": "user", "content": "Hi"},
        ]
        body = p.build_body("claude-3", msgs)
        assert "A" in body["system"]
        assert "B" in body["system"]

    def test_max_tokens_default(self):
        p = AnthropicProvider()
        body = p.build_body("claude-3", [])
        assert body["max_tokens"] == 4096

    def test_max_tokens_custom(self):
        p = AnthropicProvider()
        body = p.build_body("claude-3", [], max_tokens=1024)
        assert body["max_tokens"] == 1024

    def test_stop_to_stop_sequences(self):
        p = AnthropicProvider()
        body = p.build_body("claude-3", [], stop=["END"])
        assert body["stop_sequences"] == ["END"]

    def test_stop_string_to_list(self):
        p = AnthropicProvider()
        body = p.build_body("claude-3", [], stop="END")
        assert body["stop_sequences"] == ["END"]

    def test_tool_choice_auto(self):
        p = AnthropicProvider()
        assert p._convert_tool_choice("auto") == {"type": "auto"}

    def test_tool_choice_required(self):
        p = AnthropicProvider()
        assert p._convert_tool_choice("required") == {"type": "any"}

    def test_tool_choice_none(self):
        p = AnthropicProvider()
        assert p._convert_tool_choice("none") == {"type": "none"}

    def test_tool_choice_specific(self):
        p = AnthropicProvider()
        choice = {"type": "function", "function": {"name": "fn"}}
        result = p._convert_tool_choice(choice)
        assert result == {"type": "tool", "name": "fn"}

    def test_thinking_low(self):
        p = AnthropicProvider()
        result = p.map_thinking("low")
        assert result["thinking"]["budget_tokens"] == 1024

    def test_thinking_high(self):
        p = AnthropicProvider()
        result = p.map_thinking("high")
        assert result["thinking"]["budget_tokens"] == 4096

    def test_thinking_xhigh(self):
        p = AnthropicProvider()
        result = p.map_thinking("xhigh")
        assert result["thinking"]["budget_tokens"] == 16384

    def test_json_schema_to_tool(self):
        p = AnthropicProvider()
        schema = {"name": "extract", "schema": {"type": "object"}}
        body_updates, _ = p._map_json_schema(schema)
        assert "tools" in body_updates
        assert body_updates["tool_choice"]["type"] == "tool"

    def test_parse_response_text(self):
        p = AnthropicProvider()
        data = {
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        r = p.parse_response(data)
        assert r.choices[0].message.content == "hello"
        assert r.choices[0].finish_reason == "stop"
        assert r.usage.prompt_tokens == 10

    def test_parse_response_tool_use(self):
        p = AnthropicProvider()
        data = {
            "content": [
                {"type": "text", "text": ""},
                {"type": "tool_use", "id": "tu1", "name": "fn",
                 "input": {"a": 1}},
            ],
            "stop_reason": "tool_use",
        }
        r = p.parse_response(data)
        assert r.choices[0].finish_reason == "tool_calls"
        assert r.choices[0].message.tool_calls[0].function.name == "fn"

    def test_parse_response_thinking(self):
        p = AnthropicProvider()
        data = {
            "content": [
                {"type": "thinking", "thinking": "reasoning..."},
                {"type": "text", "text": "answer"},
            ],
            "stop_reason": "end_turn",
        }
        r = p.parse_response(data)
        assert r.choices[0].message.reasoning_content == "reasoning..."
        assert r.choices[0].message.content == "answer"

    def test_parse_stream_message_start(self):
        p = AnthropicProvider()
        data = {
            "type": "message_start",
            "message": {"id": "msg1", "model": "claude-3",
                        "usage": {"input_tokens": 10}},
        }
        r = p.parse_stream_line(data)
        assert r is not None
        assert r.choices[0].delta.role == "assistant"

    def test_parse_stream_text_delta(self):
        p = AnthropicProvider()
        data = {
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "hello"},
        }
        r = p.parse_stream_line(data)
        assert r.choices[0].delta.content == "hello"

    def test_parse_stream_thinking_delta(self):
        p = AnthropicProvider()
        data = {
            "type": "content_block_delta",
            "delta": {"type": "thinking_delta", "thinking": "hmm"},
        }
        r = p.parse_stream_line(data)
        assert r.choices[0].delta.reasoning_content == "hmm"

    def test_parse_stream_message_delta(self):
        p = AnthropicProvider()
        data = {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 20},
        }
        r = p.parse_stream_line(data)
        assert r.choices[0].finish_reason == "stop"

    def test_parse_stream_unknown_returns_none(self):
        p = AnthropicProvider()
        assert p.parse_stream_line({"type": "ping"}) is None

    def test_parse_stream_content_block_start_tool(self):
        p = AnthropicProvider()
        data = {
            "type": "content_block_start",
            "content_block": {"type": "tool_use", "id": "tu1", "name": "fn"},
        }
        r = p.parse_stream_line(data)
        assert r.choices[0].delta.tool_calls[0].function.name == "fn"

    def test_content_conversion_multimodal(self):
        p = AnthropicProvider()
        content = [
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
        ]
        result = p._convert_content(content)
        assert isinstance(result, list)
        assert result[0]["type"] == "text"
        assert result[1]["type"] == "image"

    def test_convert_tools(self):
        p = AnthropicProvider()
        tools = [{
            "type": "function",
            "function": {"name": "fn", "description": "desc",
                         "parameters": {"type": "object"}},
        }]
        result = p._convert_tools(tools)
        assert result[0]["name"] == "fn"
        assert "input_schema" in result[0]

    def test_response_format_json_object(self):
        p = AnthropicProvider()
        body = p.build_body("claude-3", [], response_format={"type": "json_object"})
        assert "json" in body.get("system", "").lower()


# ── Gemini Provider ──────────────────────────────────────────────────


class TestGeminiProvider:
    def test_config(self):
        p = GeminiProvider()
        assert p.name == "gemini"
        assert p.api_key_env == "GEMINI_API_KEY"

    def test_build_headers(self):
        p = GeminiProvider()
        h = p.build_headers("key")
        assert h == {"Content-Type": "application/json"}

    def test_build_url(self):
        p = GeminiProvider()
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            url = p.build_url("gemini-pro")
        assert "generateContent" in url
        assert "gemini-pro" in url

    def test_build_url_stream(self):
        p = GeminiProvider()
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            url = p.build_url("gemini-pro", stream=True)
        assert "streamGenerateContent" in url
        assert "alt=sse" in url

    def test_role_mapping(self):
        p = GeminiProvider()
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        sys, contents = p._convert_messages(msgs)
        assert contents[0]["role"] == "user"
        assert contents[1]["role"] == "model"

    def test_system_instruction(self):
        p = GeminiProvider()
        msgs = [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "hi"},
        ]
        body = p.build_body("gemini-pro", msgs)
        assert "systemInstruction" in body

    def test_generation_config(self):
        p = GeminiProvider()
        body = p.build_body("gemini-pro", [], max_tokens=100, temperature=0.5)
        gc = body["generationConfig"]
        assert gc["maxOutputTokens"] == 100
        assert gc["temperature"] == 0.5

    def test_thinking_config(self):
        p = GeminiProvider()
        result = p.map_thinking("high")
        assert "thinkingConfig" in result["generationConfig"]
        assert result["generationConfig"]["thinkingConfig"]["thinkingBudget"] == 16384

    def test_json_schema_to_response_mime(self):
        p = GeminiProvider()
        schema = {"schema": {"type": "object", "properties": {"name": {"type": "string"}}}}
        body = p.build_body("gemini-pro", [], json_schema=schema)
        assert body["generationConfig"]["responseMimeType"] == "application/json"

    def test_parse_response(self):
        p = GeminiProvider()
        data = {
            "candidates": [{
                "content": {"parts": [{"text": "hello"}]},
                "finishReason": "STOP",
            }],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5,
                              "totalTokenCount": 15},
        }
        r = p.parse_response(data)
        assert r.choices[0].message.content == "hello"
        assert r.usage.prompt_tokens == 10

    def test_parse_response_thinking(self):
        p = GeminiProvider()
        data = {
            "candidates": [{
                "content": {"parts": [
                    {"thought": "thinking..."},
                    {"text": "answer"},
                ]},
                "finishReason": "STOP",
            }],
        }
        r = p.parse_response(data)
        assert r.choices[0].message.reasoning_content == "thinking..."

    def test_parse_response_function_call(self):
        p = GeminiProvider()
        data = {
            "candidates": [{
                "content": {"parts": [
                    {"functionCall": {"name": "fn", "args": {"a": 1}}},
                ]},
                "finishReason": "STOP",
            }],
        }
        r = p.parse_response(data)
        tc = r.choices[0].message.tool_calls[0]
        assert tc.function.name == "fn"
        assert json.loads(tc.function.arguments) == {"a": 1}

    def test_parse_stream_line(self):
        p = GeminiProvider()
        data = {
            "candidates": [{
                "content": {"parts": [{"text": "hello"}]},
            }],
        }
        r = p.parse_stream_line(data)
        assert r.choices[0].delta.content == "hello"

    def test_parse_stream_empty_returns_none(self):
        p = GeminiProvider()
        assert p.parse_stream_line({}) is None

    def test_image_conversion(self):
        p = GeminiProvider()
        content = [
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
        ]
        parts = p._convert_content_to_parts(content)
        assert parts[0] == {"text": "describe"}
        assert "inline_data" in parts[1]


# ── Bedrock Provider ─────────────────────────────────────────────────


class TestBedrockProvider:
    def test_config(self):
        p = BedrockProvider()
        assert p.name == "bedrock"

    def test_build_url(self):
        p = BedrockProvider()
        with patch.dict(os.environ, {"AWS_REGION": "us-west-2"}):
            url = p.build_url("anthropic.claude-v2")
        assert "bedrock-runtime.us-west-2.amazonaws.com" in url
        assert "converse" in url

    def test_build_url_stream(self):
        p = BedrockProvider()
        with patch.dict(os.environ, {"AWS_REGION": "us-east-1"}):
            url = p.build_url("claude-v2", stream=True)
        assert "converse-stream" in url

    def test_build_body_inference_config(self):
        p = BedrockProvider()
        body = p.build_body("claude-v2", [{"role": "user", "content": "hi"}],
                            temperature=0.5, max_tokens=100)
        ic = body["inferenceConfig"]
        assert ic["maxTokens"] == 100
        assert ic["temperature"] == 0.5

    def test_system_extraction(self):
        p = BedrockProvider()
        msgs = [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "hi"},
        ]
        body = p.build_body("claude-v2", msgs)
        assert body["system"][0]["text"] == "be helpful"

    def test_message_format(self):
        p = BedrockProvider()
        msgs = [{"role": "user", "content": "hi"}]
        body = p.build_body("claude-v2", msgs)
        assert body["messages"][0]["role"] == "user"
        assert body["messages"][0]["content"] == [{"text": "hi"}]

    def test_parse_response(self):
        p = BedrockProvider()
        data = {
            "output": {"message": {"content": [{"text": "hello"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 5},
        }
        r = p.parse_response(data)
        assert r.choices[0].message.content == "hello"
        assert r.usage.prompt_tokens == 10

    def test_parse_stream_content_block_delta(self):
        p = BedrockProvider()
        data = {"contentBlockDelta": {"delta": {"text": "hello"}}}
        r = p.parse_stream_line(data)
        assert r.choices[0].delta.content == "hello"

    def test_parse_stream_message_stop(self):
        p = BedrockProvider()
        data = {"messageStop": {"stopReason": "end_turn"}}
        r = p.parse_stream_line(data)
        assert r.choices[0].finish_reason == "stop"

    def test_parse_stream_metadata(self):
        p = BedrockProvider()
        data = {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 5}}}
        r = p.parse_stream_line(data)
        assert r.usage.prompt_tokens == 10

    def test_has_build_signed_headers(self):
        p = BedrockProvider()
        assert hasattr(p, "build_signed_headers")


class TestSigV4Headers:
    def test_generates_authorization_header(self):
        headers = sigv4_headers(
            method="POST",
            url="https://bedrock-runtime.us-east-1.amazonaws.com/model/test/converse",
            body=b'{}',
            region="us-east-1",
            access_key="AKID",
            secret_key="SECRET",
        )
        assert "Authorization" in headers
        assert headers["Authorization"].startswith("AWS4-HMAC-SHA256")
        assert "x-amz-date" in headers

    def test_includes_session_token(self):
        headers = sigv4_headers(
            method="POST",
            url="https://example.com/test",
            body=b'{}',
            region="us-east-1",
            access_key="AKID",
            secret_key="SECRET",
            session_token="TOKEN",
        )
        assert headers["x-amz-security-token"] == "TOKEN"

    def test_no_credentials_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            # Need to also mock botocore to fail
            with patch("nanollm.providers.aws._get_credentials", return_value=("", "", "")):
                with pytest.raises(ValueError, match="AWS credentials"):
                    sigv4_headers("POST", "https://example.com/test", b'{}', "us-east-1")


# ── Azure Provider ───────────────────────────────────────────────────


class TestAzureProvider:
    def test_config(self):
        p = AzureOpenAIProvider()
        assert p.name == "azure"
        assert p.api_key_env == "AZURE_API_KEY"

    def test_build_headers(self):
        p = AzureOpenAIProvider()
        h = p.build_headers("my-key")
        assert h["api-key"] == "my-key"
        assert "Authorization" not in h

    def test_build_url(self):
        p = AzureOpenAIProvider()
        with patch.dict(os.environ, {"AZURE_API_BASE": "https://myresource.openai.azure.com"}):
            url = p.build_url("gpt-4")
        assert "openai/deployments/gpt-4" in url
        assert "api-version=" in url

    def test_build_url_no_base_raises(self):
        p = AzureOpenAIProvider()
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Azure requires"):
                p.build_url("gpt-4")

    def test_build_embedding_url(self):
        p = AzureOpenAIProvider()
        with patch.dict(os.environ, {"AZURE_API_BASE": "https://myresource.openai.azure.com"}):
            url = p.build_embedding_url("ada-002")
        assert "embeddings" in url


class TestAzureAIProvider:
    def test_build_url(self):
        p = AzureAIProvider()
        url = p.build_url("llama-3", base_url="https://endpoint.inference.ai.azure.com/v1")
        assert "chat/completions" in url

    def test_build_url_no_base_raises(self):
        p = AzureAIProvider()
        with pytest.raises(ValueError, match="Azure AI requires"):
            p.build_url("llama-3")


# ── Local Providers ──────────────────────────────────────────────────


class TestLocalProviders:
    def test_ollama_defaults(self):
        p = OllamaProvider()
        assert p.base_url == "http://localhost:11434/v1"
        assert p.get_api_key() == "ollama"

    def test_ollama_headers_no_auth(self):
        p = OllamaProvider()
        h = p.build_headers("ollama")
        assert "Authorization" not in h

    def test_ollama_chat(self):
        p = OllamaChatProvider()
        url = p.build_url("llama3")
        assert "/v1/chat/completions" in url

    def test_lm_studio_defaults(self):
        p = LMStudioProvider()
        assert p.base_url == "http://localhost:1234/v1"
        assert p.get_api_key() == "lm-studio"

    def test_vllm_defaults(self):
        p = VLLMProvider()
        assert p.base_url == "http://localhost:8000/v1"
        assert p.get_api_key() == "vllm"

    def test_text_gen_webui_defaults(self):
        p = TextGenWebUIProvider()
        assert p.base_url == "http://localhost:5000/v1"
        assert p.get_api_key() == "none"


# ── Factory-created providers ────────────────────────────────────────


class TestFactoryProviders:
    @pytest.mark.parametrize("name,expected_url,expected_env", [
        ("groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY"),
        ("together_ai", "https://api.together.xyz/v1", "TOGETHER_API_KEY"),
        ("mistral", "https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
        ("deepseek", "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
        ("perplexity", "https://api.perplexity.ai", "PERPLEXITYAI_API_KEY"),
        ("fireworks_ai", "https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY"),
        ("openrouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
        ("deepinfra", "https://api.deepinfra.com/v1/openai", "DEEPINFRA_API_KEY"),
        ("xai", "https://api.x.ai/v1", "XAI_API_KEY"),
        ("cerebras", "https://api.cerebras.ai/v1", "CEREBRAS_API_KEY"),
        ("anyscale", "https://api.anyscale.com/v1", "ANYSCALE_API_KEY"),
        ("nvidia_nim", "https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY"),
        ("sambanova", "https://api.sambanova.ai/v1", "SAMBANOVA_API_KEY"),
    ])
    def test_provider_config(self, name, expected_url, expected_env):
        p = get_provider(name)
        assert p.base_url == expected_url
        assert p.api_key_env == expected_env

    def test_factory_providers_are_openai_subclass(self):
        for name in ["groq", "together_ai", "mistral", "deepseek"]:
            p = get_provider(name)
            assert isinstance(p, OpenAIProvider)


# ── Additional provider edge cases ───────────────────────────────────


class TestProviderEdgeCases:
    def test_base_provider_parse_usage_none(self):
        bp = BaseProvider()
        assert bp._parse_usage(None) is None

    def test_base_provider_parse_usage_empty(self):
        bp = BaseProvider()
        assert bp._parse_usage({}) is None

    def test_base_provider_parse_usage_basic(self):
        bp = BaseProvider()
        u = bp._parse_usage({"prompt_tokens": 10, "completion_tokens": 5})
        assert u.prompt_tokens == 10

    def test_anthropic_tool_result_message(self):
        p = AnthropicProvider()
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc1", "function": {"name": "fn", "arguments": '{"x": 1}'}}
            ]},
            {"role": "tool", "tool_call_id": "tc1", "content": "result"},
        ]
        body = p.build_body("claude-3", msgs)
        # Should have 3 messages (user, assistant with tool_use, user with tool_result)
        assert len(body["messages"]) == 3
        assert body["messages"][2]["content"][0]["type"] == "tool_result"

    def test_anthropic_cache_read_tokens(self):
        p = AnthropicProvider()
        data = {
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5,
                      "cache_read_input_tokens": 3},
        }
        r = p.parse_response(data)
        assert r.usage.prompt_tokens_details.cached_tokens == 3

    def test_gemini_stop_sequences(self):
        p = GeminiProvider()
        body = p.build_body("gemini-pro", [], stop=["END", "STOP"])
        assert body["generationConfig"]["stopSequences"] == ["END", "STOP"]

    def test_gemini_response_format_json_object(self):
        p = GeminiProvider()
        body = p.build_body("gemini-pro", [],
                            response_format={"type": "json_object"})
        assert body["generationConfig"]["responseMimeType"] == "application/json"

    def test_gemini_response_format_json_schema(self):
        p = GeminiProvider()
        body = p.build_body("gemini-pro", [],
                            response_format={
                                "type": "json_schema",
                                "json_schema": {"schema": {"type": "object"}}
                            })
        gc = body["generationConfig"]
        assert gc["responseMimeType"] == "application/json"
        assert "responseSchema" in gc

    def test_gemini_safety_settings(self):
        p = GeminiProvider()
        body = p.build_body("gemini-pro", [],
                            safety_settings=[{"category": "HARM", "threshold": "BLOCK_NONE"}])
        assert "safetySettings" in body

    def test_gemini_tool_choice_auto(self):
        p = GeminiProvider()
        result = p._convert_tool_choice("auto")
        assert result["functionCallingConfig"]["mode"] == "AUTO"

    def test_gemini_tool_choice_required(self):
        p = GeminiProvider()
        result = p._convert_tool_choice("required")
        assert result["functionCallingConfig"]["mode"] == "ANY"

    def test_gemini_tool_choice_specific(self):
        p = GeminiProvider()
        result = p._convert_tool_choice({"function": {"name": "fn"}})
        assert "fn" in result["functionCallingConfig"]["allowedFunctionNames"]

    def test_bedrock_tool_choice_auto(self):
        p = BedrockProvider()
        result = p._convert_tool_choice("auto")
        assert "auto" in result["toolChoice"]

    def test_bedrock_tool_choice_required(self):
        p = BedrockProvider()
        result = p._convert_tool_choice("required")
        assert "any" in result["toolChoice"]

    def test_bedrock_tool_choice_none(self):
        p = BedrockProvider()
        result = p._convert_tool_choice("none")
        assert result == {}

    def test_bedrock_stop_sequences(self):
        p = BedrockProvider()
        body = p.build_body("claude-v2", [], stop="END")
        assert body["inferenceConfig"]["stopSequences"] == ["END"]

    def test_gemini_multiple_candidates(self):
        p = GeminiProvider()
        data = {
            "candidates": [
                {"content": {"parts": [{"text": "first"}]}, "finishReason": "STOP"},
                {"content": {"parts": [{"text": "second"}]}, "finishReason": "STOP"},
            ],
        }
        r = p.parse_response(data)
        assert len(r.choices) == 2
        assert r.choices[1].message.content == "second"

    def test_vertex_provider(self):
        p = get_provider("vertex_ai")
        assert isinstance(p, VertexProvider)

    def test_ollama_chat_provider(self):
        p = get_provider("ollama_chat")
        assert isinstance(p, OllamaChatProvider)
