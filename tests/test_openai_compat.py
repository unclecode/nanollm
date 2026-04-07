"""Tests for OpenAI-compatible adapter — request building, response parsing, streaming."""

import json

from nanollm._config import get_provider_config
from nanollm.adapters.openai_compat import Adapter


class TestBuildRequest:
    def setup_method(self):
        self.adapter = Adapter()
        self.config = get_provider_config("openai")

    def test_basic_request(self):
        url, headers, body = self.adapter.build_request(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
            provider_config=self.config,
        )
        assert url == "https://api.openai.com/v1/chat/completions"
        assert headers["Authorization"] == "Bearer sk-test"
        assert body["model"] == "gpt-4o"
        assert body["messages"] == [{"role": "user", "content": "Hello"}]
        assert "stream" not in body

    def test_streaming_request(self):
        _, _, body = self.adapter.build_request(
            model="gpt-4o",
            messages=[],
            stream=True,
            provider_config=self.config,
        )
        assert body["stream"] is True

    def test_extra_params(self):
        _, _, body = self.adapter.build_request(
            model="gpt-4o",
            messages=[],
            temperature=0.5,
            response_format={"type": "json_object"},
            provider_config=self.config,
        )
        assert body["temperature"] == 0.5
        assert body["response_format"] == {"type": "json_object"}

    def test_groq_base_url(self):
        config = get_provider_config("groq")
        url, headers, _ = self.adapter.build_request(
            model="llama3-70b-8192",
            messages=[],
            api_key="gsk-test",
            provider_config=config,
        )
        assert "groq.com" in url
        assert headers["Authorization"] == "Bearer gsk-test"


class TestParseResponse:
    def setup_method(self):
        self.adapter = Adapter()

    def test_standard_response(self):
        data = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        response = self.adapter.parse_response(data, "gpt-4o")
        assert response.choices[0].message.content == "Hello!"
        assert response.choices[0].finish_reason == "stop"
        assert response.usage.prompt_tokens == 10
        assert response.model == "gpt-4o"

    def test_empty_choices(self):
        data = {"choices": [], "model": "gpt-4o"}
        response = self.adapter.parse_response(data)
        assert len(response.choices) == 0

    def test_tool_calls(self):
        data = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{"id": "call_1", "type": "function"}],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
        response = self.adapter.parse_response(data)
        assert response.choices[0].message.tool_calls is not None
        assert response.choices[0].message.content is None


class TestParseStreamChunk:
    def setup_method(self):
        self.adapter = Adapter()

    def test_content_chunk(self):
        line = json.dumps({
            "choices": [{"delta": {"content": "Hello"}, "finish_reason": None}],
            "model": "gpt-4o",
        })
        chunk = self.adapter.parse_stream_chunk(line, "gpt-4o")
        assert chunk is not None
        assert chunk["choices"][0]["delta"]["content"] == "Hello"

    def test_finish_chunk(self):
        line = json.dumps({
            "choices": [{"delta": {}, "finish_reason": "stop"}],
        })
        chunk = self.adapter.parse_stream_chunk(line)
        assert chunk["choices"][0]["finish_reason"] == "stop"

    def test_invalid_json(self):
        assert self.adapter.parse_stream_chunk("not json") is None

    def test_empty_choices(self):
        line = json.dumps({"choices": []})
        assert self.adapter.parse_stream_chunk(line) is None


class TestBuildEmbeddingRequest:
    def setup_method(self):
        self.adapter = Adapter()
        self.config = get_provider_config("openai")

    def test_basic_embedding(self):
        url, headers, body = self.adapter.build_embedding_request(
            model="text-embedding-3-small",
            input=["Hello world"],
            api_key="sk-test",
            provider_config=self.config,
        )
        assert url == "https://api.openai.com/v1/embeddings"
        assert body["model"] == "text-embedding-3-small"
        assert body["input"] == ["Hello world"]


class TestParseEmbeddingResponse:
    def setup_method(self):
        self.adapter = Adapter()

    def test_standard_response(self):
        data = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3], "index": 0},
                {"embedding": [0.4, 0.5, 0.6], "index": 1},
            ],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }
        response = self.adapter.parse_embedding_response(data)
        assert len(response.data) == 2
        assert response.data[0]["embedding"] == [0.1, 0.2, 0.3]
        assert response.model == "text-embedding-3-small"


class TestFilterParams:
    def setup_method(self):
        self.adapter = Adapter()

    def test_filters_unsupported(self):
        supported = frozenset({"temperature", "max_tokens"})
        result = self.adapter.filter_params(
            {"temperature": 0.5, "unsupported_param": True, "max_tokens": 100},
            supported,
        )
        assert result == {"temperature": 0.5, "max_tokens": 100}
        assert "unsupported_param" not in result
