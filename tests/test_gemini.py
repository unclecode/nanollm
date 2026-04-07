"""Tests for Gemini adapter — message conversion, response parsing, streaming."""

import json

from nanollm.adapters.gemini import Adapter, _convert_messages


class TestConvertMessages:
    def test_system_extraction(self):
        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"},
        ]
        system, contents = _convert_messages(messages)
        assert system == "Be helpful"
        assert len(contents) == 1
        assert contents[0]["role"] == "user"
        assert contents[0]["parts"] == [{"text": "Hello"}]

    def test_assistant_becomes_model(self):
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        _, contents = _convert_messages(messages)
        assert contents[1]["role"] == "model"

    def test_multimodal_base64(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ"},
                    },
                ],
            }
        ]
        _, contents = _convert_messages(messages)
        parts = contents[0]["parts"]
        assert parts[0] == {"text": "Describe this"}
        assert "inline_data" in parts[1]
        assert parts[1]["inline_data"]["mime_type"] == "image/jpeg"


class TestBuildRequest:
    def setup_method(self):
        self.adapter = Adapter()

    def test_basic_request(self):
        from nanollm._config import get_provider_config
        config = get_provider_config("gemini")
        url, headers, body = self.adapter.build_request(
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": "Hello"}],
            api_key="test-key",
            provider_config=config,
        )
        assert "gemini-2.0-flash:generateContent" in url
        assert "key=test-key" in url
        assert "contents" in body

    def test_streaming_url(self):
        from nanollm._config import get_provider_config
        config = get_provider_config("gemini")
        url, _, _ = self.adapter.build_request(
            model="gemini-2.0-flash",
            messages=[],
            api_key="key",
            stream=True,
            provider_config=config,
        )
        assert "streamGenerateContent" in url
        assert "alt=sse" in url

    def test_json_mode(self):
        from nanollm._config import get_provider_config
        config = get_provider_config("gemini")
        _, _, body = self.adapter.build_request(
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": "JSON please"}],
            response_format={"type": "json_object"},
            provider_config=config,
        )
        assert body["generationConfig"]["responseMimeType"] == "application/json"

    def test_generation_config(self):
        from nanollm._config import get_provider_config
        config = get_provider_config("gemini")
        _, _, body = self.adapter.build_request(
            model="gemini-2.0-flash",
            messages=[],
            temperature=0.5,
            max_tokens=100,
            provider_config=config,
        )
        assert body["generationConfig"]["temperature"] == 0.5
        assert body["generationConfig"]["maxOutputTokens"] == 100


class TestParseResponse:
    def setup_method(self):
        self.adapter = Adapter()

    def test_standard_response(self):
        data = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Hello!"}],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15,
            },
        }
        response = self.adapter.parse_response(data, "gemini-2.0-flash")
        assert response.choices[0].message.content == "Hello!"
        assert response.choices[0].finish_reason == "stop"
        assert response.usage.prompt_tokens == 10

    def test_empty_candidates(self):
        data = {"candidates": []}
        response = self.adapter.parse_response(data)
        assert response.choices[0].message.content == ""


class TestParseStreamChunk:
    def setup_method(self):
        self.adapter = Adapter()

    def test_text_chunk(self):
        line = json.dumps({
            "candidates": [
                {
                    "content": {"parts": [{"text": "Hello"}]},
                }
            ]
        })
        chunk = self.adapter.parse_stream_chunk(line)
        assert chunk is not None
        assert chunk["choices"][0]["delta"]["content"] == "Hello"

    def test_finish_chunk(self):
        line = json.dumps({
            "candidates": [
                {
                    "content": {"parts": [{"text": ""}]},
                    "finishReason": "STOP",
                }
            ]
        })
        chunk = self.adapter.parse_stream_chunk(line)
        assert chunk["choices"][0]["finish_reason"] == "stop"

    def test_no_candidates(self):
        line = json.dumps({"candidates": []})
        assert self.adapter.parse_stream_chunk(line) is None
