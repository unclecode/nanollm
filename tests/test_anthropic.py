"""Tests for Anthropic adapter — message conversion, response parsing, streaming."""

import json

from nanollm.adapters.anthropic import Adapter, _convert_messages


class TestConvertMessages:
    def test_system_extraction(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        system, msgs = _convert_messages(messages)
        assert system == "You are helpful."
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "Hello"

    def test_multiple_system_messages(self):
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello"},
        ]
        system, msgs = _convert_messages(messages)
        assert "Be helpful." in system
        assert "Be concise." in system

    def test_no_system(self):
        messages = [{"role": "user", "content": "Hello"}]
        system, msgs = _convert_messages(messages)
        assert system is None
        assert len(msgs) == 1

    def test_multimodal_with_image(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgo="
                        },
                    },
                ],
            }
        ]
        _, msgs = _convert_messages(messages)
        content = msgs[0]["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image"
        assert content[1]["source"]["type"] == "base64"
        assert content[1]["source"]["media_type"] == "image/png"


class TestBuildRequest:
    def setup_method(self):
        self.adapter = Adapter()

    def test_basic_request(self):
        from nanollm._config import get_provider_config
        config = get_provider_config("anthropic")
        url, headers, body = self.adapter.build_request(
            model="claude-3-5-sonnet-20240620",
            messages=[{"role": "user", "content": "Hello"}],
            api_key="sk-ant-test",
            provider_config=config,
        )
        assert "/v1/messages" in url
        assert headers["x-api-key"] == "sk-ant-test"
        assert headers["anthropic-version"] == "2023-06-01"
        assert body["model"] == "claude-3-5-sonnet-20240620"
        assert body["max_tokens"] == 4096

    def test_json_mode_adds_instruction(self):
        from nanollm._config import get_provider_config
        config = get_provider_config("anthropic")
        _, _, body = self.adapter.build_request(
            model="claude-3-5-sonnet-20240620",
            messages=[{"role": "user", "content": "Give me JSON"}],
            response_format={"type": "json_object"},
            provider_config=config,
        )
        assert "JSON" in body.get("system", "")


class TestParseResponse:
    def setup_method(self):
        self.adapter = Adapter()

    def test_standard_response(self):
        data = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-5-sonnet-20240620",
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        response = self.adapter.parse_response(data)
        assert response.choices[0].message.content == "Hello!"
        assert response.choices[0].finish_reason == "stop"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 5
        assert response.usage.total_tokens == 15

    def test_multiple_content_blocks(self):
        data = {
            "content": [
                {"type": "text", "text": "Hello "},
                {"type": "text", "text": "World"},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
        response = self.adapter.parse_response(data)
        assert response.choices[0].message.content == "Hello World"


class TestParseStreamChunk:
    def setup_method(self):
        self.adapter = Adapter()

    def test_content_delta(self):
        line = json.dumps({
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "Hello"},
        })
        chunk = self.adapter.parse_stream_chunk(line)
        assert chunk is not None
        assert chunk["choices"][0]["delta"]["content"] == "Hello"

    def test_message_start(self):
        line = json.dumps({"type": "message_start", "message": {}})
        chunk = self.adapter.parse_stream_chunk(line)
        assert chunk["choices"][0]["delta"]["role"] == "assistant"

    def test_message_stop(self):
        line = json.dumps({"type": "message_stop"})
        chunk = self.adapter.parse_stream_chunk(line)
        assert chunk["choices"][0]["finish_reason"] == "stop"

    def test_message_delta_stop(self):
        line = json.dumps({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
        })
        chunk = self.adapter.parse_stream_chunk(line)
        assert chunk["choices"][0]["finish_reason"] == "stop"

    def test_unknown_event(self):
        line = json.dumps({"type": "ping"})
        chunk = self.adapter.parse_stream_chunk(line)
        assert chunk is None
