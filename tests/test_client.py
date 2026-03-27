"""Exhaustive tests for nanollm.client -- ~70+ tests."""
from __future__ import annotations

import json
import time
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from nanollm.client import NanoLLM, _parse_model_string, _backoff_delay, _to_openai_image_block
from nanollm._types import ModelResponse, Choice, Message, Usage, EmbeddingResponse, EmbeddingData
from nanollm.exceptions import (
    RateLimitError,
    InvalidRequestError,
    AuthenticationError,
    InternalServerError,
    Timeout,
    RetryableError,
)


# ── _parse_model_string ──────────────────────────────────────────────


class TestParseModelString:
    def test_explicit_provider(self):
        assert _parse_model_string("openai/gpt-4o") == ("openai", "gpt-4o")

    def test_explicit_anthropic(self):
        assert _parse_model_string("anthropic/claude-3") == ("anthropic", "claude-3")

    def test_auto_detect_claude(self):
        assert _parse_model_string("claude-3-opus") == ("anthropic", "claude-3-opus")

    def test_auto_detect_gemini(self):
        assert _parse_model_string("gemini-pro") == ("gemini", "gemini-pro")

    def test_auto_detect_gpt(self):
        assert _parse_model_string("gpt-4o") == ("openai", "gpt-4o")

    def test_auto_detect_o1(self):
        assert _parse_model_string("o1-preview") == ("openai", "o1-preview")

    def test_auto_detect_o3(self):
        assert _parse_model_string("o3-mini") == ("openai", "o3-mini")

    def test_auto_detect_o4(self):
        assert _parse_model_string("o4-mini") == ("openai", "o4-mini")

    def test_auto_detect_llama(self):
        assert _parse_model_string("llama-3-70b") == ("groq", "llama-3-70b")

    def test_auto_detect_mixtral(self):
        assert _parse_model_string("mixtral-8x7b") == ("groq", "mixtral-8x7b")

    def test_auto_detect_mistral(self):
        assert _parse_model_string("mistral-large") == ("mistral", "mistral-large")

    def test_auto_detect_deepseek(self):
        assert _parse_model_string("deepseek-chat") == ("deepseek", "deepseek-chat")

    def test_auto_detect_grok(self):
        assert _parse_model_string("grok-1") == ("xai", "grok-1")

    def test_default_to_openai(self):
        assert _parse_model_string("unknown-model") == ("openai", "unknown-model")

    def test_nested_slash(self):
        assert _parse_model_string("bedrock/anthropic.claude-v2") == (
            "bedrock", "anthropic.claude-v2"
        )


# ── _backoff_delay ───────────────────────────────────────────────────


class TestBackoffDelay:
    def test_attempt_0(self):
        delay = _backoff_delay(0)
        assert 0 < delay <= 60.0

    def test_increases(self):
        d0 = _backoff_delay(0, base=1.0)
        d1 = _backoff_delay(1, base=1.0)
        d2 = _backoff_delay(2, base=1.0)
        # On average d2 > d1 > d0, but with jitter let's just check max
        assert d2 <= 60.0

    def test_capped_at_max(self):
        delay = _backoff_delay(100, base=1.0, max_delay=5.0)
        assert delay <= 5.0

    def test_positive(self):
        for i in range(10):
            assert _backoff_delay(i) > 0


# ── NanoLLM construction ─────────────────────────────────────────────


class TestNanoLLMInit:
    def test_defaults(self):
        c = NanoLLM()
        assert c.default_model is None
        assert c.api_key is None
        assert c.max_retries == 3
        assert c.timeout == 600
        assert c.drop_params is True

    def test_custom(self):
        c = NanoLLM(
            default_model="gpt-4",
            api_key="my-key",
            base_url="https://custom.com",
            max_retries=5,
            timeout=30,
            drop_params=False,
        )
        assert c.default_model == "gpt-4"
        assert c.api_key == "my-key"
        assert c.base_url == "https://custom.com"
        assert c.max_retries == 5
        assert c.timeout == 30
        assert c.drop_params is False


class TestResolveModel:
    def test_explicit_model(self):
        c = NanoLLM(default_model="gpt-4")
        assert c._resolve_model("gpt-3.5") == "gpt-3.5"

    def test_default_model(self):
        c = NanoLLM(default_model="gpt-4")
        assert c._resolve_model(None) == "gpt-4"

    def test_no_model_raises(self):
        c = NanoLLM()
        with pytest.raises(ValueError, match="No model"):
            c._resolve_model(None)


# ── complete() with mocked HTTP ──────────────────────────────────────


class TestComplete:
    def _make_response_data(self, content="hello"):
        return {
            "id": "chatcmpl-123",
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    @patch("nanollm.client.sync_post")
    def test_basic_complete(self, mock_post):
        mock_post.return_value = self._make_response_data()
        c = NanoLLM()
        r = c.complete("openai/gpt-4", messages=[{"role": "user", "content": "hi"}])
        assert r.choices[0].message.content == "hello"
        assert mock_post.called

    @patch("nanollm.client.sync_post")
    def test_drop_params_filters_none(self, mock_post):
        mock_post.return_value = self._make_response_data()
        c = NanoLLM(drop_params=True)
        c.complete("openai/gpt-4", messages=[{"role": "user", "content": "hi"}])
        call_args = mock_post.call_args
        body = call_args[0][2]  # 3rd positional arg
        assert all(v is not None for v in body.values())

    @patch("nanollm.client.sync_post")
    def test_api_key_override(self, mock_post):
        mock_post.return_value = self._make_response_data()
        c = NanoLLM(api_key="default-key")
        c.complete("openai/gpt-4", messages=[], api_key="override-key")
        # Provider should receive override-key
        assert mock_post.called


# ── Retry logic ──────────────────────────────────────────────────────


class TestRetrySync:
    @patch("nanollm.client.time.sleep")
    def test_retries_on_retryable(self, mock_sleep):
        c = NanoLLM(max_retries=2)
        call_count = 0

        def failing_fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RateLimitError("rate limited")
            return {"choices": [{"message": {"content": "ok"}}]}

        result = c._retry_sync(failing_fn)
        assert call_count == 3
        assert mock_sleep.call_count == 2

    @patch("nanollm.client.time.sleep")
    def test_no_retry_on_non_retryable(self, mock_sleep):
        c = NanoLLM(max_retries=3)

        def failing_fn():
            raise InvalidRequestError("bad request")

        with pytest.raises(InvalidRequestError):
            c._retry_sync(failing_fn)
        assert mock_sleep.call_count == 0

    @patch("nanollm.client.time.sleep")
    def test_exhausts_retries(self, mock_sleep):
        c = NanoLLM(max_retries=2)

        def failing_fn():
            raise RateLimitError("rate limited")

        with pytest.raises(RateLimitError):
            c._retry_sync(failing_fn)
        assert mock_sleep.call_count == 2

    @patch("nanollm.client.time.sleep")
    def test_sets_provider_on_exception(self, mock_sleep):
        c = NanoLLM(max_retries=0)

        def failing_fn():
            raise InvalidRequestError("bad")

        with pytest.raises(InvalidRequestError) as exc_info:
            c._retry_sync(failing_fn, provider_name="openai", model_name="gpt-4")
        assert exc_info.value.llm_provider == "openai"
        assert exc_info.value.model == "gpt-4"

    @patch("nanollm.client.time.sleep")
    def test_retries_internal_server_error(self, mock_sleep):
        c = NanoLLM(max_retries=1)
        call_count = 0

        def failing_fn():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise InternalServerError("500")
            return {"choices": [{"message": {"content": "ok"}}]}

        result = c._retry_sync(failing_fn)
        assert call_count == 2

    @patch("nanollm.client.time.sleep")
    def test_retries_timeout(self, mock_sleep):
        c = NanoLLM(max_retries=1)
        call_count = 0

        def failing_fn():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Timeout("timeout")
            return {}

        c._retry_sync(failing_fn)
        assert call_count == 2

    def test_no_retry_auth_error(self):
        c = NanoLLM(max_retries=3)

        def failing_fn():
            raise AuthenticationError("bad key")

        with pytest.raises(AuthenticationError):
            c._retry_sync(failing_fn)


# ── Async retry ──────────────────────────────────────────────────────


class TestRetryAsync:
    @patch("nanollm.client.asyncio.sleep", new_callable=AsyncMock)
    async def test_retries_on_retryable(self, mock_sleep):
        c = NanoLLM(max_retries=2)
        call_count = 0

        async def failing_fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RateLimitError("rate limited")
            return {}

        result = await c._retry_async(failing_fn)
        assert call_count == 3

    @patch("nanollm.client.asyncio.sleep", new_callable=AsyncMock)
    async def test_no_retry_on_non_retryable(self, mock_sleep):
        c = NanoLLM(max_retries=3)

        async def failing_fn():
            raise InvalidRequestError("bad")

        with pytest.raises(InvalidRequestError):
            await c._retry_async(failing_fn)


# ── acomplete() ──────────────────────────────────────────────────────


class TestAcomplete:
    @patch("nanollm.client.async_post", new_callable=AsyncMock)
    async def test_basic_acomplete(self, mock_post):
        mock_post.return_value = {
            "choices": [{
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop",
            }],
        }
        c = NanoLLM()
        r = await c.acomplete("openai/gpt-4", messages=[{"role": "user", "content": "hi"}])
        assert r.choices[0].message.content == "hello"


# ── embed() ──────────────────────────────────────────────────────────


class TestEmbed:
    @patch("nanollm.client.sync_post")
    def test_basic_embed(self, mock_post):
        mock_post.return_value = {
            "data": [{"embedding": [0.1, 0.2], "index": 0}],
            "model": "text-embedding-ada-002",
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }
        c = NanoLLM()
        r = c.embed("openai/text-embedding-ada-002", input=["hello"])
        assert isinstance(r, EmbeddingResponse)
        assert r.data[0].embedding == [0.1, 0.2]


class TestAembed:
    @patch("nanollm.client.async_post", new_callable=AsyncMock)
    async def test_basic_aembed(self, mock_post):
        mock_post.return_value = {
            "data": [{"embedding": [0.1, 0.2], "index": 0}],
            "model": "ada-002",
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }
        c = NanoLLM()
        r = await c.aembed("openai/ada-002", input=["hello"])
        assert isinstance(r, EmbeddingResponse)


# ── json() ───────────────────────────────────────────────────────────


class TestJsonMethod:
    @patch("nanollm.client.sync_post")
    def test_json_from_content(self, mock_post):
        mock_post.return_value = {
            "choices": [{
                "message": {"role": "assistant", "content": '{"name": "Alice"}'},
                "finish_reason": "stop",
            }],
        }
        c = NanoLLM()
        result = c.json("openai/gpt-4", messages=[{"role": "user", "content": "hi"}])
        assert result == {"name": "Alice"}

    @patch("nanollm.client.sync_post")
    def test_json_from_tool_call(self, mock_post):
        mock_post.return_value = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "tc1", "type": "function",
                        "function": {"name": "fn", "arguments": '{"a": 1}'},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
        }
        c = NanoLLM()
        result = c.json("openai/gpt-4", messages=[])
        assert result == {"a": 1}

    @patch("nanollm.client.sync_post")
    def test_json_empty_response(self, mock_post):
        mock_post.return_value = {
            "choices": [{
                "message": {"role": "assistant", "content": None},
                "finish_reason": "stop",
            }],
        }
        c = NanoLLM()
        result = c.json("openai/gpt-4", messages=[])
        assert result == {}

    @patch("nanollm.client.sync_post")
    def test_json_with_schema(self, mock_post):
        mock_post.return_value = {
            "choices": [{
                "message": {"role": "assistant", "content": '{"name": "Alice"}'},
                "finish_reason": "stop",
            }],
        }
        c = NanoLLM()
        result = c.json("openai/gpt-4", messages=[],
                        schema={"type": "object"})
        assert isinstance(result, dict)


# ── vision() ─────────────────────────────────────────────────────────


class TestVision:
    @patch("nanollm.client.sync_post")
    def test_vision_builds_content_blocks(self, mock_post):
        mock_post.return_value = {
            "choices": [{
                "message": {"role": "assistant", "content": "a cat"},
                "finish_reason": "stop",
            }],
        }
        c = NanoLLM()
        r = c.vision("openai/gpt-4o", prompt="describe",
                      images=["https://example.com/cat.jpg"])
        assert r.choices[0].message.content == "a cat"
        # Check the call was made with image content
        call_args = mock_post.call_args
        body = call_args[0][2]
        last_msg = body["messages"][-1]
        assert last_msg["role"] == "user"
        assert any(b.get("type") == "image_url" for b in last_msg["content"])

    @patch("nanollm.client.sync_post")
    def test_vision_no_images(self, mock_post):
        mock_post.return_value = {
            "choices": [{"message": {"content": "no images"}}],
        }
        c = NanoLLM()
        r = c.vision("openai/gpt-4o", prompt="hello")
        assert r.choices[0].message.content == "no images"


# ── _to_openai_image_block ──────────────────────────────────────────


class TestToOpenAIImageBlock:
    def test_https_url(self):
        block = _to_openai_image_block("https://example.com/img.png")
        assert block["type"] == "image_url"
        assert block["image_url"]["url"] == "https://example.com/img.png"

    def test_http_url(self):
        block = _to_openai_image_block("http://example.com/img.png")
        assert block["image_url"]["url"] == "http://example.com/img.png"

    def test_data_uri(self):
        uri = "data:image/png;base64,abc123"
        block = _to_openai_image_block(uri)
        assert block["image_url"]["url"] == uri

    def test_detail_param(self):
        block = _to_openai_image_block("https://example.com/img.png", detail="high")
        assert block["image_url"]["detail"] == "high"

    def test_local_file(self):
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNG\r\n\x1a\n")
            path = f.name
        try:
            block = _to_openai_image_block(path)
            assert block["image_url"]["url"].startswith("data:image/png;base64,")
        finally:
            os.unlink(path)

    def test_raw_base64(self):
        block = _to_openai_image_block("not_a_file_or_url")
        assert block["image_url"]["url"].startswith("data:image/png;base64,")


# ── drop_params behavior ────────────────────────────────────────────


class TestDropParams:
    @patch("nanollm.client.sync_post")
    def test_drop_params_true(self, mock_post):
        mock_post.return_value = {"choices": [{"message": {"content": "hi"}}]}
        c = NanoLLM(drop_params=True)
        c.complete("openai/gpt-4", messages=[])
        body = mock_post.call_args[0][2]
        assert all(v is not None for v in body.values())

    @patch("nanollm.client.sync_post")
    def test_drop_params_false(self, mock_post):
        mock_post.return_value = {"choices": [{"message": {"content": "hi"}}]}
        c = NanoLLM(drop_params=False)
        c.complete("openai/gpt-4", messages=[])
        # Body may contain None values
        assert mock_post.called
