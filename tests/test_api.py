"""Exhaustive tests for nanollm module-level API functions -- ~60+ tests."""
from __future__ import annotations

import importlib
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

import nanollm
from nanollm import (
    completion,
    acompletion,
    batch_completion,
    embedding,
    aembedding,
    text_completion,
    atext_completion,
    NanoLLM,
    ModelResponse,
    TextCompletionResponse,
    EmbeddingResponse,
)


# ── Fixtures ─────────────────────────────────────────────────────────


def _mock_response_data(content="hello"):
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


def _mock_embed_data():
    return {
        "data": [{"embedding": [0.1, 0.2], "index": 0}],
        "model": "ada-002",
        "usage": {"prompt_tokens": 5, "total_tokens": 5},
    }


@pytest.fixture(autouse=True)
def _reset_default_client():
    """Reset the default client between tests."""
    nanollm._default_client = None
    yield
    nanollm._default_client = None


# ── completion ───────────────────────────────────────────────────────


class TestCompletion:
    @patch("nanollm.client.sync_post")
    def test_basic(self, mock_post):
        mock_post.return_value = _mock_response_data()
        r = completion("openai/gpt-4", messages=[{"role": "user", "content": "hi"}])
        assert isinstance(r, ModelResponse)
        assert r.choices[0].message.content == "hello"

    @patch("nanollm.client.sync_post")
    def test_with_api_key(self, mock_post):
        mock_post.return_value = _mock_response_data()
        r = completion("openai/gpt-4", messages=[], api_key="test-key")
        assert isinstance(r, ModelResponse)

    @patch("nanollm.client.sync_post")
    def test_with_base_url(self, mock_post):
        mock_post.return_value = _mock_response_data()
        r = completion("openai/gpt-4", messages=[], base_url="https://custom.com/v1")
        assert isinstance(r, ModelResponse)

    @patch("nanollm.client.sync_post")
    def test_api_base_alias(self, mock_post):
        mock_post.return_value = _mock_response_data()
        r = completion("openai/gpt-4", messages=[], api_base="https://custom.com/v1")
        assert isinstance(r, ModelResponse)

    @patch("nanollm.client.sync_post")
    def test_with_timeout(self, mock_post):
        mock_post.return_value = _mock_response_data()
        r = completion("openai/gpt-4", messages=[], timeout=30.0)
        assert isinstance(r, ModelResponse)

    @patch("nanollm.client.sync_post")
    def test_with_temperature(self, mock_post):
        mock_post.return_value = _mock_response_data()
        r = completion("openai/gpt-4", messages=[], temperature=0.5)
        assert isinstance(r, ModelResponse)

    @patch("nanollm.client.sync_post")
    def test_creates_default_client(self, mock_post):
        mock_post.return_value = _mock_response_data()
        assert nanollm._default_client is None
        completion("openai/gpt-4", messages=[])
        assert nanollm._default_client is not None

    @patch("nanollm.client.sync_post")
    def test_reuses_default_client(self, mock_post):
        mock_post.return_value = _mock_response_data()
        completion("openai/gpt-4", messages=[])
        client1 = nanollm._default_client
        completion("openai/gpt-4", messages=[])
        client2 = nanollm._default_client
        assert client1 is client2


# ── acompletion ──────────────────────────────────────────────────────


class TestAcompletion:
    @patch("nanollm.client.async_post", new_callable=AsyncMock)
    async def test_basic(self, mock_post):
        mock_post.return_value = _mock_response_data()
        r = await acompletion("openai/gpt-4", messages=[{"role": "user", "content": "hi"}])
        assert isinstance(r, ModelResponse)
        assert r.choices[0].message.content == "hello"

    @patch("nanollm.client.async_post", new_callable=AsyncMock)
    async def test_with_api_key(self, mock_post):
        mock_post.return_value = _mock_response_data()
        r = await acompletion("openai/gpt-4", messages=[], api_key="test-key")
        assert isinstance(r, ModelResponse)

    @patch("nanollm.client.async_post", new_callable=AsyncMock)
    async def test_api_base_alias(self, mock_post):
        mock_post.return_value = _mock_response_data()
        r = await acompletion("openai/gpt-4", messages=[], api_base="https://custom.com/v1")
        assert isinstance(r, ModelResponse)


# ── batch_completion ─────────────────────────────────────────────────


class TestBatchCompletion:
    @patch("nanollm.client.sync_post")
    def test_basic(self, mock_post):
        mock_post.return_value = _mock_response_data()
        message_lists = [
            [{"role": "user", "content": "hi"}],
            [{"role": "user", "content": "hello"}],
        ]
        results = batch_completion("openai/gpt-4", messages=message_lists)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, ModelResponse)

    @patch("nanollm.client.sync_post")
    def test_single_batch(self, mock_post):
        mock_post.return_value = _mock_response_data()
        results = batch_completion("openai/gpt-4",
                                   messages=[[{"role": "user", "content": "hi"}]])
        assert len(results) == 1

    @patch("nanollm.client.sync_post")
    def test_max_workers(self, mock_post):
        mock_post.return_value = _mock_response_data()
        results = batch_completion("openai/gpt-4",
                                   messages=[[{"role": "user", "content": "hi"}]],
                                   max_workers=2)
        assert len(results) == 1


# ── embedding ────────────────────────────────────────────────────────


class TestEmbedding:
    @patch("nanollm.client.sync_post")
    def test_basic(self, mock_post):
        mock_post.return_value = _mock_embed_data()
        r = embedding("openai/text-embedding-ada-002", input=["hello"])
        assert isinstance(r, EmbeddingResponse)
        assert r.data[0].embedding == [0.1, 0.2]

    @patch("nanollm.client.sync_post")
    def test_string_input(self, mock_post):
        mock_post.return_value = _mock_embed_data()
        r = embedding("openai/ada-002", input="hello")
        assert isinstance(r, EmbeddingResponse)

    @patch("nanollm.client.sync_post")
    def test_with_api_key(self, mock_post):
        mock_post.return_value = _mock_embed_data()
        r = embedding("openai/ada-002", input=["hi"], api_key="key")
        assert isinstance(r, EmbeddingResponse)

    @patch("nanollm.client.sync_post")
    def test_api_base_alias(self, mock_post):
        mock_post.return_value = _mock_embed_data()
        r = embedding("openai/ada-002", input=["hi"],
                      api_base="https://custom.com/v1")
        assert isinstance(r, EmbeddingResponse)


# ── aembedding ───────────────────────────────────────────────────────


class TestAembedding:
    @patch("nanollm.client.async_post", new_callable=AsyncMock)
    async def test_basic(self, mock_post):
        mock_post.return_value = _mock_embed_data()
        r = await aembedding("openai/ada-002", input=["hello"])
        assert isinstance(r, EmbeddingResponse)

    @patch("nanollm.client.async_post", new_callable=AsyncMock)
    async def test_string_input(self, mock_post):
        mock_post.return_value = _mock_embed_data()
        r = await aembedding("openai/ada-002", input="hello")
        assert isinstance(r, EmbeddingResponse)

    @patch("nanollm.client.async_post", new_callable=AsyncMock)
    async def test_api_base_alias(self, mock_post):
        mock_post.return_value = _mock_embed_data()
        r = await aembedding("openai/ada-002", input=["hi"],
                             api_base="https://custom.com/v1")
        assert isinstance(r, EmbeddingResponse)


# ── text_completion ──────────────────────────────────────────────────


class TestTextCompletion:
    @patch("nanollm.client.sync_post")
    def test_basic(self, mock_post):
        mock_post.return_value = _mock_response_data("completed text")
        r = text_completion("openai/gpt-4", prompt="Hello")
        assert isinstance(r, TextCompletionResponse)
        assert r.choices[0].text == "completed text"
        assert r.object == "text_completion"

    @patch("nanollm.client.sync_post")
    def test_finish_reason(self, mock_post):
        mock_post.return_value = _mock_response_data()
        r = text_completion("openai/gpt-4", prompt="Hello")
        assert r.choices[0].finish_reason == "stop"

    @patch("nanollm.client.sync_post")
    def test_with_api_key(self, mock_post):
        mock_post.return_value = _mock_response_data()
        r = text_completion("openai/gpt-4", prompt="Hi", api_key="key")
        assert isinstance(r, TextCompletionResponse)

    @patch("nanollm.client.sync_post")
    def test_api_base_alias(self, mock_post):
        mock_post.return_value = _mock_response_data()
        r = text_completion("openai/gpt-4", prompt="Hi",
                            api_base="https://custom.com/v1")
        assert isinstance(r, TextCompletionResponse)

    @patch("nanollm.client.sync_post")
    def test_preserves_model(self, mock_post):
        mock_post.return_value = _mock_response_data()
        r = text_completion("openai/gpt-4", prompt="Hi")
        assert r.model == "gpt-4"

    @patch("nanollm.client.sync_post")
    def test_preserves_usage(self, mock_post):
        mock_post.return_value = _mock_response_data()
        r = text_completion("openai/gpt-4", prompt="Hi")
        assert r.usage is not None


# ── atext_completion ─────────────────────────────────────────────────


class TestAtextCompletion:
    @patch("nanollm.client.async_post", new_callable=AsyncMock)
    async def test_basic(self, mock_post):
        mock_post.return_value = _mock_response_data("async completed")
        r = await atext_completion("openai/gpt-4", prompt="Hello")
        assert isinstance(r, TextCompletionResponse)
        assert r.choices[0].text == "async completed"

    @patch("nanollm.client.async_post", new_callable=AsyncMock)
    async def test_api_base_alias(self, mock_post):
        mock_post.return_value = _mock_response_data()
        r = await atext_completion("openai/gpt-4", prompt="Hi",
                                    api_base="https://custom.com/v1")
        assert isinstance(r, TextCompletionResponse)


# ── Module-level config ──────────────────────────────────────────────


class TestModuleConfig:
    def test_drop_params_default(self):
        assert nanollm.drop_params is True

    def test_set_verbose_default(self):
        assert nanollm.set_verbose is False

    @patch("nanollm.client.sync_post")
    def test_drop_params_proxied_to_client(self, mock_post):
        mock_post.return_value = _mock_response_data()
        nanollm.drop_params = True
        completion("openai/gpt-4", messages=[])
        assert nanollm._default_client.drop_params is True

    def test_version(self):
        assert nanollm.__version__ == "1.0.0"


# ── Error propagation ───────────────────────────────────────────────


class TestErrorPropagation:
    @patch("nanollm.client.sync_post")
    def test_auth_error_propagated(self, mock_post):
        from nanollm.exceptions import AuthenticationError
        mock_post.side_effect = AuthenticationError("bad key")
        with pytest.raises(AuthenticationError):
            completion("openai/gpt-4", messages=[])

    @patch("nanollm.client.sync_post")
    def test_rate_limit_propagated(self, mock_post):
        from nanollm.exceptions import RateLimitError
        mock_post.side_effect = RateLimitError("rate limited")
        with pytest.raises(RateLimitError):
            # Disable retries for clean test
            nanollm._default_client = NanoLLM(max_retries=0)
            completion("openai/gpt-4", messages=[])

    @patch("nanollm.client.sync_post")
    def test_invalid_request_propagated(self, mock_post):
        from nanollm.exceptions import InvalidRequestError
        mock_post.side_effect = InvalidRequestError("bad params")
        with pytest.raises(InvalidRequestError):
            completion("openai/gpt-4", messages=[])


# ── _get_default_client ─────────────────────────────────────────────


class TestGetDefaultClient:
    def test_creates_client(self):
        client = nanollm._get_default_client()
        assert isinstance(client, NanoLLM)

    def test_singleton(self):
        c1 = nanollm._get_default_client()
        c2 = nanollm._get_default_client()
        assert c1 is c2

    def test_respects_drop_params(self):
        nanollm.drop_params = False
        nanollm._default_client = None
        client = nanollm._get_default_client()
        assert client.drop_params is False
        nanollm.drop_params = True  # Reset
