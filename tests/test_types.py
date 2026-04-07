"""Tests for response types — attribute and dict access."""

from nanollm._types import (
    Choice,
    EmbeddingData,
    EmbeddingResponse,
    Message,
    ModelResponse,
    Usage,
    make_model_response,
    make_embedding_response,
    make_stream_chunk,
)


class TestMessage:
    def test_attribute_access(self):
        msg = Message(content="Hello", role="assistant")
        assert msg.content == "Hello"
        assert msg.role == "assistant"

    def test_dict_access(self):
        msg = Message(content="Hello", role="assistant")
        assert msg["content"] == "Hello"
        assert msg["role"] == "assistant"

    def test_get_method(self):
        msg = Message(content="Hello")
        assert msg.get("content") == "Hello"
        assert msg.get("nonexistent", "default") == "default"


class TestChoice:
    def test_message_access(self):
        choice = Choice(
            message=Message(content="Hi"),
            index=0,
            finish_reason="stop",
        )
        assert choice.message.content == "Hi"
        assert choice["message"]["content"] == "Hi"
        assert choice.finish_reason == "stop"


class TestModelResponse:
    def test_standard_access_pattern(self):
        """Test response.choices[0].message.content — the main crawl4ai pattern."""
        response = make_model_response(content="Hello world", model="gpt-4o")
        assert response.choices[0].message.content == "Hello world"

    def test_dict_style_access(self):
        response = make_model_response(content="Hello")
        assert response["choices"][0]["message"]["content"] == "Hello"

    def test_usage(self):
        response = make_model_response(
            content="Hi",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
        assert response.usage.prompt_tokens == 10
        assert response.usage.total_tokens == 15

    def test_has_id_and_created(self):
        response = make_model_response(content="Hi")
        assert response.id.startswith("chatcmpl-")
        assert response.created > 0

    def test_usage_token_details_default_none(self):
        """crawl4ai checks hasattr/getattr on these fields."""
        response = make_model_response(content="Hi")
        assert response.usage.completion_tokens_details is None
        assert response.usage.prompt_tokens_details is None

    def test_usage_token_details_hasattr(self):
        """crawl4ai uses hasattr-style guard on these."""
        response = make_model_response(content="Hi")
        assert hasattr(response.usage, "completion_tokens_details")
        assert hasattr(response.usage, "prompt_tokens_details")

    def test_usage_token_details_dict_access(self):
        """crawl4ai calls .__dict__ on these when they're not None.

        Pattern from extraction_strategy.py:
            response.usage.completion_tokens_details.__dict__
            if response.usage.completion_tokens_details
            else {}
        """
        from nanollm._types import _AttrDict, Usage
        usage = Usage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            completion_tokens_details=_AttrDict({"reasoning_tokens": 3, "accepted_prediction_tokens": 0}),
            prompt_tokens_details=_AttrDict({"cached_tokens": 2}),
        )
        # __dict__ access (the crawl4ai pattern)
        ctd = usage.completion_tokens_details.__dict__ if usage.completion_tokens_details else {}
        ptd = usage.prompt_tokens_details.__dict__ if usage.prompt_tokens_details else {}
        assert ctd == {"reasoning_tokens": 3, "accepted_prediction_tokens": 0}
        assert ptd == {"cached_tokens": 2}

        # Attribute access also works
        assert usage.completion_tokens_details.reasoning_tokens == 3
        assert usage.prompt_tokens_details.cached_tokens == 2

    def test_usage_token_details_none_guard(self):
        """When None, the guard produces empty dict (crawl4ai pattern)."""
        from nanollm._types import Usage
        usage = Usage()
        ctd = usage.completion_tokens_details.__dict__ if usage.completion_tokens_details else {}
        ptd = usage.prompt_tokens_details.__dict__ if usage.prompt_tokens_details else {}
        assert ctd == {}
        assert ptd == {}


class TestEmbeddingResponse:
    def test_data_access(self):
        """Test response.data[i]['embedding'] — the crawl4ai pattern."""
        response = make_embedding_response(
            embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            model="text-embedding-3-small",
        )
        assert len(response.data) == 2
        assert response.data[0]["embedding"] == [0.1, 0.2, 0.3]
        assert response.data[1]["embedding"] == [0.4, 0.5, 0.6]
        assert response.data[0]["index"] == 0
        assert response.data[1]["index"] == 1

    def test_attribute_access(self):
        response = make_embedding_response(embeddings=[[0.1]])
        assert response.data[0].embedding == [0.1]


class TestStreamChunk:
    def test_content_chunk(self):
        """Test chunk["choices"][0]["delta"].get("content") — the crawl4ai pattern."""
        chunk = make_stream_chunk(content="Hello", model="gpt-4o")
        assert chunk["choices"][0]["delta"].get("content") == "Hello"
        assert chunk["choices"][0]["finish_reason"] is None

    def test_role_chunk(self):
        chunk = make_stream_chunk(role="assistant")
        assert chunk["choices"][0]["delta"].get("role") == "assistant"
        assert chunk["choices"][0]["delta"].get("content") is None

    def test_finish_chunk(self):
        chunk = make_stream_chunk(finish_reason="stop")
        assert chunk["choices"][0]["finish_reason"] == "stop"

    def test_chunk_has_id(self):
        chunk = make_stream_chunk(content="x")
        assert chunk["id"].startswith("chatcmpl-")
        assert chunk["object"] == "chat.completion.chunk"
