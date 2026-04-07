"""Crawl4ai compatibility tests.

Exercises the exact import patterns and access patterns used by crawl4ai
to ensure NanoLLM is a drop-in replacement for litellm.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestLitellmImportCompat:
    """Test that all crawl4ai import patterns work."""

    def test_from_litellm_import_completion(self):
        from litellm import completion
        assert callable(completion)

    def test_from_litellm_import_acompletion(self):
        from litellm import acompletion
        assert callable(acompletion)

    def test_from_litellm_import_batch_completion(self):
        from litellm import batch_completion
        assert callable(batch_completion)

    def test_from_litellm_import_aembedding(self):
        from litellm import aembedding
        assert callable(aembedding)

    def test_litellm_drop_params(self):
        import litellm
        import nanollm
        litellm.drop_params = True
        assert nanollm.drop_params is True
        litellm.drop_params = False
        assert nanollm.drop_params is False
        # Reset
        litellm.drop_params = True

    def test_litellm_set_verbose(self):
        import litellm
        import nanollm
        litellm.set_verbose = False
        assert nanollm.set_verbose is False

    def test_from_litellm_exceptions_import_ratelimiterror(self):
        from litellm.exceptions import RateLimitError
        assert issubclass(RateLimitError, Exception)

    def test_ratelimiterror_is_same_class(self):
        from litellm.exceptions import RateLimitError as LitellmRLE
        from nanollm.exceptions import RateLimitError as NanoRLE
        assert LitellmRLE is NanoRLE


class TestResponseAccessPatterns:
    """Test the exact access patterns used in crawl4ai code."""

    def test_completion_response_content(self):
        """crawl4ai/utils.py: response.choices[0].message.content"""
        from nanollm._types import make_model_response
        response = make_model_response(content="Hello world")
        assert response.choices[0].message.content == "Hello world"

    def test_embedding_response_data(self):
        """crawl4ai/utils.py: response.data[i]['embedding']"""
        from nanollm._types import make_embedding_response
        response = make_embedding_response(
            embeddings=[[0.1, 0.2], [0.3, 0.4]]
        )
        assert response.data[0]["embedding"] == [0.1, 0.2]
        assert response.data[1]["embedding"] == [0.3, 0.4]

    def test_streaming_chunk_content(self):
        """crawl4ai/cli.py: chunk["choices"][0]["delta"].get("content")"""
        from nanollm._types import make_stream_chunk
        chunk = make_stream_chunk(content="Hello")
        content = chunk["choices"][0]["delta"].get("content")
        assert content == "Hello"

    def test_streaming_chunk_no_content(self):
        """Ensure .get("content") returns None when no content."""
        from nanollm._types import make_stream_chunk
        chunk = make_stream_chunk(role="assistant")
        content = chunk["choices"][0]["delta"].get("content")
        assert content is None

    def test_batch_completion_response_list(self):
        """crawl4ai/legacy/llmtxt.py: response.choices[0].message.content for each in responses"""
        from nanollm._types import make_model_response
        responses = [
            make_model_response(content=f"Response {i}")
            for i in range(3)
        ]
        for i, response in enumerate(responses):
            assert response.choices[0].message.content == f"Response {i}"


class TestExceptionCompat:
    """Test that exceptions work as expected."""

    def test_ratelimiterror_has_message(self):
        from litellm.exceptions import RateLimitError
        err = RateLimitError(message="Rate limited", status_code=429)
        assert err.message == "Rate limited"
        assert err.status_code == 429

    def test_ratelimiterror_is_exception(self):
        from litellm.exceptions import RateLimitError
        with pytest.raises(RateLimitError):
            raise RateLimitError(message="test", status_code=429)

    def test_ratelimiterror_str(self):
        from litellm.exceptions import RateLimitError
        err = RateLimitError(message="Rate limited")
        assert "Rate limited" in str(err)


class TestProviderFormats:
    """Test that all provider formats used in crawl4ai config.py resolve correctly."""

    def test_openai_format(self):
        from nanollm._router import parse_model_string
        assert parse_model_string("openai/gpt-4o") == ("openai", "gpt-4o")

    def test_anthropic_format(self):
        from nanollm._router import parse_model_string
        assert parse_model_string("anthropic/claude-3-5-sonnet-20240620") == (
            "anthropic", "claude-3-5-sonnet-20240620"
        )

    def test_groq_format(self):
        from nanollm._router import parse_model_string
        assert parse_model_string("groq/llama3-70b-8192") == ("groq", "llama3-70b-8192")

    def test_gemini_format(self):
        from nanollm._router import parse_model_string
        assert parse_model_string("gemini/gemini-2.0-flash") == ("gemini", "gemini-2.0-flash")

    def test_deepseek_format(self):
        from nanollm._router import parse_model_string
        assert parse_model_string("deepseek/deepseek-chat") == ("deepseek", "deepseek-chat")

    def test_ollama_format(self):
        from nanollm._router import parse_model_string
        assert parse_model_string("ollama/llama3") == ("ollama", "llama3")
