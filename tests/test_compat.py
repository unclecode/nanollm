"""Exhaustive tests for litellm compatibility layer -- ~60+ tests."""
from __future__ import annotations

import sys
from unittest.mock import patch, AsyncMock

import pytest

import nanollm


# ── litellm imports ──────────────────────────────────────────────────


class TestLitellmImports:
    def test_import_litellm(self):
        import litellm
        assert litellm is not None

    def test_import_completion(self):
        from litellm import completion
        assert callable(completion)

    def test_import_acompletion(self):
        from litellm import acompletion
        assert callable(acompletion)

    def test_import_batch_completion(self):
        from litellm import batch_completion
        assert callable(batch_completion)

    def test_import_embedding(self):
        from litellm import embedding
        assert callable(embedding)

    def test_import_aembedding(self):
        from litellm import aembedding
        assert callable(aembedding)

    def test_import_text_completion(self):
        from litellm import text_completion
        assert callable(text_completion)

    def test_import_atext_completion(self):
        from litellm import atext_completion
        assert callable(atext_completion)

    def test_import_model_response(self):
        from litellm import ModelResponse
        assert ModelResponse is nanollm.ModelResponse

    def test_import_embedding_response(self):
        from litellm import EmbeddingResponse
        assert EmbeddingResponse is nanollm.EmbeddingResponse

    def test_import_text_completion_response(self):
        from litellm import TextCompletionResponse
        assert TextCompletionResponse is nanollm.TextCompletionResponse

    def test_import_stream_chunk_builder(self):
        from litellm import stream_chunk_builder
        assert stream_chunk_builder is nanollm.stream_chunk_builder

    def test_import_message(self):
        from litellm import Message
        assert Message is nanollm.Message

    def test_import_choice(self):
        from litellm import Choice
        assert Choice is nanollm.Choice

    def test_import_delta(self):
        from litellm import Delta
        assert Delta is nanollm.Delta

    def test_import_stream_choice(self):
        from litellm import StreamChoice
        assert StreamChoice is nanollm.StreamChoice

    def test_import_usage(self):
        from litellm import Usage
        assert Usage is nanollm.Usage

    def test_import_prompt_tokens_details(self):
        from litellm import PromptTokensDetails
        assert PromptTokensDetails is nanollm.PromptTokensDetails

    def test_import_completion_tokens_details(self):
        from litellm import CompletionTokensDetails
        assert CompletionTokensDetails is nanollm.CompletionTokensDetails

    def test_import_tool_call(self):
        from litellm import ToolCall
        assert ToolCall is nanollm.ToolCall

    def test_import_function_call(self):
        from litellm import FunctionCall
        assert FunctionCall is nanollm.FunctionCall

    def test_import_text_choice(self):
        from litellm import TextChoice
        assert TextChoice is nanollm.TextChoice

    def test_import_embedding_data(self):
        from litellm import EmbeddingData
        assert EmbeddingData is nanollm.EmbeddingData

    def test_import_extract_json(self):
        from litellm import extract_json
        assert extract_json is nanollm.extract_json

    def test_import_validate_json_response(self):
        from litellm import validate_json_response
        assert validate_json_response is nanollm.validate_json_response

    def test_import_nanollm_client(self):
        from litellm import NanoLLM
        assert NanoLLM is nanollm.NanoLLM


# ── litellm.exceptions imports ───────────────────────────────────────


class TestLitellmExceptions:
    def test_import_nanollm_exception(self):
        from litellm.exceptions import NanoLLMException
        from nanollm.exceptions import NanoLLMException as Original
        assert NanoLLMException is Original

    def test_import_openai_error(self):
        from litellm.exceptions import OpenAIError
        from nanollm.exceptions import OpenAIError as Original
        assert OpenAIError is Original

    def test_import_invalid_request(self):
        from litellm.exceptions import InvalidRequestError
        from nanollm.exceptions import InvalidRequestError as Original
        assert InvalidRequestError is Original

    def test_import_auth_error(self):
        from litellm.exceptions import AuthenticationError
        from nanollm.exceptions import AuthenticationError as Original
        assert AuthenticationError is Original

    def test_import_rate_limit(self):
        from litellm.exceptions import RateLimitError
        from nanollm.exceptions import RateLimitError as Original
        assert RateLimitError is Original

    def test_import_api_error(self):
        from litellm.exceptions import APIError
        from nanollm.exceptions import APIError as Original
        assert APIError is Original

    def test_import_api_connection_error(self):
        from litellm.exceptions import APIConnectionError
        from nanollm.exceptions import APIConnectionError as Original
        assert APIConnectionError is Original

    def test_import_timeout(self):
        from litellm.exceptions import Timeout
        from nanollm.exceptions import Timeout as Original
        assert Timeout is Original

    def test_import_retryable_error(self):
        from litellm.exceptions import RetryableError
        from nanollm.exceptions import RetryableError as Original
        assert RetryableError is Original

    def test_import_bad_request(self):
        from litellm.exceptions import BadRequestError
        from nanollm.exceptions import BadRequestError as Original
        assert BadRequestError is Original

    def test_import_not_found(self):
        from litellm.exceptions import NotFoundError
        from nanollm.exceptions import NotFoundError as Original
        assert NotFoundError is Original

    def test_import_permission_denied(self):
        from litellm.exceptions import PermissionDeniedError
        from nanollm.exceptions import PermissionDeniedError as Original
        assert PermissionDeniedError is Original

    def test_import_internal_server(self):
        from litellm.exceptions import InternalServerError
        from nanollm.exceptions import InternalServerError as Original
        assert InternalServerError is Original

    def test_import_service_unavailable(self):
        from litellm.exceptions import ServiceUnavailableError
        from nanollm.exceptions import ServiceUnavailableError as Original
        assert ServiceUnavailableError is Original

    def test_import_bad_gateway(self):
        from litellm.exceptions import BadGatewayError
        from nanollm.exceptions import BadGatewayError as Original
        assert BadGatewayError is Original


# ── Exception bridging ───────────────────────────────────────────────


class TestExceptionBridging:
    def test_catch_nanollm_as_litellm(self):
        from litellm.exceptions import InvalidRequestError
        from nanollm.exceptions import InvalidRequestError as NanoInvalid
        with pytest.raises(InvalidRequestError):
            raise NanoInvalid("test")

    def test_catch_litellm_as_nanollm(self):
        from litellm.exceptions import RateLimitError
        from nanollm.exceptions import NanoLLMException
        with pytest.raises(NanoLLMException):
            raise RateLimitError("test")

    def test_catch_as_openai_error(self):
        from litellm.exceptions import OpenAIError, AuthenticationError
        with pytest.raises(OpenAIError):
            raise AuthenticationError("test")


# ── drop_params proxy ────────────────────────────────────────────────


class TestDropParamsProxy:
    def test_read_drop_params(self):
        import litellm
        nanollm.drop_params = True
        assert litellm.drop_params is True

    def test_write_drop_params(self):
        import litellm
        litellm.drop_params = False
        assert nanollm.drop_params is False
        nanollm.drop_params = True  # Reset

    def test_read_set_verbose(self):
        import litellm
        nanollm.set_verbose = False
        assert litellm.set_verbose is False

    def test_write_set_verbose(self):
        import litellm
        litellm.set_verbose = True
        assert nanollm.set_verbose is True
        nanollm.set_verbose = False  # Reset


# ── Response access patterns (crawl4ai compatibility) ────────────────


class TestResponseAccessPatterns:
    def test_dict_access(self):
        from litellm import ModelResponse, Choice, Message
        r = ModelResponse(
            choices=[Choice(message=Message(content="hello"), finish_reason="stop")],
        )
        assert r["choices"][0]["message"]["content"] == "hello"

    def test_attr_access(self):
        from litellm import ModelResponse, Choice, Message
        r = ModelResponse(
            choices=[Choice(message=Message(content="hello"))],
        )
        assert r.choices[0].message.content == "hello"

    def test_get_with_default(self):
        from litellm import ModelResponse
        r = ModelResponse()
        assert r.get("nonexistent", "default") == "default"

    def test_to_dict(self):
        from litellm import ModelResponse, Choice, Message
        r = ModelResponse(choices=[Choice(message=Message(content="hi"))])
        d = r.to_dict()
        assert isinstance(d, dict)
        assert "choices" in d

    def test_model_dump(self):
        from litellm import ModelResponse, Choice, Message
        r = ModelResponse(choices=[Choice(message=Message(content="hi"))])
        d = r.model_dump()
        assert isinstance(d, dict)

    def test_json_serialization(self):
        import json
        from litellm import ModelResponse, Choice, Message
        r = ModelResponse(
            model="gpt-4",
            choices=[Choice(message=Message(content="hi"))],
        )
        j = r.json()
        data = json.loads(j)
        assert data["model"] == "gpt-4"

    def test_usage_dict_access(self):
        from litellm import Usage
        u = Usage(prompt_tokens=10, completion_tokens=5)
        assert u["prompt_tokens"] == 10

    def test_usage_details_dunder_dict(self):
        """crawl4ai accesses token details via __dict__."""
        from litellm import Usage
        u = Usage(
            prompt_tokens=10,
            prompt_tokens_details={"cached_tokens": 5},
            completion_tokens_details={"reasoning_tokens": 2},
        )
        # __dict__ must work on details
        assert u.prompt_tokens_details.__dict__ == {"cached_tokens": 5}
        assert u.completion_tokens_details.__dict__ == {"reasoning_tokens": 2}

    def test_usage_details_attr_access(self):
        from litellm import Usage
        u = Usage(prompt_tokens_details={"cached_tokens": 5})
        assert u.prompt_tokens_details.cached_tokens == 5


# ── Streaming chunk access ───────────────────────────────────────────


class TestStreamingChunkAccess:
    def test_stream_chunk_delta(self):
        from litellm import ModelResponse, StreamChoice, Delta
        chunk = ModelResponse(
            choices=[StreamChoice(delta=Delta(content="hello"))],
            stream=True,
        )
        assert chunk.choices[0].delta.content == "hello"

    def test_stream_chunk_dict_access(self):
        from litellm import ModelResponse, StreamChoice, Delta
        chunk = ModelResponse(
            choices=[StreamChoice(delta=Delta(content="hello"))],
            stream=True,
        )
        assert chunk["choices"][0]["delta"]["content"] == "hello"

    def test_stream_chunk_builder(self):
        from litellm import stream_chunk_builder, ModelResponse, StreamChoice, Delta
        chunks = [
            ModelResponse(
                choices=[StreamChoice(delta=Delta(content="hel"))],
                stream=True,
            ),
            ModelResponse(
                choices=[StreamChoice(delta=Delta(content="lo"))],
                stream=True,
            ),
        ]
        r = stream_chunk_builder(chunks)
        assert r.choices[0].message.content == "hello"

    def test_stream_chunk_object_type(self):
        from litellm import ModelResponse, StreamChoice, Delta
        chunk = ModelResponse(stream=True)
        assert chunk.object == "chat.completion.chunk"


# ── Type exports ─────────────────────────────────────────────────────


class TestTypeExports:
    def test_all_types_importable(self):
        from litellm import (
            ModelResponse,
            TextCompletionResponse,
            EmbeddingResponse,
            EmbeddingData,
            Message,
            Choice,
            Delta,
            StreamChoice,
            Usage,
            PromptTokensDetails,
            CompletionTokensDetails,
            ToolCall,
            FunctionCall,
            TextChoice,
        )
        # All should be classes
        for cls in [ModelResponse, TextCompletionResponse, EmbeddingResponse,
                    EmbeddingData, Message, Choice, Delta, StreamChoice, Usage,
                    PromptTokensDetails, CompletionTokensDetails, ToolCall,
                    FunctionCall, TextChoice]:
            assert isinstance(cls, type)

    def test_version_matches(self):
        import litellm
        assert litellm.__version__ == nanollm.__version__
