"""End-to-end mocked tests simulating real crawl4ai usage patterns.

These tests mock the HTTP layer with respx to verify the full pipeline:
model string → routing → adapter → HTTP request → response parsing.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
import respx

import nanollm
from nanollm import acompletion, aembedding, batch_completion, completion
from nanollm.exceptions import AuthenticationError, RateLimitError


# ── Fixtures ──


OPENAI_COMPLETION_RESPONSE = {
    "id": "chatcmpl-abc123",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello! How can I help?"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18},
}

OPENAI_EMBEDDING_RESPONSE = {
    "object": "list",
    "data": [
        {"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0},
        {"object": "embedding", "embedding": [0.4, 0.5, 0.6], "index": 1},
    ],
    "model": "text-embedding-3-small",
    "usage": {"prompt_tokens": 5, "total_tokens": 5},
}

OPENAI_STREAM_CHUNKS = [
    'data: {"id":"chatcmpl-1","choices":[{"delta":{"role":"assistant"},"index":0,"finish_reason":null}],"model":"gpt-4o"}\n\n',
    'data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"Hello"},"index":0,"finish_reason":null}],"model":"gpt-4o"}\n\n',
    'data: {"id":"chatcmpl-1","choices":[{"delta":{"content":" world"},"index":0,"finish_reason":null}],"model":"gpt-4o"}\n\n',
    'data: {"id":"chatcmpl-1","choices":[{"delta":{},"index":0,"finish_reason":"stop"}],"model":"gpt-4o"}\n\n',
    "data: [DONE]\n\n",
]

ANTHROPIC_COMPLETION_RESPONSE = {
    "id": "msg_abc123",
    "type": "message",
    "role": "assistant",
    "model": "claude-3-haiku-20240307",
    "content": [{"type": "text", "text": "Hello! I'm Claude."}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 10, "output_tokens": 5},
}

ANTHROPIC_STREAM_CHUNKS = [
    'data: {"type":"message_start","message":{"id":"msg_1","role":"assistant","model":"claude-3-haiku-20240307"}}\n\n',
    'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n',
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}\n\n',
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}\n\n',
    'data: {"type":"content_block_stop","index":0}\n\n',
    'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}\n\n',
    'data: {"type":"message_stop"}\n\n',
]

GEMINI_COMPLETION_RESPONSE = {
    "candidates": [
        {
            "content": {
                "parts": [{"text": "Hello from Gemini!"}],
                "role": "model",
            },
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 8,
        "candidatesTokenCount": 4,
        "totalTokenCount": 12,
    },
}

GEMINI_STREAM_CHUNKS = [
    'data: {"candidates":[{"content":{"parts":[{"text":"Hello"}],"role":"model"}}]}\n\n',
    'data: {"candidates":[{"content":{"parts":[{"text":" Gemini"}],"role":"model"}}]}\n\n',
    'data: {"candidates":[{"content":{"parts":[{"text":"!"}],"role":"model"},"finishReason":"STOP"}]}\n\n',
]


# ── OpenAI-compat tests ──


class TestOpenAICompletion:
    @respx.mock
    def test_sync_completion(self):
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=OPENAI_COMPLETION_RESPONSE)
        )

        response = completion(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            api_key="sk-test",
            temperature=0.01,
        )

        assert response.choices[0].message.content == "Hello! How can I help?"
        assert response.choices[0].finish_reason == "stop"
        assert response.usage.prompt_tokens == 12
        assert response.usage.total_tokens == 18
        assert response.model == "gpt-4o"

    @respx.mock
    def test_sync_completion_with_json_mode(self):
        json_response = {
            **OPENAI_COMPLETION_RESPONSE,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": '{"name": "Alice", "age": 30}',
                    },
                    "finish_reason": "stop",
                }
            ],
        }
        route = respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=json_response)
        )

        response = completion(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": "Give me JSON"}],
            api_key="sk-test",
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        parsed = json.loads(content)
        assert parsed["name"] == "Alice"

        # Verify response_format was sent in the request body
        request_body = json.loads(route.calls[0].request.content)
        assert request_body["response_format"] == {"type": "json_object"}

    @respx.mock
    def test_sync_streaming(self):
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                stream=httpx.ByteStream(
                    "".join(OPENAI_STREAM_CHUNKS).encode()
                ),
                headers={"content-type": "text/event-stream"},
            )
        )

        chunks = list(
            completion(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": "Hi"}],
                api_key="sk-test",
                stream=True,
            )
        )

        # Collect content from chunks
        content_parts = []
        for chunk in chunks:
            c = chunk["choices"][0]["delta"].get("content")
            if c:
                content_parts.append(c)

        assert "".join(content_parts) == "Hello world"

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_completion(self):
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=OPENAI_COMPLETION_RESPONSE)
        )

        response = await acompletion(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            api_key="sk-test",
        )

        assert response.choices[0].message.content == "Hello! How can I help?"

    @respx.mock
    def test_batch_completion(self):
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=OPENAI_COMPLETION_RESPONSE)
        )

        messages_list = [
            [{"role": "user", "content": f"Message {i}"}] for i in range(3)
        ]

        responses = batch_completion(
            model="openai/gpt-4o",
            messages=messages_list,
            api_key="sk-test",
            temperature=0.01,
        )

        assert len(responses) == 3
        for r in responses:
            assert r.choices[0].message.content == "Hello! How can I help?"

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_embedding(self):
        respx.post("https://api.openai.com/v1/embeddings").mock(
            return_value=httpx.Response(200, json=OPENAI_EMBEDDING_RESPONSE)
        )

        response = await aembedding(
            model="openai/text-embedding-3-small",
            input=["Hello", "World"],
            api_key="sk-test",
        )

        assert len(response.data) == 2
        assert response.data[0]["embedding"] == [0.1, 0.2, 0.3]
        assert response.data[1]["embedding"] == [0.4, 0.5, 0.6]


# ── Groq (OpenAI-compat) tests ──


class TestGroqCompletion:
    @respx.mock
    def test_routes_to_groq(self):
        route = respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=OPENAI_COMPLETION_RESPONSE)
        )

        response = completion(
            model="groq/llama3-70b-8192",
            messages=[{"role": "user", "content": "Hello"}],
            api_key="gsk-test",
            temperature=0.01,
        )

        assert response.choices[0].message.content == "Hello! How can I help?"
        assert route.called

        # Verify it hit groq's URL
        request = route.calls[0].request
        assert "groq.com" in str(request.url)
        assert request.headers["Authorization"] == "Bearer gsk-test"


# ── DeepSeek (OpenAI-compat) tests ──


class TestDeepSeekCompletion:
    @respx.mock
    def test_routes_to_deepseek(self):
        route = respx.post("https://api.deepseek.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=OPENAI_COMPLETION_RESPONSE)
        )

        response = completion(
            model="deepseek/deepseek-chat",
            messages=[{"role": "user", "content": "Hello"}],
            api_key="ds-test",
        )

        assert response.choices[0].message.content == "Hello! How can I help?"
        assert route.called


# ── Anthropic tests ──


class TestAnthropicCompletion:
    @respx.mock
    def test_sync_completion(self):
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=ANTHROPIC_COMPLETION_RESPONSE)
        )

        response = completion(
            model="anthropic/claude-3-haiku-20240307",
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
            api_key="sk-ant-test",
            temperature=0.01,
        )

        assert response.choices[0].message.content == "Hello! I'm Claude."
        assert response.choices[0].finish_reason == "stop"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 5

        # Verify request format
        request_body = json.loads(route.calls[0].request.content)
        assert request_body["model"] == "claude-3-haiku-20240307"
        assert request_body["system"] == "You are helpful."
        assert len(request_body["messages"]) == 1  # system extracted
        assert request_body["messages"][0]["role"] == "user"

        # Verify headers
        assert route.calls[0].request.headers["x-api-key"] == "sk-ant-test"
        assert route.calls[0].request.headers["anthropic-version"] == "2023-06-01"

    @respx.mock
    def test_sync_streaming(self):
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(
                200,
                stream=httpx.ByteStream(
                    "".join(ANTHROPIC_STREAM_CHUNKS).encode()
                ),
                headers={"content-type": "text/event-stream"},
            )
        )

        chunks = list(
            completion(
                model="anthropic/claude-3-haiku-20240307",
                messages=[{"role": "user", "content": "Hi"}],
                api_key="sk-ant-test",
                stream=True,
            )
        )

        content_parts = []
        for chunk in chunks:
            c = chunk["choices"][0]["delta"].get("content")
            if c:
                content_parts.append(c)

        assert "".join(content_parts) == "Hello world"

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_completion(self):
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=ANTHROPIC_COMPLETION_RESPONSE)
        )

        response = await acompletion(
            model="anthropic/claude-3-haiku-20240307",
            messages=[{"role": "user", "content": "Hello"}],
            api_key="sk-ant-test",
        )

        assert response.choices[0].message.content == "Hello! I'm Claude."

    @respx.mock
    def test_multimodal_image(self):
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=ANTHROPIC_COMPLETION_RESPONSE)
        )

        response = completion(
            model="anthropic/claude-3-haiku-20240307",
            messages=[
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
            ],
            api_key="sk-ant-test",
        )

        # Verify the image was converted to Anthropic format
        request_body = json.loads(route.calls[0].request.content)
        content = request_body["messages"][0]["content"]
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image"
        assert content[1]["source"]["type"] == "base64"


# ── Gemini tests ──


class TestGeminiCompletion:
    @respx.mock
    def test_sync_completion(self):
        route = respx.post(url__regex=r".*gemini-2\.0-flash:generateContent.*").mock(
            return_value=httpx.Response(200, json=GEMINI_COMPLETION_RESPONSE)
        )

        response = completion(
            model="gemini/gemini-2.0-flash",
            messages=[
                {"role": "system", "content": "Be brief"},
                {"role": "user", "content": "Hello"},
            ],
            api_key="gemini-test-key",
            temperature=0.5,
        )

        assert response.choices[0].message.content == "Hello from Gemini!"
        assert response.choices[0].finish_reason == "stop"
        assert response.usage.prompt_tokens == 8

        # Verify request format
        request_body = json.loads(route.calls[0].request.content)
        assert "contents" in request_body
        assert "systemInstruction" in request_body
        assert request_body["generationConfig"]["temperature"] == 0.5

    @respx.mock
    def test_json_mode(self):
        route = respx.post(url__regex=r".*gemini.*generateContent.*").mock(
            return_value=httpx.Response(200, json=GEMINI_COMPLETION_RESPONSE)
        )

        completion(
            model="gemini/gemini-2.0-flash",
            messages=[{"role": "user", "content": "JSON"}],
            api_key="key",
            response_format={"type": "json_object"},
        )

        request_body = json.loads(route.calls[0].request.content)
        assert (
            request_body["generationConfig"]["responseMimeType"]
            == "application/json"
        )

    @respx.mock
    def test_streaming(self):
        respx.post(url__regex=r".*streamGenerateContent.*").mock(
            return_value=httpx.Response(
                200,
                stream=httpx.ByteStream(
                    "".join(GEMINI_STREAM_CHUNKS).encode()
                ),
                headers={"content-type": "text/event-stream"},
            )
        )

        chunks = list(
            completion(
                model="gemini/gemini-2.0-flash",
                messages=[{"role": "user", "content": "Hi"}],
                api_key="key",
                stream=True,
            )
        )

        content_parts = []
        for chunk in chunks:
            c = chunk["choices"][0]["delta"].get("content")
            if c:
                content_parts.append(c)

        assert "".join(content_parts) == "Hello Gemini!"


# ── Ollama tests ──


class TestOllamaCompletion:
    @respx.mock
    def test_routes_to_localhost(self):
        route = respx.post("http://localhost:11434/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=OPENAI_COMPLETION_RESPONSE)
        )

        response = completion(
            model="ollama/llama3",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert response.choices[0].message.content == "Hello! How can I help?"
        assert route.called

        # Verify NO auth header
        request = route.calls[0].request
        assert "Authorization" not in request.headers
        assert "api-key" not in request.headers


# ── Error handling tests ──


class TestErrorHandling:
    @respx.mock
    def test_rate_limit_raises_ratelimiterror(self):
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                429,
                json={"error": {"message": "Rate limit exceeded"}},
            )
        )

        with pytest.raises(RateLimitError) as exc_info:
            completion(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": "Hi"}],
                api_key="sk-test",
            )

        assert exc_info.value.status_code == 429
        assert "Rate limit" in exc_info.value.message

    @respx.mock
    def test_auth_error_raises_authenticationerror(self):
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                401,
                json={"error": {"message": "Invalid API key"}},
            )
        )

        with pytest.raises(AuthenticationError) as exc_info:
            completion(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": "Hi"}],
                api_key="bad-key",
            )

        assert exc_info.value.status_code == 401

    @respx.mock
    def test_litellm_ratelimiterror_compat(self):
        """Test that catching litellm.exceptions.RateLimitError works."""
        from litellm.exceptions import RateLimitError as LitellmRateLimitError

        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                429,
                json={"error": {"message": "Too many requests"}},
            )
        )

        with pytest.raises(LitellmRateLimitError):
            completion(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": "Hi"}],
                api_key="sk-test",
            )


# ── drop_params tests ──


class TestDropParams:
    @respx.mock
    def test_unsupported_params_stripped(self):
        route = respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=OPENAI_COMPLETION_RESPONSE)
        )

        nanollm.drop_params = True
        completion(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            api_key="sk-test",
            temperature=0.5,
            unsupported_custom_param="should_be_dropped",
            another_fake_param=42,
        )

        request_body = json.loads(route.calls[0].request.content)
        assert request_body["temperature"] == 0.5
        assert "unsupported_custom_param" not in request_body
        assert "another_fake_param" not in request_body

    @respx.mock
    def test_drop_params_disabled(self):
        route = respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=OPENAI_COMPLETION_RESPONSE)
        )

        nanollm.drop_params = False
        completion(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            api_key="sk-test",
            custom_param="should_be_kept",
        )

        request_body = json.loads(route.calls[0].request.content)
        assert "custom_param" in request_body

        # Reset
        nanollm.drop_params = True


OPENAI_COMPLETION_RESPONSE_WITH_DETAILS = {
    "id": "chatcmpl-abc456",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello!"},
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 12,
        "completion_tokens": 6,
        "total_tokens": 18,
        "completion_tokens_details": {
            "reasoning_tokens": 3,
            "accepted_prediction_tokens": 0,
            "rejected_prediction_tokens": 0,
        },
        "prompt_tokens_details": {
            "cached_tokens": 5,
        },
    },
}


# ── Crawl4ai simulation tests ──


class TestCrawl4aiSimulation:
    """Simulate the exact code paths crawl4ai uses."""

    @respx.mock
    def test_perform_completion_with_backoff_pattern(self):
        """Simulates crawl4ai/utils.py perform_completion_with_backoff."""
        from litellm import completion
        from litellm.exceptions import RateLimitError
        import litellm

        litellm.drop_params = True

        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=OPENAI_COMPLETION_RESPONSE)
        )

        provider = "openai/gpt-4o"
        api_token = "sk-test"
        base_url = None
        prompt_with_variables = "Extract data from this HTML"
        json_response = True

        extra_args = {"temperature": 0.01, "api_key": api_token, "base_url": base_url}
        if json_response:
            extra_args["response_format"] = {"type": "json_object"}

        response = completion(
            model=provider,
            messages=[{"role": "user", "content": prompt_with_variables}],
            **extra_args,
        )

        # This is how crawl4ai accesses the response
        content = response.choices[0].message.content
        assert content is not None

    @respx.mock
    @pytest.mark.asyncio
    async def test_aperform_completion_with_backoff_pattern(self):
        """Simulates crawl4ai/utils.py aperform_completion_with_backoff."""
        from litellm import acompletion
        from litellm.exceptions import RateLimitError
        import litellm

        litellm.drop_params = True

        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=OPENAI_COMPLETION_RESPONSE)
        )

        response = await acompletion(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": "test"}],
            temperature=0.01,
            api_key="sk-test",
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        assert content is not None

    @respx.mock
    def test_cli_streaming_pattern(self):
        """Simulates crawl4ai/cli.py streaming pattern."""
        from litellm import completion

        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                stream=httpx.ByteStream(
                    "".join(OPENAI_STREAM_CHUNKS).encode()
                ),
                headers={"content-type": "text/event-stream"},
            )
        )

        response = completion(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            api_key="sk-test",
            stream=True,
        )

        collected = []
        for chunk in response:
            if content := chunk["choices"][0]["delta"].get("content"):
                collected.append(content)

        assert "".join(collected) == "Hello world"

    @respx.mock
    def test_batch_completion_pattern(self):
        """Simulates crawl4ai/utils.py extract_blocks_batch."""
        from litellm import batch_completion

        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=OPENAI_COMPLETION_RESPONSE)
        )

        messages = [
            [{"role": "user", "content": f"Extract blocks from URL {i}"}]
            for i in range(3)
        ]

        responses = batch_completion(
            model="openai/gpt-4o",
            messages=messages,
            temperature=0.01,
            api_key="sk-test",
        )

        assert len(responses) == 3
        for response in responses:
            content = response.choices[0].message.content
            assert content is not None

    @respx.mock
    @pytest.mark.asyncio
    async def test_aembedding_pattern(self):
        """Simulates crawl4ai/utils.py compute_embeddings_litellm."""
        from litellm import aembedding

        respx.post("https://api.openai.com/v1/embeddings").mock(
            return_value=httpx.Response(200, json=OPENAI_EMBEDDING_RESPONSE)
        )

        response = await aembedding(
            model="openai/text-embedding-3-small",
            input=["text1", "text2"],
            api_key="sk-test",
        )

        # This is how crawl4ai accesses embeddings
        embeddings = []
        for item in response.data:
            embeddings.append(item["embedding"])

        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.4, 0.5, 0.6]

    @respx.mock
    def test_rate_limit_retry_pattern(self):
        """Simulates crawl4ai's retry logic with RateLimitError."""
        from litellm import completion
        from litellm.exceptions import RateLimitError

        call_count = 0

        def side_effect(request):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return httpx.Response(
                    429, json={"error": {"message": "Rate limited"}}
                )
            return httpx.Response(200, json=OPENAI_COMPLETION_RESPONSE)

        respx.post("https://api.openai.com/v1/chat/completions").mock(
            side_effect=side_effect
        )

        # Simulate crawl4ai's retry logic
        max_attempts = 3
        base_delay = 0.01  # Tiny delay for tests
        result = None

        for attempt in range(max_attempts):
            try:
                result = completion(
                    model="openai/gpt-4o",
                    messages=[{"role": "user", "content": "Hi"}],
                    api_key="sk-test",
                )
                break
            except RateLimitError:
                if attempt == max_attempts - 1:
                    raise
                import time
                time.sleep(base_delay)

        assert result is not None
        assert result.choices[0].message.content == "Hello! How can I help?"
        assert call_count == 3

    @respx.mock
    def test_extraction_strategy_token_details_pattern(self):
        """Simulates crawl4ai/extraction_strategy.py token details access.

        The exact pattern:
            completion_tokens_details=response.usage.completion_tokens_details.__dict__
            if response.usage.completion_tokens_details
            else {},
        """
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, json=OPENAI_COMPLETION_RESPONSE_WITH_DETAILS
            )
        )

        response = completion(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            api_key="sk-test",
        )

        # Exact crawl4ai pattern from extraction_strategy.py
        ctd = (
            response.usage.completion_tokens_details.__dict__
            if response.usage.completion_tokens_details
            else {}
        )
        ptd = (
            response.usage.prompt_tokens_details.__dict__
            if response.usage.prompt_tokens_details
            else {}
        )

        assert ctd == {
            "reasoning_tokens": 3,
            "accepted_prediction_tokens": 0,
            "rejected_prediction_tokens": 0,
        }
        assert ptd == {"cached_tokens": 5}

    @respx.mock
    def test_token_details_none_pattern(self):
        """When provider doesn't return details, guard produces empty dict."""
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=OPENAI_COMPLETION_RESPONSE)
        )

        response = completion(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            api_key="sk-test",
        )

        ctd = (
            response.usage.completion_tokens_details.__dict__
            if response.usage.completion_tokens_details
            else {}
        )
        ptd = (
            response.usage.prompt_tokens_details.__dict__
            if response.usage.prompt_tokens_details
            else {}
        )

        assert ctd == {}
        assert ptd == {}
