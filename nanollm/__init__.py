"""NanoLLM -- Minimal, zero-bloat LLM API wrapper.

Drop-in replacement for litellm.  Supports all major LLM providers
through a unified OpenAI-compatible interface with multimodal support.

Two ways to use:

1. **Client API** (recommended)::

    from nanollm import NanoLLM

    client = NanoLLM()
    response = client.complete("openai/gpt-4o", messages=[
        {"role": "user", "content": "Hello!"}
    ])

2. **Module-level functions** (litellm-compatible)::

    from nanollm import completion, acompletion

    response = completion(model="openai/gpt-4o", messages=[
        {"role": "user", "content": "Hello!"}
    ])
"""

from __future__ import annotations

__version__ = "1.0.0"

# -- Client API --------------------------------------------------------------

from .client import NanoLLM

# -- Types (slots-based, from types.py for provider use) ---------------------

from ._types import (
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
    stream_chunk_builder,
)

# -- Exceptions (all 19) ----------------------------------------------------

from .exceptions import (
    NanoLLMException,
    OpenAIError,
    RetryableError,
    AuthenticationError,
    PermissionDeniedError,
    InvalidRequestError,
    BadRequestError,
    NotFoundError,
    ContextWindowExceededError,
    ContentPolicyViolationError,
    UnsupportedParamsError,
    JSONSchemaValidationError,
    APIError,
    InternalServerError,
    BadGatewayError,
    ServiceUnavailableError,
    RateLimitError,
    APIConnectionError,
    Timeout,
    BudgetExceededError,
)

# -- Structured output utilities ---------------------------------------------

from ._structured import extract_json, validate_json_response

# -- Provider registry -------------------------------------------------------

from .providers import get_provider, list_providers

# -- Image utilities (exposed for advanced users) ----------------------------

from . import _image

# -- Module-level configuration (litellm compatibility) ----------------------

drop_params: bool = True
set_verbose: bool = False

# -- Default client (lazily created) -----------------------------------------

_default_client: NanoLLM | None = None


def _get_default_client() -> NanoLLM:
    """Lazily create the default client, proxying module-level config."""
    global _default_client
    if _default_client is None:
        _default_client = NanoLLM(drop_params=drop_params)
    return _default_client


# -- Module-level functions wrapping default client --------------------------


def completion(
    model: str,
    messages: list | None = None,
    *,
    stream: bool | None = None,
    timeout: float | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    api_base: str | None = None,
    **kwargs,
) -> ModelResponse:
    """Synchronous chat completion.

    Args:
        model: Provider/model string (e.g., "openai/gpt-4o").
        messages: List of message dicts.
        stream: Whether to stream the response.
        timeout: Request timeout in seconds.
        api_key: API key (overrides env var).
        base_url: Base URL (overrides provider default).
        api_base: Alias for base_url (litellm compat).
        **kwargs: Additional params (temperature, response_format, etc.)
    """
    effective_base_url = base_url or api_base
    return _get_default_client().complete(
        model=model, messages=messages,
        stream=bool(stream) if stream else False,
        timeout=timeout, api_key=api_key, base_url=effective_base_url,
        **kwargs,
    )


async def acompletion(
    model: str,
    messages: list | None = None,
    *,
    stream: bool | None = None,
    timeout: float | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    api_base: str | None = None,
    **kwargs,
) -> ModelResponse:
    """Asynchronous chat completion. Same parameters as completion()."""
    effective_base_url = base_url or api_base
    return await _get_default_client().acomplete(
        model=model, messages=messages,
        stream=bool(stream) if stream else False,
        timeout=timeout, api_key=api_key, base_url=effective_base_url,
        **kwargs,
    )


def batch_completion(
    model: str,
    messages: list[list[dict]],
    *,
    timeout: float | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    max_workers: int = 100,
    logger_fn=None,
    **kwargs,
) -> list[ModelResponse]:
    """Parallel batch completion using thread pool.

    Args:
        model: Provider/model string.
        messages: List of message lists (each is a separate completion call).
        max_workers: Max concurrent threads (default 100).
    """
    from concurrent.futures import ThreadPoolExecutor

    def _single(msgs: list[dict]) -> ModelResponse:
        return completion(
            model=model, messages=msgs,
            timeout=timeout, api_key=api_key, base_url=base_url,
            **kwargs,
        )

    with ThreadPoolExecutor(max_workers=min(max_workers, len(messages))) as executor:
        return list(executor.map(_single, messages))


def embedding(
    model: str,
    input: list[str] | str,
    *,
    timeout: float | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    base_url: str | None = None,
    **kwargs,
) -> EmbeddingResponse:
    """Synchronous embedding call."""
    if isinstance(input, str):
        input = [input]
    effective_base_url = base_url or api_base
    return _get_default_client().embed(
        model=model, input=input,
        timeout=timeout, api_key=api_key, base_url=effective_base_url,
        **kwargs,
    )


async def aembedding(
    model: str,
    input: list[str] | str,
    *,
    timeout: float | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    base_url: str | None = None,
    **kwargs,
) -> EmbeddingResponse:
    """Asynchronous embedding call."""
    if isinstance(input, str):
        input = [input]
    effective_base_url = base_url or api_base
    return await _get_default_client().aembed(
        model=model, input=input,
        timeout=timeout, api_key=api_key, base_url=effective_base_url,
        **kwargs,
    )


def text_completion(
    model: str,
    prompt: str,
    *,
    stream: bool | None = None,
    timeout: float | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    api_base: str | None = None,
    **kwargs,
) -> TextCompletionResponse | ModelResponse:
    """Synchronous text completion (prompt-based, not chat).

    Converts the prompt into a chat message internally.
    """
    messages = [{"role": "user", "content": prompt}]

    if stream:
        return completion(
            model=model, messages=messages, stream=True,
            timeout=timeout, api_key=api_key, base_url=base_url,
            api_base=api_base, **kwargs,
        )

    response = completion(
        model=model, messages=messages, stream=False,
        timeout=timeout, api_key=api_key, base_url=base_url,
        api_base=api_base, **kwargs,
    )

    text = response.choices[0].message.content if response.choices else ""
    return TextCompletionResponse(
        id=response.id,
        choices=[TextChoice(
            text=text or "",
            index=0,
            finish_reason=response.choices[0].finish_reason if response.choices else None,
        )],
        model=response.model,
        usage=response.usage,
    )


async def atext_completion(
    model: str,
    prompt: str,
    *,
    stream: bool | None = None,
    timeout: float | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    api_base: str | None = None,
    **kwargs,
) -> TextCompletionResponse | ModelResponse:
    """Asynchronous text completion. Same as text_completion() but async."""
    messages = [{"role": "user", "content": prompt}]

    if stream:
        return await acompletion(
            model=model, messages=messages, stream=True,
            timeout=timeout, api_key=api_key, base_url=base_url,
            api_base=api_base, **kwargs,
        )

    response = await acompletion(
        model=model, messages=messages, stream=False,
        timeout=timeout, api_key=api_key, base_url=base_url,
        api_base=api_base, **kwargs,
    )

    text = response.choices[0].message.content if response.choices else ""
    return TextCompletionResponse(
        id=response.id,
        choices=[TextChoice(
            text=text or "",
            index=0,
            finish_reason=response.choices[0].finish_reason if response.choices else None,
        )],
        model=response.model,
        usage=response.usage,
    )


# -- Public API --------------------------------------------------------------

__all__ = [
    # Client
    "NanoLLM",
    # Module-level functions
    "completion",
    "acompletion",
    "batch_completion",
    "embedding",
    "aembedding",
    "text_completion",
    "atext_completion",
    # Types
    "ModelResponse",
    "TextCompletionResponse",
    "EmbeddingResponse",
    "EmbeddingData",
    "Message",
    "Choice",
    "Delta",
    "StreamChoice",
    "Usage",
    "PromptTokensDetails",
    "CompletionTokensDetails",
    "ToolCall",
    "FunctionCall",
    "TextChoice",
    "stream_chunk_builder",
    # Exceptions
    "NanoLLMException",
    "OpenAIError",
    "RetryableError",
    "AuthenticationError",
    "PermissionDeniedError",
    "InvalidRequestError",
    "BadRequestError",
    "NotFoundError",
    "ContextWindowExceededError",
    "ContentPolicyViolationError",
    "UnsupportedParamsError",
    "JSONSchemaValidationError",
    "APIError",
    "InternalServerError",
    "BadGatewayError",
    "ServiceUnavailableError",
    "RateLimitError",
    "APIConnectionError",
    "Timeout",
    "BudgetExceededError",
    # Structured output
    "extract_json",
    "validate_json_response",
    # Providers
    "get_provider",
    "list_providers",
    # Image utilities
    "_image",
    # Version
    "__version__",
]
