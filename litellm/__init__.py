"""litellm compatibility shim.

Re-exports nanollm's public API under the ``litellm`` namespace so that
existing code using ``from litellm import completion`` works unchanged.
"""

from nanollm import (
    __version__,
    NanoLLM,
    # Module-level functions
    acompletion,
    aembedding,
    atext_completion,
    batch_completion,
    completion,
    embedding,
    text_completion,
    # Types
    stream_chunk_builder,
    EmbeddingResponse,
    ModelResponse,
    TextCompletionResponse,
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
    # Structured output
    extract_json,
    validate_json_response,
)
from nanollm import drop_params, set_verbose
from nanollm.exceptions import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    BadGatewayError,
    BadRequestError,
    BudgetExceededError,
    ContentPolicyViolationError,
    ContextWindowExceededError,
    InternalServerError,
    InvalidRequestError,
    JSONSchemaValidationError,
    NanoLLMException,
    NotFoundError,
    OpenAIError,
    PermissionDeniedError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
    UnsupportedParamsError,
)

import nanollm as _nanollm
import sys


class _LitellmModule(sys.modules[__name__].__class__):
    """Module class that proxies drop_params/set_verbose to nanollm."""

    @property
    def drop_params(self):
        return _nanollm.drop_params

    @drop_params.setter
    def drop_params(self, value):
        _nanollm.drop_params = value

    @property
    def set_verbose(self):
        return _nanollm.set_verbose

    @set_verbose.setter
    def set_verbose(self, value):
        _nanollm.set_verbose = value


sys.modules[__name__].__class__ = _LitellmModule
