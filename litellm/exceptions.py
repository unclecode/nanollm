"""litellm.exceptions compatibility shim.

Re-exports nanollm exceptions under the litellm.exceptions namespace.
"""

from nanollm.exceptions import (
    APIError,
    AuthenticationError,
    ContextWindowExceededError,
    InvalidRequestError,
    NanoLLMException,
    RateLimitError,
    ServiceUnavailableError,
)

__all__ = [
    "APIError",
    "AuthenticationError",
    "ContextWindowExceededError",
    "InvalidRequestError",
    "NanoLLMException",
    "RateLimitError",
    "ServiceUnavailableError",
]
