"""NanoLLM exception hierarchy.

Maps HTTP status codes to specific exception types for consistent
error handling across all providers.
"""

from __future__ import annotations


class NanoLLMException(Exception):
    """Base exception for all NanoLLM errors."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        llm_provider: str | None = None,
        model: str | None = None,
    ):
        self.message = message
        self.status_code = status_code
        self.llm_provider = llm_provider
        self.model = model
        super().__init__(message)


class RateLimitError(NanoLLMException):
    """HTTP 429 — rate limit exceeded."""
    pass


class AuthenticationError(NanoLLMException):
    """HTTP 401/403 — invalid or missing credentials."""
    pass


class InvalidRequestError(NanoLLMException):
    """HTTP 400 — malformed request or invalid parameters."""
    pass


class APIError(NanoLLMException):
    """Generic API error for unexpected status codes."""
    pass


class ServiceUnavailableError(NanoLLMException):
    """HTTP 503 — provider temporarily unavailable."""
    pass


class ContextWindowExceededError(InvalidRequestError):
    """Input exceeds the model's context window."""
    pass


# Status code → exception class mapping
STATUS_CODE_MAP: dict[int, type[NanoLLMException]] = {
    400: InvalidRequestError,
    401: AuthenticationError,
    403: AuthenticationError,
    404: APIError,
    408: APIError,
    422: InvalidRequestError,
    429: RateLimitError,
    500: APIError,
    503: ServiceUnavailableError,
}


def raise_for_status(
    status_code: int,
    body: dict | str,
    provider: str | None = None,
    model: str | None = None,
) -> None:
    """Raise the appropriate exception for an HTTP error response."""
    if 200 <= status_code < 300:
        return

    if isinstance(body, dict):
        message = (
            body.get("error", {}).get("message")
            or body.get("error", {}).get("msg")
            or body.get("message")
            or body.get("detail")
            or str(body)
        )
    else:
        message = str(body)

    exc_class = STATUS_CODE_MAP.get(status_code, APIError)
    raise exc_class(
        message=message,
        status_code=status_code,
        llm_provider=provider,
        model=model,
    )
