"""NanoLLM exception hierarchy.

Maps HTTP status codes to specific exception types for consistent error
handling across all providers.  Retryable exceptions also inherit from
the ``RetryableError`` mixin so callers can decide whether to retry
automatically with a simple ``isinstance`` check.

Hierarchy::

    NanoLLMException (base)
    +-- AuthenticationError          (401)
    +-- PermissionDeniedError        (403)
    +-- InvalidRequestError          (400/422)
    |   +-- BadRequestError          (400, litellm compat alias)
    |   +-- ContextWindowExceededError
    |   +-- ContentPolicyViolationError
    |   +-- UnsupportedParamsError
    |   +-- JSONSchemaValidationError
    +-- NotFoundError                (404)
    +-- APIError                     (generic server-side)
    |   +-- InternalServerError*     (500, retryable)
    |   +-- BadGatewayError          (502)
    |   +-- ServiceUnavailableError* (503, retryable)
    +-- RateLimitError*              (429, retryable)
    +-- APIConnectionError*          (network, retryable)
    |   +-- Timeout*                 (408 / request timeout, retryable)
    +-- BudgetExceededError

    * = also inherits RetryableError

    OpenAIError = NanoLLMException   (litellm compat alias)
"""

from __future__ import annotations


# ── Retryable mixin ──────────────────────────────────────────────────


class RetryableError:
    """Mixin that marks an exception as safe to retry with backoff."""

    pass


# ── Base exception ───────────────────────────────────────────────────


class NanoLLMException(Exception):
    """Base exception for all NanoLLM errors."""

    def __init__(
        self,
        message: str = "",
        *,
        status_code: int | None = None,
        llm_provider: str | None = None,
        model: str | None = None,
    ):
        self.message = message
        self.status_code = status_code
        self.llm_provider = llm_provider
        self.model = model
        super().__init__(message)

    def __str__(self) -> str:
        parts = [self.message]
        if self.llm_provider:
            parts.append(f"Provider: {self.llm_provider}")
        if self.model:
            parts.append(f"Model: {self.model}")
        return " | ".join(parts)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(message={self.message!r}, "
            f"status_code={self.status_code}, "
            f"llm_provider={self.llm_provider!r}, "
            f"model={self.model!r})"
        )


# litellm backward-compat alias
OpenAIError = NanoLLMException


# ── Auth / permissions ───────────────────────────────────────────────


class AuthenticationError(NanoLLMException):
    """HTTP 401 -- invalid or missing credentials."""

    def __init__(self, message: str = "Authentication failed", **kw):
        kw.setdefault("status_code", 401)
        super().__init__(message, **kw)


class PermissionDeniedError(NanoLLMException):
    """HTTP 403 -- access denied."""

    def __init__(self, message: str = "Permission denied", **kw):
        kw.setdefault("status_code", 403)
        super().__init__(message, **kw)


# ── Client errors ────────────────────────────────────────────────────


class InvalidRequestError(NanoLLMException):
    """HTTP 400/422 -- malformed request or invalid parameters."""

    def __init__(self, message: str = "Invalid request", **kw):
        kw.setdefault("status_code", 400)
        super().__init__(message, **kw)


class BadRequestError(InvalidRequestError):
    """HTTP 400 -- alias for InvalidRequestError (litellm compat)."""

    def __init__(self, message: str = "Bad request", **kw):
        kw.setdefault("status_code", 400)
        super().__init__(message, **kw)


class NotFoundError(NanoLLMException):
    """HTTP 404 -- requested resource not found."""

    def __init__(self, message: str = "Resource not found", **kw):
        kw.setdefault("status_code", 404)
        super().__init__(message, **kw)


class ContextWindowExceededError(InvalidRequestError):
    """Input exceeds the model's context window."""

    pass


class ContentPolicyViolationError(InvalidRequestError):
    """Response blocked by content policy / safety filter."""

    pass


class UnsupportedParamsError(InvalidRequestError):
    """Request contains parameters not supported by the provider."""

    pass


class JSONSchemaValidationError(InvalidRequestError):
    """JSON schema validation failed."""

    pass


# ── Server errors ────────────────────────────────────────────────────


class APIError(NanoLLMException):
    """Generic API error for unexpected status codes."""

    pass


class InternalServerError(RetryableError, APIError):
    """HTTP 500 -- server-side error, may resolve on retry."""

    def __init__(self, message: str = "Internal server error", **kw):
        kw.setdefault("status_code", 500)
        super().__init__(message, **kw)


class BadGatewayError(APIError):
    """HTTP 502 -- bad gateway."""

    def __init__(self, message: str = "Bad gateway", **kw):
        kw.setdefault("status_code", 502)
        super().__init__(message, **kw)


class ServiceUnavailableError(RetryableError, APIError):
    """HTTP 503 -- provider temporarily unavailable."""

    def __init__(self, message: str = "Service unavailable", **kw):
        kw.setdefault("status_code", 503)
        super().__init__(message, **kw)


# ── Rate limiting ────────────────────────────────────────────────────


class RateLimitError(RetryableError, NanoLLMException):
    """HTTP 429 -- rate limit exceeded, should retry with backoff."""

    def __init__(self, message: str = "Rate limited", **kw):
        kw.setdefault("status_code", 429)
        super().__init__(message, **kw)


# ── Connection / timeout ─────────────────────────────────────────────


class APIConnectionError(RetryableError, NanoLLMException):
    """Connection error -- network unreachable, DNS failure, etc."""

    def __init__(self, message: str = "Connection failed", **kw):
        kw.setdefault("status_code", None)
        super().__init__(message, **kw)


class Timeout(APIConnectionError):
    """HTTP 408 or request timeout."""

    def __init__(self, message: str = "Request timed out", **kw):
        super().__init__(message, **kw)


# ── Budget ───────────────────────────────────────────────────────────


class BudgetExceededError(NanoLLMException):
    """Budget limit exceeded."""

    pass


# ── Status code mapping ─────────────────────────────────────────────


STATUS_CODE_MAP: dict[int, type[NanoLLMException]] = {
    400: BadRequestError,
    401: AuthenticationError,
    403: PermissionDeniedError,
    404: NotFoundError,
    408: Timeout,
    422: InvalidRequestError,
    429: RateLimitError,
    500: InternalServerError,
    502: BadGatewayError,
    503: ServiceUnavailableError,
}


def raise_for_status(
    status_code: int,
    body: dict | str,
    *,
    provider: str | None = None,
    model: str | None = None,
) -> None:
    """Raise the appropriate exception for an HTTP error response.

    Does nothing for 2xx status codes.  Extracts the error message from
    common JSON error-body layouts before raising.
    """
    if 200 <= status_code < 300:
        return

    # Extract human-readable message from the response body
    if isinstance(body, dict):
        error_val = body.get("error", {})
        if isinstance(error_val, dict):
            error_msg = error_val.get("message") or error_val.get("msg")
        elif isinstance(error_val, str):
            error_msg = error_val
        else:
            error_msg = None
        message = (
            error_msg
            or body.get("message")
            or body.get("detail")
            or str(body)
        )
    else:
        message = str(body)

    exc_class = STATUS_CODE_MAP.get(status_code)
    if exc_class is None:
        if 400 <= status_code < 500:
            exc_class = InvalidRequestError
        elif 500 <= status_code < 600:
            exc_class = APIError
        else:
            exc_class = NanoLLMException

    raise exc_class(
        message=message,
        status_code=status_code,
        llm_provider=provider,
        model=model,
    )
