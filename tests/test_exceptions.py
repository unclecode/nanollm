"""Exhaustive tests for nanollm.exceptions -- ~60+ tests."""
from __future__ import annotations

import pytest

from nanollm.exceptions import (
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
    STATUS_CODE_MAP,
    raise_for_status,
)


# ── Base exception ───────────────────────────────────────────────────


class TestNanoLLMException:
    def test_is_exception(self):
        assert issubclass(NanoLLMException, Exception)

    def test_message(self):
        e = NanoLLMException("oops")
        assert e.message == "oops"

    def test_status_code(self):
        e = NanoLLMException("err", status_code=500)
        assert e.status_code == 500

    def test_provider(self):
        e = NanoLLMException("err", llm_provider="openai")
        assert e.llm_provider == "openai"

    def test_model(self):
        e = NanoLLMException("err", model="gpt-4")
        assert e.model == "gpt-4"

    def test_str(self):
        e = NanoLLMException("err", llm_provider="openai", model="gpt-4")
        s = str(e)
        assert "err" in s
        assert "openai" in s
        assert "gpt-4" in s

    def test_str_no_provider(self):
        e = NanoLLMException("err")
        assert str(e) == "err"

    def test_repr(self):
        e = NanoLLMException("err", status_code=500)
        r = repr(e)
        assert "NanoLLMException" in r
        assert "500" in r

    def test_defaults(self):
        e = NanoLLMException()
        assert e.message == ""
        assert e.status_code is None
        assert e.llm_provider is None
        assert e.model is None


# ── OpenAIError alias ────────────────────────────────────────────────


class TestOpenAIErrorAlias:
    def test_is_same_as_base(self):
        assert OpenAIError is NanoLLMException

    def test_catch_as_openai_error(self):
        with pytest.raises(OpenAIError):
            raise NanoLLMException("test")


# ── RetryableError mixin ────────────────────────────────────────────


class TestRetryableError:
    def test_rate_limit_is_retryable(self):
        assert issubclass(RateLimitError, RetryableError)

    def test_internal_server_is_retryable(self):
        assert issubclass(InternalServerError, RetryableError)

    def test_service_unavailable_is_retryable(self):
        assert issubclass(ServiceUnavailableError, RetryableError)

    def test_api_connection_is_retryable(self):
        assert issubclass(APIConnectionError, RetryableError)

    def test_timeout_is_retryable(self):
        assert issubclass(Timeout, RetryableError)

    def test_invalid_request_is_NOT_retryable(self):
        assert not issubclass(InvalidRequestError, RetryableError)

    def test_auth_error_is_NOT_retryable(self):
        assert not issubclass(AuthenticationError, RetryableError)

    def test_not_found_is_NOT_retryable(self):
        assert not issubclass(NotFoundError, RetryableError)

    def test_bad_gateway_is_NOT_retryable(self):
        assert not issubclass(BadGatewayError, RetryableError)

    def test_budget_is_NOT_retryable(self):
        assert not issubclass(BudgetExceededError, RetryableError)

    def test_permission_denied_is_NOT_retryable(self):
        assert not issubclass(PermissionDeniedError, RetryableError)

    def test_isinstance_check(self):
        e = RateLimitError()
        assert isinstance(e, RetryableError)

    def test_isinstance_check_negative(self):
        e = InvalidRequestError()
        assert not isinstance(e, RetryableError)


# ── Specific exceptions ─────────────────────────────────────────────


class TestAuthenticationError:
    def test_default_status(self):
        e = AuthenticationError()
        assert e.status_code == 401

    def test_default_message(self):
        e = AuthenticationError()
        assert "Authentication" in e.message

    def test_is_base(self):
        assert issubclass(AuthenticationError, NanoLLMException)


class TestPermissionDeniedError:
    def test_default_status(self):
        e = PermissionDeniedError()
        assert e.status_code == 403


class TestInvalidRequestError:
    def test_default_status(self):
        e = InvalidRequestError()
        assert e.status_code == 400

    def test_custom_status(self):
        e = InvalidRequestError(status_code=422)
        assert e.status_code == 422


class TestBadRequestError:
    def test_default_status(self):
        e = BadRequestError()
        assert e.status_code == 400

    def test_is_invalid_request(self):
        assert issubclass(BadRequestError, InvalidRequestError)


class TestNotFoundError:
    def test_default_status(self):
        e = NotFoundError()
        assert e.status_code == 404


class TestContextWindowExceeded:
    def test_is_invalid_request(self):
        assert issubclass(ContextWindowExceededError, InvalidRequestError)

    def test_catchable_as_parent(self):
        with pytest.raises(InvalidRequestError):
            raise ContextWindowExceededError("too long")


class TestContentPolicyViolation:
    def test_is_invalid_request(self):
        assert issubclass(ContentPolicyViolationError, InvalidRequestError)


class TestUnsupportedParams:
    def test_is_invalid_request(self):
        assert issubclass(UnsupportedParamsError, InvalidRequestError)


class TestJSONSchemaValidation:
    def test_is_invalid_request(self):
        assert issubclass(JSONSchemaValidationError, InvalidRequestError)


class TestAPIError:
    def test_is_base(self):
        assert issubclass(APIError, NanoLLMException)


class TestInternalServerError:
    def test_default_status(self):
        e = InternalServerError()
        assert e.status_code == 500

    def test_is_api_error(self):
        assert issubclass(InternalServerError, APIError)


class TestBadGatewayError:
    def test_default_status(self):
        e = BadGatewayError()
        assert e.status_code == 502


class TestServiceUnavailableError:
    def test_default_status(self):
        e = ServiceUnavailableError()
        assert e.status_code == 503


class TestRateLimitError:
    def test_default_status(self):
        e = RateLimitError()
        assert e.status_code == 429

    def test_is_base(self):
        assert issubclass(RateLimitError, NanoLLMException)


class TestAPIConnectionError:
    def test_default_status_none(self):
        e = APIConnectionError()
        assert e.status_code is None


class TestTimeout:
    def test_is_connection_error(self):
        assert issubclass(Timeout, APIConnectionError)

    def test_default_message(self):
        e = Timeout()
        assert "timed out" in e.message.lower()


class TestBudgetExceededError:
    def test_is_base(self):
        assert issubclass(BudgetExceededError, NanoLLMException)


# ── STATUS_CODE_MAP ──────────────────────────────────────────────────


class TestStatusCodeMap:
    def test_400(self):
        assert STATUS_CODE_MAP[400] is BadRequestError

    def test_401(self):
        assert STATUS_CODE_MAP[401] is AuthenticationError

    def test_403(self):
        assert STATUS_CODE_MAP[403] is PermissionDeniedError

    def test_404(self):
        assert STATUS_CODE_MAP[404] is NotFoundError

    def test_408(self):
        assert STATUS_CODE_MAP[408] is Timeout

    def test_422(self):
        assert STATUS_CODE_MAP[422] is InvalidRequestError

    def test_429(self):
        assert STATUS_CODE_MAP[429] is RateLimitError

    def test_500(self):
        assert STATUS_CODE_MAP[500] is InternalServerError

    def test_502(self):
        assert STATUS_CODE_MAP[502] is BadGatewayError

    def test_503(self):
        assert STATUS_CODE_MAP[503] is ServiceUnavailableError


# ── raise_for_status ─────────────────────────────────────────────────


class TestRaiseForStatus:
    def test_200_no_raise(self):
        raise_for_status(200, {})

    def test_201_no_raise(self):
        raise_for_status(201, {})

    def test_204_no_raise(self):
        raise_for_status(204, "")

    def test_400_raises_bad_request(self):
        with pytest.raises(BadRequestError):
            raise_for_status(400, {"error": {"message": "bad"}})

    def test_401_raises_auth(self):
        with pytest.raises(AuthenticationError):
            raise_for_status(401, {})

    def test_403_raises_permission(self):
        with pytest.raises(PermissionDeniedError):
            raise_for_status(403, {})

    def test_404_raises_not_found(self):
        with pytest.raises(NotFoundError):
            raise_for_status(404, {})

    def test_429_raises_rate_limit(self):
        with pytest.raises(RateLimitError):
            raise_for_status(429, {})

    def test_500_raises_internal(self):
        with pytest.raises(InternalServerError):
            raise_for_status(500, {})

    def test_unknown_4xx_raises_invalid_request(self):
        with pytest.raises(InvalidRequestError):
            raise_for_status(418, {})

    def test_unknown_5xx_raises_api_error(self):
        with pytest.raises(APIError):
            raise_for_status(599, {})

    def test_unknown_other_raises_base(self):
        with pytest.raises(NanoLLMException):
            raise_for_status(600, {})

    def test_error_message_dict_nested(self):
        with pytest.raises(BadRequestError, match="bad param"):
            raise_for_status(400, {"error": {"message": "bad param"}})

    def test_error_message_dict_msg(self):
        with pytest.raises(BadRequestError, match="bad msg"):
            raise_for_status(400, {"error": {"msg": "bad msg"}})

    def test_error_message_top_level(self):
        with pytest.raises(BadRequestError, match="top msg"):
            raise_for_status(400, {"message": "top msg"})

    def test_error_message_detail(self):
        with pytest.raises(BadRequestError, match="detail msg"):
            raise_for_status(400, {"detail": "detail msg"})

    def test_error_message_string_body(self):
        with pytest.raises(BadRequestError, match="raw string"):
            raise_for_status(400, "raw string")

    def test_provider_and_model_set(self):
        try:
            raise_for_status(400, {}, provider="openai", model="gpt-4")
        except BadRequestError as e:
            assert e.llm_provider == "openai"
            assert e.model == "gpt-4"

    def test_status_code_set(self):
        try:
            raise_for_status(500, {})
        except InternalServerError as e:
            assert e.status_code == 500


# ── Backward compat (parent catches child) ───────────────────────────


class TestBackwardCompat:
    def test_catch_bad_request_as_invalid_request(self):
        with pytest.raises(InvalidRequestError):
            raise BadRequestError("test")

    def test_catch_context_window_as_invalid_request(self):
        with pytest.raises(InvalidRequestError):
            raise ContextWindowExceededError("too long")

    def test_catch_internal_as_api_error(self):
        with pytest.raises(APIError):
            raise InternalServerError("fail")

    def test_catch_timeout_as_connection_error(self):
        with pytest.raises(APIConnectionError):
            raise Timeout("timed out")

    def test_catch_all_as_base(self):
        for exc_cls in [
            AuthenticationError, PermissionDeniedError, InvalidRequestError,
            BadRequestError, NotFoundError, APIError, InternalServerError,
            BadGatewayError, ServiceUnavailableError, RateLimitError,
            APIConnectionError, Timeout, BudgetExceededError,
        ]:
            with pytest.raises(NanoLLMException):
                raise exc_cls("test")

    def test_catch_all_as_openai_error(self):
        for exc_cls in [AuthenticationError, RateLimitError, APIError]:
            with pytest.raises(OpenAIError):
                raise exc_cls("test")


# ── Additional edge cases ────────────────────────────────────────────


class TestExceptionEdgeCases:
    def test_raise_for_status_unknown_418(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            raise_for_status(418, "I'm a teapot")
        assert exc_info.value.status_code == 418

    def test_raise_for_status_502(self):
        with pytest.raises(BadGatewayError):
            raise_for_status(502, {})

    def test_raise_for_status_503(self):
        with pytest.raises(ServiceUnavailableError):
            raise_for_status(503, {})

    def test_raise_for_status_408(self):
        with pytest.raises(Timeout):
            raise_for_status(408, {})

    def test_raise_for_status_422(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            raise_for_status(422, {})
        assert exc_info.value.status_code == 422

    def test_exception_str_only_message(self):
        e = RateLimitError("limited")
        s = str(e)
        assert "limited" in s

    def test_exception_repr_has_class_name(self):
        e = AuthenticationError("bad key")
        assert "AuthenticationError" in repr(e)

    def test_retryable_mixin_is_not_exception(self):
        assert not issubclass(RetryableError, Exception)

    def test_raise_for_status_fallback_dict(self):
        with pytest.raises(BadRequestError):
            raise_for_status(400, {"error": {}})

    def test_context_window_status_code(self):
        e = ContextWindowExceededError("too long")
        assert e.status_code == 400  # inherited default

    def test_content_policy_status_code(self):
        e = ContentPolicyViolationError("blocked")
        assert e.status_code == 400

    def test_unsupported_params_status_code(self):
        e = UnsupportedParamsError("unsupported")
        assert e.status_code == 400

    def test_json_schema_validation_status_code(self):
        e = JSONSchemaValidationError("invalid schema")
        assert e.status_code == 400

    def test_budget_exceeded_default(self):
        e = BudgetExceededError("over budget")
        assert e.message == "over budget"
