"""Base provider class -- defines the interface all providers implement.

Every provider inherits from ``BaseProvider`` and overrides the methods
that differ from the default OpenAI-compatible behavior.  The base class
handles the common case (OpenAI-format requests/responses) so most
providers only need to customize a handful of methods.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from .._types import (
    Choice,
    CompletionTokensDetails,
    Delta,
    EmbeddingData,
    EmbeddingResponse,
    FunctionCall,
    Message,
    ModelResponse,
    PromptTokensDetails,
    StreamChoice,
    ToolCall,
    Usage,
)


class BaseProvider:
    """Abstract base for all LLM providers.

    Subclasses **must** set :attr:`name`, :attr:`base_url`, and
    :attr:`api_key_env`.  Override methods to customize request building,
    response parsing, and parameter mapping.
    """

    # ── Class-level config (override in subclasses) ──────────────────

    name: str = ""
    base_url: str = ""
    api_key_env: str = ""

    # Parameters this provider actually supports.  If non-empty,
    # :meth:`filter_params` will drop anything not in this set.
    supported_params: frozenset[str] = frozenset()

    # ── API key resolution ───────────────────────────────────────────

    def get_api_key(self, api_key: Optional[str] = None) -> str:
        """Resolve the API key: explicit arg > environment variable."""
        if api_key:
            return api_key
        if self.api_key_env:
            val = os.environ.get(self.api_key_env, "")
            if val:
                return val
        return ""

    # ── Request building ─────────────────────────────────────────────

    def build_headers(self, api_key: str) -> dict[str, str]:
        """Build HTTP headers for a request."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def build_url(
        self,
        model: str,
        endpoint: str = "chat/completions",
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Build the full request URL."""
        base = (base_url or self.base_url).rstrip("/")
        return f"{base}/{endpoint}"

    def build_body(
        self,
        model: str,
        messages: list,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict:
        """Build the request body (OpenAI format by default)."""
        body: dict[str, Any] = {
            "model": model,
            "messages": self.transform_messages(messages),
            "stream": stream,
        }

        # Apply standard params
        mapped = self.map_params(**kwargs)
        filtered = self.filter_params(mapped)
        body.update(filtered)

        return body

    # ── Message / param transformation ───────────────────────────────

    def transform_messages(self, messages: list) -> list | dict:
        """Transform OpenAI-format messages to provider format.

        Default: passthrough (OpenAI format).
        """
        return messages

    def transform_image_block(self, block: dict) -> dict:
        """Transform an ``image_url`` content block to provider format.

        Default: passthrough (OpenAI format).
        """
        return block

    def map_params(self, **kwargs: Any) -> dict:
        """Map standard parameter names to provider-specific names.

        Default: passthrough.
        """
        return kwargs

    def map_thinking(self, reasoning_effort: str) -> dict:
        """Map ``reasoning_effort`` to provider-specific thinking config.

        Returns a dict to merge into the request body.
        """
        return {}

    def map_response_format(self, response_format: dict) -> tuple[dict, dict]:
        """Map ``response_format`` to provider format.

        Returns ``(body_updates, header_updates)``.
        """
        return {"response_format": response_format}, {}

    def filter_params(self, kwargs: dict) -> dict:
        """Filter out unsupported parameters.

        If :attr:`supported_params` is set, only keeps params in that set.
        Always drops ``None`` values.
        """
        result = {k: v for k, v in kwargs.items() if v is not None}
        if self.supported_params:
            result = {
                k: v for k, v in result.items() if k in self.supported_params
            }
        return result

    # ── Response parsing ─────────────────────────────────────────────

    def parse_response(self, data: dict, model: str = "") -> ModelResponse:
        """Parse a provider JSON response into a ``ModelResponse``.

        Default implementation handles standard OpenAI-format responses.
        """
        choices = []
        for c in data.get("choices", []):
            msg_data = c.get("message", {})
            tool_calls = None
            if msg_data.get("tool_calls"):
                tool_calls = [
                    ToolCall(
                        id=tc.get("id", ""),
                        type=tc.get("type", "function"),
                        function=FunctionCall(
                            name=tc.get("function", {}).get("name", ""),
                            arguments=tc.get("function", {}).get(
                                "arguments", ""
                            ),
                        ),
                    )
                    for tc in msg_data["tool_calls"]
                ]

            msg = Message(
                role=msg_data.get("role", "assistant"),
                content=msg_data.get("content"),
                tool_calls=tool_calls,
                reasoning_content=msg_data.get("reasoning_content"),
            )
            choices.append(
                Choice(
                    index=c.get("index", 0),
                    message=msg,
                    finish_reason=c.get("finish_reason"),
                )
            )

        usage = self._parse_usage(data.get("usage")) if "usage" in data else None

        return ModelResponse(
            id=data.get("id"),
            model=data.get("model", model),
            choices=choices or [Choice()],
            usage=usage,
            system_fingerprint=data.get("system_fingerprint"),
        )

    def parse_stream_line(
        self, data: dict, model: str = ""
    ) -> ModelResponse | None:
        """Parse a single SSE data dict into a streaming ``ModelResponse``.

        Returns ``None`` if the line should be skipped.
        """
        if not data or not data.get("choices"):
            # Could be a usage-only chunk
            if data and "usage" in data:
                usage = self._parse_usage(data["usage"])
                return ModelResponse(
                    id=data.get("id"),
                    model=data.get("model", model),
                    choices=[StreamChoice(delta=Delta())],
                    usage=usage,
                    stream=True,
                )
            return None

        choices = []
        for c in data.get("choices", []):
            delta_data = c.get("delta", {})
            tool_calls = None
            if delta_data.get("tool_calls"):
                tool_calls = []
                for tc in delta_data["tool_calls"]:
                    fn_data = tc.get("function", {})
                    tool_calls.append(
                        ToolCall(
                            id=tc.get("id", ""),
                            type=tc.get("type", "function"),
                            function=FunctionCall(
                                name=fn_data.get("name", ""),
                                arguments=fn_data.get("arguments", ""),
                            ),
                        )
                    )

            delta = Delta(
                role=delta_data.get("role"),
                content=delta_data.get("content"),
                tool_calls=tool_calls,
                reasoning_content=delta_data.get("reasoning_content"),
            )
            choices.append(
                StreamChoice(
                    index=c.get("index", 0),
                    delta=delta,
                    finish_reason=c.get("finish_reason"),
                )
            )

        usage = None
        if "usage" in data and data["usage"]:
            usage = self._parse_usage(data["usage"])

        return ModelResponse(
            id=data.get("id"),
            model=data.get("model", model),
            choices=choices,
            usage=usage,
            stream=True,
        )

    def _parse_usage(self, u: dict | None) -> Usage | None:
        """Parse a usage dict into a ``Usage`` object."""
        if not u:
            return None

        ptd = None
        if u.get("prompt_tokens_details"):
            d = u["prompt_tokens_details"]
            ptd = PromptTokensDetails(
                cached_tokens=d.get("cached_tokens", 0),
                audio_tokens=d.get("audio_tokens", 0),
            )

        ctd = None
        if u.get("completion_tokens_details"):
            d = u["completion_tokens_details"]
            ctd = CompletionTokensDetails(
                reasoning_tokens=d.get("reasoning_tokens", 0),
                audio_tokens=d.get("audio_tokens", 0),
                accepted_prediction_tokens=d.get(
                    "accepted_prediction_tokens", 0
                ),
                rejected_prediction_tokens=d.get(
                    "rejected_prediction_tokens", 0
                ),
            )

        return Usage(
            prompt_tokens=u.get("prompt_tokens", 0),
            completion_tokens=u.get("completion_tokens", 0),
            total_tokens=u.get("total_tokens", 0),
            prompt_tokens_details=ptd,
            completion_tokens_details=ctd,
        )

    # ── Embeddings ───────────────────────────────────────────────────

    def build_embedding_url(
        self, model: str, base_url: Optional[str] = None
    ) -> str:
        """Build URL for the embeddings endpoint."""
        base = (base_url or self.base_url).rstrip("/")
        return f"{base}/embeddings"

    def build_embedding_body(
        self, model: str, input: list | str, **kwargs: Any
    ) -> dict:
        """Build request body for the embeddings endpoint."""
        body: dict[str, Any] = {"model": model, "input": input}
        body.update(self.filter_params(kwargs))
        return body

    def parse_embedding_response(
        self, data: dict, model: str = ""
    ) -> EmbeddingResponse:
        """Parse an embeddings response into an ``EmbeddingResponse``."""
        embeddings = []
        for item in data.get("data", []):
            embeddings.append(
                EmbeddingData(
                    embedding=item.get("embedding", []),
                    index=item.get("index", 0),
                )
            )
        usage = None
        if "usage" in data:
            u = data["usage"]
            usage = Usage(
                prompt_tokens=u.get("prompt_tokens", 0),
                total_tokens=u.get("total_tokens", 0),
            )
        return EmbeddingResponse(
            model=data.get("model", model),
            data=embeddings,
            usage=usage,
        )

    # ── Provider identification ──────────────────────────────────────

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
