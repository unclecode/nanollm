"""Response types for NanoLLM.

All classes use ``__slots__`` for memory efficiency and support **both**
attribute access (``response.choices[0].message.content``) and dict-style
access (``response["choices"][0]["message"]["content"]``) so downstream
code that treats them as dicts keeps working.

Token-detail objects use the ``_AttrDict`` wrapper so that the
``obj.__dict__`` access pattern used by crawl4ai works correctly.
"""

from __future__ import annotations

import time
import uuid as _uuid
from typing import Any, Optional


def _generate_id(prefix: str = "chatcmpl-") -> str:
    return prefix + _uuid.uuid4().hex[:12]


# ── Dict-like access mixin ───────────────────────────────────────────


class _DictAccessMixin:
    """Provide ``__getitem__``, ``get()``, ``keys()``, ``items()``,
    ``__contains__``, ``to_dict()``, ``model_dump()``, and ``json()``
    on any ``__slots__``-based class.
    """

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def keys(self) -> list[str]:
        return [s for s in self.__slots__ if not s.startswith("_")]

    def items(self) -> list[tuple[str, Any]]:
        return [(k, getattr(self, k, None)) for k in self.keys()]

    def to_dict(self) -> dict:
        """Recursively convert to a plain dict, omitting ``None`` values."""
        result: dict[str, Any] = {}
        for k in self.keys():
            v = getattr(self, k, None)
            if v is None:
                continue
            if isinstance(v, _DictAccessMixin):
                result[k] = v.to_dict()
            elif isinstance(v, list):
                result[k] = [
                    x.to_dict() if isinstance(x, _DictAccessMixin) else x
                    for x in v
                ]
            else:
                result[k] = v
        return result

    def model_dump(self) -> dict:
        """Alias for ``to_dict()``, matching pydantic convention."""
        return self.to_dict()

    def json(self) -> str:
        """Serialize to a JSON string."""
        import json as _json

        return _json.dumps(self.to_dict())

    def __repr__(self) -> str:
        fields = ", ".join(
            f"{k}={getattr(self, k, None)!r}"
            for k in self.keys()
            if getattr(self, k, None) is not None
        )
        return f"{type(self).__name__}({fields})"


# ── _AttrDict for token detail objects ───────────────────────────────


class _AttrDict(dict):
    """A dict that also supports attribute access and ``__dict__``.

    litellm returns token-detail objects that crawl4ai accesses via
    ``.__dict__``.  This class wraps a plain dict so that both
    ``obj.__dict__`` and ``obj.key`` work as expected.
    """

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    @property
    def __dict__(self) -> dict:  # type: ignore[override]
        return dict(self)


# ── Token usage detail types ─────────────────────────────────────────


class PromptTokensDetails(_DictAccessMixin):
    """Breakdown of prompt token usage."""

    __slots__ = ("cached_tokens", "audio_tokens")

    def __init__(self, cached_tokens: int = 0, audio_tokens: int = 0):
        self.cached_tokens = cached_tokens
        self.audio_tokens = audio_tokens


class CompletionTokensDetails(_DictAccessMixin):
    """Breakdown of completion token usage."""

    __slots__ = (
        "reasoning_tokens",
        "audio_tokens",
        "accepted_prediction_tokens",
        "rejected_prediction_tokens",
    )

    def __init__(
        self,
        reasoning_tokens: int = 0,
        audio_tokens: int = 0,
        accepted_prediction_tokens: int = 0,
        rejected_prediction_tokens: int = 0,
    ):
        self.reasoning_tokens = reasoning_tokens
        self.audio_tokens = audio_tokens
        self.accepted_prediction_tokens = accepted_prediction_tokens
        self.rejected_prediction_tokens = rejected_prediction_tokens


class Usage(_DictAccessMixin):
    """Token usage summary."""

    __slots__ = (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_tokens_details",
        "completion_tokens_details",
    )

    def __init__(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        prompt_tokens_details: Optional[PromptTokensDetails] = None,
        completion_tokens_details: Optional[CompletionTokensDetails] = None,
        **kwargs: Any,
    ):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens or (prompt_tokens + completion_tokens)
        # Wrap token detail dicts into _AttrDict so .__dict__ works
        self.prompt_tokens_details = (
            _AttrDict(prompt_tokens_details)
            if isinstance(prompt_tokens_details, dict)
            else prompt_tokens_details
        )
        self.completion_tokens_details = (
            _AttrDict(completion_tokens_details)
            if isinstance(completion_tokens_details, dict)
            else completion_tokens_details
        )


# ── Chat message types ───────────────────────────────────────────────


class FunctionCall(_DictAccessMixin):
    """A function call within a tool call."""

    __slots__ = ("name", "arguments")

    def __init__(self, name: str = "", arguments: str = ""):
        self.name = name
        self.arguments = arguments


class ToolCall(_DictAccessMixin):
    """A tool call returned by the model."""

    __slots__ = ("id", "type", "function")

    def __init__(
        self,
        id: str = "",
        type: str = "function",
        function: Optional[FunctionCall] = None,
    ):
        self.id = id
        self.type = type
        self.function = function or FunctionCall()


class Message(_DictAccessMixin):
    """A chat message (assistant response)."""

    __slots__ = ("role", "content", "tool_calls", "function_call", "reasoning_content")

    def __init__(
        self,
        role: str = "assistant",
        content: Optional[str] = None,
        tool_calls: Optional[list[ToolCall]] = None,
        function_call: Optional[FunctionCall] = None,
        reasoning_content: Optional[str] = None,
    ):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls
        self.function_call = function_call
        self.reasoning_content = reasoning_content


class Choice(_DictAccessMixin):
    """A single completion choice."""

    __slots__ = ("index", "message", "finish_reason")

    def __init__(
        self,
        index: int = 0,
        message: Optional[Message] = None,
        finish_reason: Optional[str] = None,
    ):
        self.index = index
        self.message = message or Message()
        self.finish_reason = finish_reason


# ── Stream delta types ───────────────────────────────────────────────


class Delta(_DictAccessMixin):
    """Incremental content in a streaming chunk."""

    __slots__ = ("role", "content", "tool_calls", "function_call", "reasoning_content")

    def __init__(
        self,
        role: Optional[str] = None,
        content: Optional[str] = None,
        tool_calls: Optional[list] = None,
        function_call: Optional[FunctionCall] = None,
        reasoning_content: Optional[str] = None,
    ):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls
        self.function_call = function_call
        self.reasoning_content = reasoning_content


class StreamChoice(_DictAccessMixin):
    """A single choice in a streaming chunk."""

    __slots__ = ("index", "delta", "finish_reason")

    def __init__(
        self,
        index: int = 0,
        delta: Optional[Delta] = None,
        finish_reason: Optional[str] = None,
    ):
        self.index = index
        self.delta = delta or Delta()
        self.finish_reason = finish_reason


# ── Main response objects ────────────────────────────────────────────


class ModelResponse(_DictAccessMixin):
    """Chat completion response, compatible with litellm.ModelResponse."""

    __slots__ = (
        "id",
        "object",
        "created",
        "model",
        "choices",
        "usage",
        "system_fingerprint",
        "stream",
        "_hidden_params",
    )

    def __init__(
        self,
        id: Optional[str] = None,
        object: str = "chat.completion",
        created: Optional[int] = None,
        model: str = "",
        choices: Optional[list] = None,
        usage: Optional[Usage] = None,
        system_fingerprint: Optional[str] = None,
        stream: bool = False,
        _hidden_params: Optional[dict] = None,
    ):
        self.id = id or _generate_id()
        self.object = object
        self.created = created or int(time.time())
        self.model = model
        self.stream = stream
        self._hidden_params = _hidden_params or {}

        if stream:
            self.choices = choices or [StreamChoice()]
            self.object = "chat.completion.chunk"
        else:
            self.choices = choices or [Choice()]
        self.usage = usage
        self.system_fingerprint = system_fingerprint


# ── Text completion types ────────────────────────────────────────────


class TextChoice(_DictAccessMixin):
    """A single choice in a text completion response."""

    __slots__ = ("text", "index", "finish_reason", "logprobs")

    def __init__(
        self,
        text: str = "",
        index: int = 0,
        finish_reason: Optional[str] = None,
        logprobs: Any = None,
    ):
        self.text = text
        self.index = index
        self.finish_reason = finish_reason
        self.logprobs = logprobs


class TextCompletionResponse(_DictAccessMixin):
    """Legacy text completion response."""

    __slots__ = ("id", "object", "created", "model", "choices", "usage")

    def __init__(
        self,
        id: Optional[str] = None,
        model: str = "",
        choices: Optional[list] = None,
        usage: Optional[Usage] = None,
    ):
        self.id = id or _generate_id(prefix="cmpl-")
        self.object = "text_completion"
        self.created = int(time.time())
        self.model = model
        self.choices = choices or []
        self.usage = usage


# ── Embedding types ──────────────────────────────────────────────────


class EmbeddingData(_DictAccessMixin):
    """A single embedding vector."""

    __slots__ = ("object", "embedding", "index")

    def __init__(
        self, embedding: Optional[list[float]] = None, index: int = 0
    ):
        self.object = "embedding"
        self.embedding = embedding or []
        self.index = index


class EmbeddingResponse(_DictAccessMixin):
    """Response from an embeddings endpoint."""

    __slots__ = ("object", "data", "model", "usage")

    def __init__(
        self,
        model: str = "",
        data: Optional[list[EmbeddingData]] = None,
        usage: Optional[Usage] = None,
    ):
        self.object = "list"
        self.data = data or []
        self.model = model
        self.usage = usage or Usage()


# ── Factory helpers ──────────────────────────────────────────────────


def make_model_response(
    content: str,
    model: str = "",
    finish_reason: str = "stop",
    usage: dict | None = None,
    provider: str | None = None,
    tool_calls: Optional[list] = None,
    reasoning_content: Optional[str] = None,
) -> ModelResponse:
    """Build a ``ModelResponse`` from basic completion results."""
    u = Usage(**(usage or {}))
    return ModelResponse(
        choices=[
            Choice(
                message=Message(
                    content=content,
                    role="assistant",
                    tool_calls=tool_calls,
                    reasoning_content=reasoning_content,
                ),
                index=0,
                finish_reason=finish_reason,
            )
        ],
        model=model,
        usage=u,
        _hidden_params={"custom_llm_provider": provider or ""},
    )


def make_embedding_response(
    embeddings: list[list[float]],
    model: str = "",
    usage: dict | None = None,
) -> EmbeddingResponse:
    """Build an ``EmbeddingResponse`` from raw embedding vectors."""
    data = [
        EmbeddingData(embedding=emb, index=i) for i, emb in enumerate(embeddings)
    ]
    return EmbeddingResponse(
        data=data,
        model=model,
        usage=Usage(**(usage or {})),
    )


def make_stream_chunk(
    content: Optional[str] = None,
    role: Optional[str] = None,
    finish_reason: Optional[str] = None,
    model: str = "",
    tool_calls: Optional[list] = None,
    reasoning_content: Optional[str] = None,
) -> ModelResponse:
    """Build a single streaming chunk as a ``ModelResponse``."""
    delta = Delta(
        role=role,
        content=content,
        tool_calls=tool_calls,
        reasoning_content=reasoning_content,
    )
    return ModelResponse(
        model=model,
        choices=[StreamChoice(index=0, delta=delta, finish_reason=finish_reason)],
        stream=True,
    )


# ── Stream chunk builder ────────────────────────────────────────────


def stream_chunk_builder(
    chunks: list[ModelResponse],
    messages: Optional[list] = None,
) -> ModelResponse:
    """Merge a list of streaming chunks into a single ``ModelResponse``.

    Accumulates content strings, reasoning content, tool calls, and usage
    across all chunks.

    Args:
        chunks: List of streaming ``ModelResponse`` objects.
        messages: Original messages (unused, kept for litellm compat).

    Returns:
        A non-streaming ``ModelResponse`` with accumulated content and usage.
    """
    if not chunks:
        return ModelResponse()

    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls_map: dict[int, ToolCall] = {}
    finish_reason: Optional[str] = None
    model = ""
    response_id = ""
    role = "assistant"
    usage: Optional[Usage] = None

    for chunk in chunks:
        if chunk.model:
            model = chunk.model
        if chunk.id:
            response_id = chunk.id

        for choice in chunk.choices or []:
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue

            if delta.role:
                role = delta.role
            if delta.content:
                content_parts.append(delta.content)
            if delta.reasoning_content:
                reasoning_parts.append(delta.reasoning_content)

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = getattr(tc, "index", 0) if hasattr(tc, "index") else 0
                    if idx not in tool_calls_map:
                        tool_calls_map[idx] = ToolCall(
                            id=getattr(tc, "id", "") or "",
                            type=getattr(tc, "type", "function") or "function",
                            function=FunctionCall(),
                        )
                    existing = tool_calls_map[idx]
                    fn = getattr(tc, "function", None)
                    if fn:
                        if getattr(fn, "name", None):
                            existing.function.name = fn.name
                        if getattr(fn, "arguments", None):
                            existing.function.arguments += fn.arguments

            if choice.finish_reason:
                finish_reason = choice.finish_reason

        if chunk.usage:
            usage = chunk.usage

    content = "".join(content_parts) or None
    reasoning = "".join(reasoning_parts) or None
    tool_calls_list = (
        [tool_calls_map[k] for k in sorted(tool_calls_map)] or None
    )

    msg = Message(
        role=role,
        content=content,
        tool_calls=tool_calls_list,
        reasoning_content=reasoning,
    )

    return ModelResponse(
        id=response_id or None,
        model=model,
        choices=[Choice(index=0, message=msg, finish_reason=finish_reason)],
        usage=usage,
        stream=False,
    )
