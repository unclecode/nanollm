"""Response types for NanoLLM.

Dataclasses with dual attribute/dict access to maintain compatibility
with litellm's response objects. Streaming chunks use plain dicts.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


class _DictAccessMixin:
    """Allows dict-style access on dataclass instances."""

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


class _AttrDict(dict):
    """A dict that also supports attribute access and __dict__.

    litellm returns token detail objects that crawl4ai accesses via
    .__dict__. This class wraps a plain dict so that both
    obj.__dict__ and obj.key work as expected.
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


@dataclass
class Message(_DictAccessMixin):
    content: str | None = None
    role: str = "assistant"
    tool_calls: list[dict] | None = None
    function_call: dict | None = None


@dataclass
class Choice(_DictAccessMixin):
    message: Message = field(default_factory=Message)
    index: int = 0
    finish_reason: str | None = None


@dataclass
class Usage(_DictAccessMixin):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    completion_tokens_details: Any = None
    prompt_tokens_details: Any = None


@dataclass
class ModelResponse(_DictAccessMixin):
    id: str = field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    choices: list[Choice] = field(default_factory=list)
    model: str = ""
    usage: Usage = field(default_factory=Usage)
    created: int = field(default_factory=lambda: int(time.time()))
    object: str = "chat.completion"

    _hidden_params: dict = field(default_factory=dict, repr=False)


@dataclass
class EmbeddingData(_DictAccessMixin):
    embedding: list[float] = field(default_factory=list)
    index: int = 0
    object: str = "embedding"


@dataclass
class EmbeddingResponse(_DictAccessMixin):
    data: list[EmbeddingData] = field(default_factory=list)
    model: str = ""
    usage: Usage = field(default_factory=Usage)
    object: str = "list"


def make_model_response(
    content: str,
    model: str = "",
    finish_reason: str = "stop",
    usage: dict | None = None,
    provider: str | None = None,
) -> ModelResponse:
    """Helper to build a ModelResponse from a completion result."""
    u = Usage(**(usage or {}))
    return ModelResponse(
        choices=[
            Choice(
                message=Message(content=content, role="assistant"),
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
    """Helper to build an EmbeddingResponse."""
    data = [
        EmbeddingData(embedding=emb, index=i)
        for i, emb in enumerate(embeddings)
    ]
    return EmbeddingResponse(
        data=data,
        model=model,
        usage=Usage(**(usage or {})),
    )


@dataclass
class Delta(_DictAccessMixin):
    content: str | None = None
    role: str | None = None
    tool_calls: list[dict] | None = None


@dataclass
class StreamChoice(_DictAccessMixin):
    delta: Delta = field(default_factory=Delta)
    index: int = 0
    finish_reason: str | None = None


@dataclass
class StreamChunk(_DictAccessMixin):
    id: str = field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    choices: list[StreamChoice] = field(default_factory=list)
    model: str = ""
    created: int = field(default_factory=lambda: int(time.time()))
    object: str = "chat.completion.chunk"


def make_stream_chunk(
    content: str | None = None,
    role: str | None = None,
    finish_reason: str | None = None,
    model: str = "",
) -> StreamChunk:
    """Build a streaming chunk with dual dict/attribute access."""
    delta = Delta(content=content, role=role)
    return StreamChunk(
        choices=[
            StreamChoice(
                delta=delta,
                index=0,
                finish_reason=finish_reason,
            )
        ],
        model=model,
    )
