"""Base adapter interface for NanoLLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .._types import EmbeddingResponse, ModelResponse, StreamChunk


class BaseAdapter(ABC):
    """Abstract base class for provider adapters.

    Each adapter translates between NanoLLM's unified interface and
    a specific provider's API format.
    """

    @abstractmethod
    def build_request(
        self,
        model: str,
        messages: list[dict],
        api_key: str | None = None,
        base_url: str | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        """Build an HTTP request for a completion call.

        Returns:
            Tuple of (url, headers, body)
        """
        ...

    @abstractmethod
    def parse_response(self, data: dict, model: str = "") -> ModelResponse:
        """Parse a JSON response into a ModelResponse."""
        ...

    @abstractmethod
    def parse_stream_chunk(self, line: str, model: str = "") -> StreamChunk | None:
        """Parse an SSE data line into a StreamChunk.

        Returns None if the line should be skipped.
        """
        ...

    def build_embedding_request(
        self,
        model: str,
        input: list[str],
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        """Build an HTTP request for an embedding call.

        Returns:
            Tuple of (url, headers, body)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support embeddings"
        )

    def parse_embedding_response(
        self, data: dict, model: str = ""
    ) -> EmbeddingResponse:
        """Parse a JSON response into an EmbeddingResponse."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support embeddings"
        )

    def filter_params(
        self, kwargs: dict[str, Any], supported: frozenset[str]
    ) -> dict[str, Any]:
        """Strip unsupported parameters (implements drop_params behavior)."""
        return {k: v for k, v in kwargs.items() if k in supported}
