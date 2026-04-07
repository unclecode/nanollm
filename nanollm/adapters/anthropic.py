"""Anthropic Messages API adapter for NanoLLM.

Translates OpenAI-format messages to Anthropic's Messages API format
and normalizes responses back to the OpenAI-compatible ModelResponse.
"""

from __future__ import annotations

import json
from typing import Any

from .._config import ProviderConfig
from .._types import (
    Choice,
    EmbeddingResponse,
    Message,
    ModelResponse,
    Usage,
    make_stream_chunk,
)
from ._base import BaseAdapter

# Anthropic API version
_API_VERSION = "2023-06-01"


def _convert_messages(
    messages: list[dict],
) -> tuple[str | None, list[dict]]:
    """Convert OpenAI-format messages to Anthropic format.

    Extracts system messages into a separate system parameter.
    Converts content blocks for multimodal (image) support.

    Returns:
        (system_prompt, anthropic_messages)
    """
    system_parts: list[str] = []
    anthropic_messages: list[dict] = []

    for msg in messages:
        role = msg.get("role", "user")

        if role == "system":
            # Anthropic takes system as a top-level param
            content = msg.get("content", "")
            if isinstance(content, str):
                system_parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, str):
                        system_parts.append(part)
                    elif isinstance(part, dict) and part.get("type") == "text":
                        system_parts.append(part.get("text", ""))
            continue

        # Map "function" role to "user" for compat
        if role == "function":
            role = "user"

        content = msg.get("content", "")
        anthropic_content = _convert_content(content)

        anthropic_messages.append({
            "role": role,
            "content": anthropic_content,
        })

    system = "\n\n".join(system_parts) if system_parts else None
    return system, anthropic_messages


def _convert_content(content: Any) -> str | list[dict]:
    """Convert content to Anthropic format.

    Handles:
    - Plain strings
    - OpenAI content blocks (text + image_url)
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        blocks: list[dict] = []
        for part in content:
            if isinstance(part, str):
                blocks.append({"type": "text", "text": part})
            elif isinstance(part, dict):
                part_type = part.get("type", "")
                if part_type == "text":
                    blocks.append({"type": "text", "text": part.get("text", "")})
                elif part_type == "image_url":
                    image_url = part.get("image_url", {})
                    url = image_url.get("url", "") if isinstance(image_url, dict) else str(image_url)
                    if url.startswith("data:"):
                        # Base64 encoded image
                        media_type, _, data = url.partition(";base64,")
                        media_type = media_type.replace("data:", "")
                        blocks.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": data,
                            },
                        })
                    else:
                        # URL-based image
                        blocks.append({
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": url,
                            },
                        })
        return blocks

    return str(content)


class Adapter(BaseAdapter):
    """Adapter for Anthropic's Messages API."""

    def build_request(
        self,
        model: str,
        messages: list[dict],
        api_key: str | None = None,
        base_url: str | None = None,
        stream: bool = False,
        provider_config: ProviderConfig | None = None,
        **kwargs: Any,
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        base = base_url or (provider_config.base_url if provider_config else "https://api.anthropic.com")
        url = f"{base.rstrip('/')}/v1/messages"

        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "anthropic-version": _API_VERSION,
        }
        if api_key:
            headers["x-api-key"] = api_key

        system, anthropic_messages = _convert_messages(messages)

        body: dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": kwargs.pop("max_tokens", 4096),
        }

        if system:
            body["system"] = system
        if stream:
            body["stream"] = True

        # Map OpenAI params to Anthropic params
        for key in ("temperature", "top_p", "top_k", "stop", "tools", "tool_choice", "metadata"):
            if key in kwargs and kwargs[key] is not None:
                body[key] = kwargs[key]

        # Handle response_format for JSON mode
        if "response_format" in kwargs:
            rf = kwargs["response_format"]
            if isinstance(rf, dict) and rf.get("type") == "json_object":
                # Anthropic doesn't have native JSON mode — add instruction
                if system:
                    body["system"] = system + "\n\nRespond with valid JSON only."
                else:
                    body["system"] = "Respond with valid JSON only."

        return url, headers, body

    def parse_response(self, data: dict, model: str = "") -> ModelResponse:
        # Extract text from Anthropic's content blocks
        content_blocks = data.get("content", [])
        text_parts: list[str] = []
        for block in content_blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))

        content = "".join(text_parts)

        usage_data = data.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("input_tokens", 0),
            completion_tokens=usage_data.get("output_tokens", 0),
            total_tokens=usage_data.get("input_tokens", 0) + usage_data.get("output_tokens", 0),
        )

        stop_reason = data.get("stop_reason", "stop")
        finish_reason = "stop" if stop_reason == "end_turn" else stop_reason

        return ModelResponse(
            id=data.get("id", ""),
            choices=[
                Choice(
                    message=Message(content=content, role="assistant"),
                    index=0,
                    finish_reason=finish_reason,
                )
            ],
            model=data.get("model", model),
            usage=usage,
            _hidden_params={"custom_llm_provider": "anthropic"},
        )

    def parse_stream_chunk(self, line: str, model: str = "") -> dict | None:
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return None

        event_type = data.get("type", "")

        if event_type == "content_block_delta":
            delta = data.get("delta", {})
            if delta.get("type") == "text_delta":
                return make_stream_chunk(
                    content=delta.get("text", ""),
                    model=model,
                )

        elif event_type == "message_start":
            return make_stream_chunk(role="assistant", model=model)

        elif event_type == "message_delta":
            stop_reason = data.get("delta", {}).get("stop_reason")
            if stop_reason:
                finish = "stop" if stop_reason == "end_turn" else stop_reason
                return make_stream_chunk(finish_reason=finish, model=model)

        elif event_type == "message_stop":
            return make_stream_chunk(finish_reason="stop", model=model)

        return None
