"""Anthropic (Claude) provider.

Handles the full Anthropic Messages API including:
- System message extraction (Anthropic takes system as a top-level param)
- Content block conversion: text blocks + image_url -> to_anthropic_image()
- max_tokens default 4096
- Thinking: reasoning_effort -> thinking param with budget_tokens
- Structured output: json_schema -> Anthropic's tool-based schema extraction
- tool_choice mapping: "auto", "required" -> {"type": "any"}, specific -> {"type": "tool"}
- Parse response: join text blocks, extract tool_use blocks, extract thinking -> reasoning_content
- Parse stream: Anthropic SSE events (content_block_delta, message_delta, etc.)
- stop -> stop_sequences mapping
"""

from __future__ import annotations

import json
from typing import Any, Optional

from . import register
from .base import BaseProvider
from .._image import extract_image_url, to_anthropic_image
from .._types import (
    ModelResponse, Choice, Message, Usage, ToolCall, FunctionCall,
    Delta, StreamChoice, PromptTokensDetails, CompletionTokensDetails,
)


@register("anthropic")
class AnthropicProvider(BaseProvider):
    """Anthropic Messages API provider."""

    name = "anthropic"
    base_url = "https://api.anthropic.com/v1"
    api_key_env = "ANTHROPIC_API_KEY"

    _API_VERSION = "2023-06-01"
    _MAX_TOKENS_DEFAULT = 4096

    supported_params = frozenset({
        "temperature", "top_p", "top_k", "max_tokens", "max_completion_tokens",
        "stop_sequences", "stop", "tools", "tool_choice", "metadata",
        "reasoning_effort", "response_format", "json_schema",
    })

    # -- Headers -------------------------------------------------------------

    def build_headers(self, api_key: str) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": self._API_VERSION,
        }

    # -- URL -----------------------------------------------------------------

    def build_url(self, model: str, endpoint: str = "messages",
                  base_url: Optional[str] = None, **kwargs: Any) -> str:
        base = (base_url or self.base_url).rstrip("/")
        return f"{base}/{endpoint}"

    # -- Body ----------------------------------------------------------------

    def build_body(self, model: str, messages: list, stream: bool = False,
                   **kwargs: Any) -> dict:
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        response_format = kwargs.pop("response_format", None)
        json_schema = kwargs.pop("json_schema", None)
        max_tokens = kwargs.pop("max_tokens", None) or kwargs.pop(
            "max_completion_tokens", None) or self._MAX_TOKENS_DEFAULT

        # Extract system + transform messages
        system_content, transformed = self._extract_system(messages)

        body: dict[str, Any] = {
            "model": model,
            "messages": transformed,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        if system_content:
            body["system"] = system_content

        # Map remaining params
        mapped = self.map_params(**kwargs)
        filtered = self.filter_params(mapped)
        body.update(filtered)

        # Handle response_format for JSON mode
        if response_format and isinstance(response_format, dict):
            if response_format.get("type") == "json_object":
                json_instruction = "You must respond with valid JSON only. No other text."
                if body.get("system"):
                    body["system"] = f"{body['system']}\n\n{json_instruction}"
                else:
                    body["system"] = json_instruction

        # Thinking / reasoning
        if reasoning_effort:
            body.update(self.map_thinking(reasoning_effort))

        # Structured output via json_schema -> tool-based extraction
        if json_schema:
            body_updates, header_updates = self._map_json_schema(json_schema)
            body.update(body_updates)

        return body

    # -- Message transformation ----------------------------------------------

    def _extract_system(self, messages: list) -> tuple[str | list, list]:
        """Extract system messages and convert the rest to Anthropic format."""
        system_parts: list[str] = []
        converted: list[dict] = []

        for msg in messages:
            role = msg.get("role", "user")

            if role == "system":
                content = msg.get("content", "")
                if isinstance(content, str):
                    system_parts.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, str):
                            system_parts.append(block)
                        elif isinstance(block, dict) and block.get("type") == "text":
                            system_parts.append(block.get("text", ""))
                continue

            # Map 'tool' role to Anthropic's 'user' with tool_result
            if role == "tool":
                tool_call_id = msg.get("tool_call_id", "")
                content = msg.get("content", "")
                converted.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_call_id,
                        "content": content if isinstance(content, str) else json.dumps(content),
                    }],
                })
                continue

            # Convert content to Anthropic blocks
            content = msg.get("content", "")
            anthropic_content = self._convert_content(content)

            # Handle tool_calls in assistant messages
            if role == "assistant" and msg.get("tool_calls"):
                blocks: list[dict] = []
                if content:
                    text = content if isinstance(content, str) else str(content)
                    blocks.append({"type": "text", "text": text})
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    args_str = fn.get("arguments", "{}")
                    try:
                        args = json.loads(args_str)
                    except (json.JSONDecodeError, TypeError):
                        args = {"raw": args_str}
                    blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": fn.get("name", ""),
                        "input": args,
                    })
                converted.append({"role": "assistant", "content": blocks})
                continue

            converted.append({
                "role": role,
                "content": anthropic_content,
            })

        system = "\n\n".join(system_parts) if system_parts else ""
        return system, converted

    def _convert_content(self, content: str | list | None) -> str | list:
        """Convert message content to Anthropic format."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content

        # List of content blocks
        blocks = []
        for block in content:
            if isinstance(block, str):
                blocks.append({"type": "text", "text": block})
            elif isinstance(block, dict):
                btype = block.get("type", "")
                if btype == "text":
                    blocks.append({"type": "text", "text": block.get("text", "")})
                elif btype == "image_url":
                    url = extract_image_url(block)
                    if url:
                        blocks.append(to_anthropic_image(url))
                    else:
                        blocks.append({"type": "text", "text": "[image]"})
                else:
                    blocks.append(block)
        return blocks

    # -- Parameter mapping ---------------------------------------------------

    def map_params(self, **kwargs: Any) -> dict:
        result = {}
        for k, v in kwargs.items():
            if k == "stop":
                result["stop_sequences"] = v if isinstance(v, list) else [v]
            elif k == "tools":
                result["tools"] = self._convert_tools(v)
            elif k == "tool_choice":
                result["tool_choice"] = self._convert_tool_choice(v)
            elif k in ("temperature", "top_p", "top_k", "metadata"):
                result[k] = v
            # Skip params Anthropic doesn't support
        return result

    def _convert_tools(self, tools: list) -> list:
        """Convert OpenAI tools format to Anthropic format."""
        converted = []
        for tool in tools:
            if tool.get("type") == "function":
                fn = tool["function"]
                converted.append({
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                })
            else:
                converted.append(tool)
        return converted

    def _convert_tool_choice(self, choice: str | dict) -> dict:
        """Convert OpenAI tool_choice to Anthropic format.

        "auto" -> {"type": "auto"}
        "none" -> remove tools
        "required" -> {"type": "any"}
        {"type": "function", "function": {"name": X}} -> {"type": "tool", "name": X}
        """
        if isinstance(choice, str):
            mapping = {
                "auto": {"type": "auto"},
                "none": {"type": "none"},
                "required": {"type": "any"},
            }
            return mapping.get(choice, {"type": "auto"})
        if isinstance(choice, dict) and choice.get("type") == "function":
            return {"type": "tool", "name": choice["function"]["name"]}
        if isinstance(choice, dict) and "function" in choice:
            return {"type": "tool", "name": choice["function"]["name"]}
        return {"type": "auto"}

    # -- Thinking / reasoning ------------------------------------------------

    def map_thinking(self, reasoning_effort: str) -> dict:
        """Map reasoning_effort to Anthropic's thinking config.

        Anthropic uses budget_tokens inside a thinking block.
        low=1024, medium=2048, high=4096.
        """
        budget_map = {
            "low": 1024,
            "medium": 2048,
            "high": 4096,
            "xhigh": 16384,
        }
        budget = budget_map.get(reasoning_effort, 4096)
        return {
            "thinking": {
                "type": "enabled",
                "budget_tokens": budget,
            }
        }

    # -- Structured output ---------------------------------------------------

    def _map_json_schema(self, schema: dict) -> tuple[dict, dict]:
        """Map json_schema to Anthropic tool_use pattern for structured output.

        Anthropic uses a forced tool call to extract structured JSON.
        """
        tool_name = schema.get("name", "structured_output")
        body_updates = {
            "tools": [{
                "name": tool_name,
                "description": schema.get("description",
                                          "Generate structured output"),
                "input_schema": schema.get("schema", schema),
            }],
            "tool_choice": {"type": "tool", "name": tool_name},
        }
        return body_updates, {}

    # -- Response parsing ----------------------------------------------------

    def parse_response(self, data: dict, model: str = "") -> ModelResponse:
        """Parse Anthropic response into a ModelResponse.

        Joins text blocks, extracts tool_use blocks, extracts thinking blocks
        into reasoning_content.
        """
        content_text = ""
        reasoning_text = ""
        tool_calls = []

        for block in data.get("content", []):
            btype = block.get("type", "")
            if btype == "text":
                content_text += block.get("text", "")
            elif btype == "thinking":
                reasoning_text += block.get("thinking", "")
            elif btype == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.get("id", ""),
                    type="function",
                    function=FunctionCall(
                        name=block.get("name", ""),
                        arguments=json.dumps(block.get("input", {})),
                    ),
                ))

        # Map Anthropic stop reasons to OpenAI finish reasons
        stop_reason = data.get("stop_reason", "")
        finish_reason_map = {
            "end_turn": "stop",
            "stop_sequence": "stop",
            "max_tokens": "length",
            "tool_use": "tool_calls",
        }
        finish_reason = finish_reason_map.get(stop_reason, stop_reason or "stop")

        msg = Message(
            role="assistant",
            content=content_text or None,
            tool_calls=tool_calls or None,
            reasoning_content=reasoning_text or None,
        )

        # Parse usage
        usage = None
        if "usage" in data:
            u = data["usage"]
            cache_read = u.get("cache_read_input_tokens", 0)
            ptd = PromptTokensDetails(cached_tokens=cache_read) if cache_read else None
            usage = Usage(
                prompt_tokens=u.get("input_tokens", 0),
                completion_tokens=u.get("output_tokens", 0),
                prompt_tokens_details=ptd,
            )

        return ModelResponse(
            id=data.get("id"),
            model=data.get("model", model),
            choices=[Choice(index=0, message=msg, finish_reason=finish_reason)],
            usage=usage,
        )

    # -- Streaming -----------------------------------------------------------

    def parse_stream_line(self, data: dict, model: str = "") -> ModelResponse | None:
        """Parse Anthropic SSE event into a streaming ModelResponse.

        Handles event types: message_start, content_block_start,
        content_block_delta, content_block_stop, message_delta, message_stop.
        """
        event_type = data.get("type", "")

        if event_type == "message_start":
            msg = data.get("message", {})
            usage_data = msg.get("usage", {})
            usage = Usage(
                prompt_tokens=usage_data.get("input_tokens", 0),
            ) if usage_data else None
            return ModelResponse(
                id=msg.get("id"),
                model=msg.get("model", model),
                choices=[StreamChoice(delta=Delta(role="assistant"))],
                usage=usage,
                stream=True,
            )

        if event_type == "content_block_delta":
            delta_data = data.get("delta", {})
            delta_type = delta_data.get("type", "")

            if delta_type == "text_delta":
                return ModelResponse(
                    model=model,
                    choices=[StreamChoice(delta=Delta(
                        content=delta_data.get("text", ""),
                    ))],
                    stream=True,
                )
            elif delta_type == "thinking_delta":
                return ModelResponse(
                    model=model,
                    choices=[StreamChoice(delta=Delta(
                        reasoning_content=delta_data.get("thinking", ""),
                    ))],
                    stream=True,
                )
            elif delta_type == "input_json_delta":
                # Tool call argument streaming
                partial_json = delta_data.get("partial_json", "")
                if partial_json:
                    return ModelResponse(
                        model=model,
                        choices=[StreamChoice(delta=Delta(
                            tool_calls=[ToolCall(
                                function=FunctionCall(arguments=partial_json),
                            )],
                        ))],
                        stream=True,
                    )

        if event_type == "content_block_start":
            block = data.get("content_block", {})
            if block.get("type") == "tool_use":
                return ModelResponse(
                    model=model,
                    choices=[StreamChoice(delta=Delta(
                        tool_calls=[ToolCall(
                            id=block.get("id", ""),
                            type="function",
                            function=FunctionCall(
                                name=block.get("name", ""),
                            ),
                        )],
                    ))],
                    stream=True,
                )

        if event_type == "message_delta":
            delta_data = data.get("delta", {})
            stop_reason = delta_data.get("stop_reason", "")
            finish_map = {
                "end_turn": "stop", "stop_sequence": "stop",
                "max_tokens": "length", "tool_use": "tool_calls",
            }
            finish = finish_map.get(stop_reason, stop_reason)
            usage_data = data.get("usage", {})
            usage = Usage(
                completion_tokens=usage_data.get("output_tokens", 0),
            ) if usage_data else None
            return ModelResponse(
                model=model,
                choices=[StreamChoice(
                    delta=Delta(),
                    finish_reason=finish,
                )],
                usage=usage,
                stream=True,
            )

        return None
