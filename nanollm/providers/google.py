"""Google AI providers: Gemini (AI Studio) and Vertex AI.

Both share the same message format (Gemini API) but differ in auth and URLs.
Gemini uses an API key in the URL; Vertex uses OAuth2 bearer tokens.

Key features:
- Convert messages: roles (assistant -> model), content blocks -> parts
- Image blocks -> to_gemini_image()
- System messages -> system_instruction
- Params -> generationConfig (max_tokens -> maxOutputTokens, etc.)
- Thinking: reasoning_effort -> thinkingConfig
- Structured: json_schema -> responseMimeType + responseSchema
- URL: {base_url}/models/{model}:generateContent?key={key}
- Parse: candidates[0].content.parts[0].text
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from . import register
from .base import BaseProvider
from .._image import extract_image_url, to_gemini_image
from .._types import (
    ModelResponse, Choice, Message, Usage, ToolCall, FunctionCall,
    Delta, StreamChoice, PromptTokensDetails, CompletionTokensDetails,
)


@register("gemini")
class GeminiProvider(BaseProvider):
    """Google AI Studio (Gemini) provider."""

    name = "gemini"
    base_url = "https://generativelanguage.googleapis.com/v1beta"
    api_key_env = "GEMINI_API_KEY"

    supported_params = frozenset({
        "temperature", "top_p", "top_k", "max_tokens", "max_completion_tokens",
        "stop", "stop_sequences", "tools", "tool_choice", "response_format",
        "json_schema", "reasoning_effort", "n", "safety_settings",
    })

    # -- Headers -------------------------------------------------------------

    def build_headers(self, api_key: str) -> dict[str, str]:
        return {"Content-Type": "application/json"}

    # -- URL -----------------------------------------------------------------

    def build_url(self, model: str, endpoint: str = "",
                  base_url: Optional[str] = None, stream: bool = False,
                  **kwargs: Any) -> str:
        base = (base_url or self.base_url).rstrip("/")
        api_key = self.get_api_key(kwargs.get("api_key"))
        action = "streamGenerateContent" if stream else "generateContent"
        url = f"{base}/models/{model}:{action}"
        if api_key:
            url += f"?key={api_key}"
        if stream:
            url += "&alt=sse" if "?" in url else "?alt=sse"
        return url

    # -- Body ----------------------------------------------------------------

    def build_body(self, model: str, messages: list, stream: bool = False,
                   **kwargs: Any) -> dict:
        # Pop special params
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        response_format = kwargs.pop("response_format", None)
        json_schema = kwargs.pop("json_schema", None)
        max_tokens = kwargs.pop("max_tokens", None) or kwargs.pop(
            "max_completion_tokens", None)
        temperature = kwargs.pop("temperature", None)
        top_p = kwargs.pop("top_p", None)
        top_k = kwargs.pop("top_k", None)
        stop = kwargs.pop("stop", None) or kwargs.pop("stop_sequences", None)
        tools = kwargs.pop("tools", None)
        tool_choice = kwargs.pop("tool_choice", None)
        n = kwargs.pop("n", None)
        safety_settings = kwargs.pop("safety_settings", None)
        kwargs.pop("api_key", None)

        # Build contents
        system_instruction, contents = self._convert_messages(messages)

        body: dict[str, Any] = {"contents": contents}

        if system_instruction:
            body["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        # Generation config
        gen_config: dict[str, Any] = {}
        if max_tokens is not None:
            gen_config["maxOutputTokens"] = max_tokens
        if temperature is not None:
            gen_config["temperature"] = temperature
        if top_p is not None:
            gen_config["topP"] = top_p
        if top_k is not None:
            gen_config["topK"] = top_k
        if stop:
            gen_config["stopSequences"] = stop if isinstance(stop, list) else [stop]
        if n is not None:
            gen_config["candidateCount"] = n

        # Response format / JSON schema
        if json_schema:
            gen_config["responseMimeType"] = "application/json"
            schema = json_schema.get("schema", json_schema)
            gen_config["responseSchema"] = schema
        elif response_format:
            rf_type = response_format.get("type", "") if isinstance(response_format, dict) else ""
            if rf_type == "json_object":
                gen_config["responseMimeType"] = "application/json"
            elif rf_type == "json_schema":
                gen_config["responseMimeType"] = "application/json"
                schema = response_format.get("json_schema", {}).get("schema", {})
                if schema:
                    gen_config["responseSchema"] = schema

        if gen_config:
            body["generationConfig"] = gen_config

        # Thinking
        if reasoning_effort:
            thinking_updates = self.map_thinking(reasoning_effort)
            # Merge thinkingConfig into existing generationConfig
            if "generationConfig" in thinking_updates:
                existing = body.get("generationConfig", {})
                existing.update(thinking_updates["generationConfig"])
                body["generationConfig"] = existing
            else:
                body.update(thinking_updates)

        # Safety settings
        if safety_settings:
            body["safetySettings"] = safety_settings

        # Tools
        if tools:
            body["tools"] = [{"functionDeclarations": self._convert_tools(tools)}]

        if tool_choice:
            body["toolConfig"] = self._convert_tool_choice(tool_choice)

        return body

    # -- Message conversion --------------------------------------------------

    def _convert_messages(self, messages: list) -> tuple[str, list]:
        """Convert OpenAI messages to Gemini contents format.

        Roles: assistant -> model, user -> user
        System messages -> systemInstruction
        """
        system_parts: list[str] = []
        contents: list[dict] = []

        for msg in messages:
            role = msg.get("role", "user")

            if role == "system":
                text = msg.get("content", "")
                if isinstance(text, str):
                    system_parts.append(text)
                elif isinstance(text, list):
                    for block in text:
                        if isinstance(block, str):
                            system_parts.append(block)
                        elif isinstance(block, dict) and block.get("type") == "text":
                            system_parts.append(block.get("text", ""))
                continue

            # Map roles
            gemini_role = "model" if role == "assistant" else "user"

            # Handle tool role
            if role == "tool":
                contents.append({
                    "role": "user",
                    "parts": [{
                        "functionResponse": {
                            "name": msg.get("name", msg.get("tool_call_id", "")),
                            "response": {"result": msg.get("content", "")},
                        }
                    }],
                })
                continue

            parts = self._convert_content_to_parts(msg.get("content", ""))

            # Tool calls from assistant
            if role == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    parts.append({
                        "functionCall": {"name": fn.get("name", ""), "args": args}
                    })

            # Group consecutive messages with the same role
            if contents and contents[-1]["role"] == gemini_role:
                contents[-1]["parts"].extend(parts)
            else:
                contents.append({"role": gemini_role, "parts": parts})

        system = "\n\n".join(system_parts) if system_parts else ""
        return system, contents

    def _convert_content_to_parts(self, content: str | list | None) -> list:
        """Convert message content to Gemini parts."""
        if content is None:
            return [{"text": ""}]
        if isinstance(content, str):
            return [{"text": content}]

        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append({"text": block})
            elif isinstance(block, dict):
                btype = block.get("type", "")
                if btype == "text":
                    parts.append({"text": block.get("text", "")})
                elif btype == "image_url":
                    url = extract_image_url(block)
                    if url:
                        parts.append(to_gemini_image(url))
                    else:
                        parts.append({"text": "[image]"})
                else:
                    parts.append(block)
        return parts or [{"text": ""}]

    def _convert_tools(self, tools: list) -> list:
        """Convert OpenAI tools to Gemini function declarations."""
        declarations = []
        for tool in tools:
            if tool.get("type") == "function":
                fn = tool["function"]
                decl: dict[str, Any] = {
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                }
                params = fn.get("parameters")
                if params:
                    decl["parameters"] = params
                declarations.append(decl)
        return declarations

    def _convert_tool_choice(self, choice: str | dict) -> dict:
        """Convert OpenAI tool_choice to Gemini toolConfig."""
        if isinstance(choice, str):
            mapping = {
                "auto": "AUTO",
                "none": "NONE",
                "required": "ANY",
            }
            return {"functionCallingConfig": {"mode": mapping.get(choice, "AUTO")}}
        if isinstance(choice, dict) and "function" in choice:
            return {
                "functionCallingConfig": {
                    "mode": "ANY",
                    "allowedFunctionNames": [choice["function"]["name"]],
                }
            }
        return {"functionCallingConfig": {"mode": "AUTO"}}

    # -- Thinking ------------------------------------------------------------

    def map_thinking(self, reasoning_effort: str) -> dict:
        """Gemini uses thinkingConfig with thinkingBudget."""
        budget_map = {
            "low": 1024,
            "medium": 4096,
            "high": 16384,
            "xhigh": 32768,
        }
        budget = budget_map.get(reasoning_effort, 4096)
        return {
            "generationConfig": {
                "thinkingConfig": {"thinkingBudget": budget},
            }
        }

    # -- Response parsing ----------------------------------------------------

    def parse_response(self, data: dict, model: str = "") -> ModelResponse:
        """Parse Gemini response into ModelResponse.

        Extracts text from candidates[0].content.parts, function calls,
        and thinking parts.
        """
        candidates = data.get("candidates", [{}])
        choices = []

        for i, candidate in enumerate(candidates):
            content = candidate.get("content", {})
            parts = content.get("parts", [])

            text_parts: list[str] = []
            thinking_parts: list[str] = []
            tool_calls: list[ToolCall] = []

            for part in parts:
                if "text" in part:
                    text_parts.append(part["text"])
                elif "thought" in part:
                    thinking_parts.append(part["thought"])
                elif "functionCall" in part:
                    fc = part["functionCall"]
                    tool_calls.append(ToolCall(
                        id=f"call_{i}_{len(tool_calls)}",
                        type="function",
                        function=FunctionCall(
                            name=fc.get("name", ""),
                            arguments=json.dumps(fc.get("args", {})),
                        ),
                    ))

            finish_reason_map = {
                "STOP": "stop",
                "MAX_TOKENS": "length",
                "SAFETY": "content_filter",
                "RECITATION": "content_filter",
                "OTHER": "stop",
            }
            raw_reason = candidate.get("finishReason", "STOP")
            finish_reason = finish_reason_map.get(raw_reason, "stop")

            msg = Message(
                role="assistant",
                content="\n".join(text_parts) if text_parts else None,
                tool_calls=tool_calls or None,
                reasoning_content="\n".join(thinking_parts) if thinking_parts else None,
            )
            choices.append(Choice(index=i, message=msg, finish_reason=finish_reason))

        # Usage
        usage = None
        usage_meta = data.get("usageMetadata", {})
        if usage_meta:
            ctd = None
            if usage_meta.get("thoughtsTokenCount"):
                ctd = CompletionTokensDetails(
                    reasoning_tokens=usage_meta["thoughtsTokenCount"],
                )
            usage = Usage(
                prompt_tokens=usage_meta.get("promptTokenCount", 0),
                completion_tokens=usage_meta.get("candidatesTokenCount", 0),
                total_tokens=usage_meta.get("totalTokenCount", 0),
                completion_tokens_details=ctd,
            )
            cached = usage_meta.get("cachedContentTokenCount", 0)
            if cached:
                usage.prompt_tokens_details = PromptTokensDetails(
                    cached_tokens=cached,
                )

        return ModelResponse(
            id=data.get("id"),
            model=model,
            choices=choices or [Choice()],
            usage=usage,
        )

    def parse_stream_line(self, data: dict, model: str = "") -> ModelResponse | None:
        """Parse Gemini streaming SSE chunk.

        Each chunk has the same structure as a full response
        (candidates[].content.parts[].text).
        """
        candidates = data.get("candidates", [])
        if not candidates and "usageMetadata" not in data:
            return None

        choices = []
        for i, candidate in enumerate(candidates):
            content = candidate.get("content", {})
            parts = content.get("parts", [])

            text = ""
            thinking = ""
            tool_calls = []

            for part in parts:
                if "text" in part:
                    text += part["text"]
                elif "thought" in part:
                    thinking += part["thought"]
                elif "functionCall" in part:
                    fc = part["functionCall"]
                    tool_calls.append(ToolCall(
                        id=f"call_{i}_{len(tool_calls)}",
                        type="function",
                        function=FunctionCall(
                            name=fc.get("name", ""),
                            arguments=json.dumps(fc.get("args", {})),
                        ),
                    ))

            finish_reason = None
            raw = candidate.get("finishReason")
            if raw:
                finish_map = {
                    "STOP": "stop", "MAX_TOKENS": "length",
                    "SAFETY": "content_filter", "RECITATION": "content_filter",
                }
                finish_reason = finish_map.get(raw, raw)

            delta = Delta(
                content=text or None,
                reasoning_content=thinking or None,
                tool_calls=tool_calls or None,
            )
            choices.append(StreamChoice(
                index=i, delta=delta, finish_reason=finish_reason,
            ))

        usage = None
        usage_meta = data.get("usageMetadata", {})
        if usage_meta:
            usage = Usage(
                prompt_tokens=usage_meta.get("promptTokenCount", 0),
                completion_tokens=usage_meta.get("candidatesTokenCount", 0),
                total_tokens=usage_meta.get("totalTokenCount", 0),
            )

        return ModelResponse(
            model=model,
            choices=choices or [StreamChoice(delta=Delta())],
            usage=usage,
            stream=True,
        )


# -- Vertex AI ---------------------------------------------------------------

@register("vertex_ai")
class VertexProvider(GeminiProvider):
    """Google Vertex AI provider.

    Uses the same Gemini format but different auth (OAuth2 bearer token)
    and different URL structure:
    https://{location}-aiplatform.googleapis.com/v1/projects/{project}/
    locations/{location}/publishers/google/models/{model}:generateContent
    """

    name = "vertex_ai"
    api_key_env = "VERTEX_API_KEY"

    _project_env = "VERTEX_PROJECT"
    _location_env = "VERTEX_LOCATION"

    def _get_project(self) -> str:
        return os.environ.get(self._project_env, "")

    def _get_location(self) -> str:
        return os.environ.get(self._location_env, "us-central1")

    @property
    def base_url(self) -> str:  # type: ignore[override]
        location = self._get_location()
        return f"https://{location}-aiplatform.googleapis.com/v1"

    def build_headers(self, api_key: str) -> dict[str, str]:
        """Vertex uses bearer token auth (google.auth if available, else VERTEX_API_KEY)."""
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def build_url(self, model: str, endpoint: str = "",
                  base_url: Optional[str] = None, stream: bool = False,
                  **kwargs: Any) -> str:
        project = self._get_project()
        location = self._get_location()

        if base_url:
            base = base_url.rstrip("/")
        else:
            base = f"https://{location}-aiplatform.googleapis.com/v1"

        action = "streamGenerateContent" if stream else "generateContent"
        url = (f"{base}/projects/{project}/locations/{location}"
               f"/publishers/google/models/{model}:{action}")
        if stream:
            url += "?alt=sse"
        return url
