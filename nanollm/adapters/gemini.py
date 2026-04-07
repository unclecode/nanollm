"""Google Gemini adapter for NanoLLM.

Translates OpenAI-format messages to Google's generateContent API
and normalizes responses back to ModelResponse.
"""

from __future__ import annotations

import json
from typing import Any

from .._config import ProviderConfig
from .._types import (
    Choice,
    Message,
    ModelResponse,
    Usage,
    make_stream_chunk,
)
from ._base import BaseAdapter


def _convert_messages(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Convert OpenAI messages to Gemini contents format.

    Returns:
        (system_instruction, contents)
    """
    system_parts: list[str] = []
    contents: list[dict] = []

    for msg in messages:
        role = msg.get("role", "user")

        if role == "system":
            content = msg.get("content", "")
            if isinstance(content, str):
                system_parts.append(content)
            continue

        # Gemini uses "model" instead of "assistant"
        gemini_role = "model" if role == "assistant" else "user"

        content = msg.get("content", "")
        parts = _convert_content(content)

        contents.append({
            "role": gemini_role,
            "parts": parts,
        })

    system = "\n\n".join(system_parts) if system_parts else None
    return system, contents


def _convert_content(content: Any) -> list[dict]:
    """Convert content to Gemini parts format."""
    if isinstance(content, str):
        return [{"text": content}]

    if isinstance(content, list):
        parts: list[dict] = []
        for part in content:
            if isinstance(part, str):
                parts.append({"text": part})
            elif isinstance(part, dict):
                part_type = part.get("type", "")
                if part_type == "text":
                    parts.append({"text": part.get("text", "")})
                elif part_type == "image_url":
                    image_url = part.get("image_url", {})
                    url = image_url.get("url", "") if isinstance(image_url, dict) else str(image_url)
                    if url.startswith("data:"):
                        media_type, _, data = url.partition(";base64,")
                        media_type = media_type.replace("data:", "")
                        parts.append({
                            "inline_data": {
                                "mime_type": media_type,
                                "data": data,
                            },
                        })
                    else:
                        parts.append({
                            "file_data": {
                                "file_uri": url,
                                "mime_type": "image/jpeg",
                            },
                        })
        return parts

    return [{"text": str(content)}]


class Adapter(BaseAdapter):
    """Adapter for Google Gemini's generateContent API."""

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
        base = base_url or (provider_config.base_url if provider_config else "https://generativelanguage.googleapis.com/v1beta")

        if stream:
            url = f"{base.rstrip('/')}/models/{model}:streamGenerateContent?alt=sse&key={api_key or ''}"
        else:
            url = f"{base.rstrip('/')}/models/{model}:generateContent?key={api_key or ''}"

        headers = {"Content-Type": "application/json"}

        system_instruction, contents = _convert_messages(messages)

        body: dict[str, Any] = {
            "contents": contents,
        }

        if system_instruction:
            body["systemInstruction"] = {
                "parts": [{"text": system_instruction}],
            }

        # Build generationConfig
        gen_config: dict[str, Any] = {}
        if "temperature" in kwargs and kwargs["temperature"] is not None:
            gen_config["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs and kwargs["top_p"] is not None:
            gen_config["topP"] = kwargs["top_p"]
        if "top_k" in kwargs and kwargs["top_k"] is not None:
            gen_config["topK"] = kwargs["top_k"]
        max_tokens = kwargs.get("max_tokens") or kwargs.get("max_output_tokens")
        if max_tokens is not None:
            gen_config["maxOutputTokens"] = max_tokens
        if "stop" in kwargs and kwargs["stop"] is not None:
            stop = kwargs["stop"]
            gen_config["stopSequences"] = stop if isinstance(stop, list) else [stop]

        # Handle response_format for JSON mode
        if "response_format" in kwargs:
            rf = kwargs["response_format"]
            if isinstance(rf, dict) and rf.get("type") == "json_object":
                gen_config["responseMimeType"] = "application/json"

        if gen_config:
            body["generationConfig"] = gen_config

        if "safety_settings" in kwargs:
            body["safetySettings"] = kwargs["safety_settings"]

        return url, headers, body

    def parse_response(self, data: dict, model: str = "") -> ModelResponse:
        candidates = data.get("candidates", [])

        content = ""
        finish_reason = "stop"
        if candidates:
            candidate = candidates[0]
            parts = candidate.get("content", {}).get("parts", [])
            text_parts = [p.get("text", "") for p in parts if "text" in p]
            content = "".join(text_parts)

            reason = candidate.get("finishReason", "STOP")
            finish_reason = "stop" if reason in ("STOP", "MAX_TOKENS") else reason.lower()

        usage_data = data.get("usageMetadata", {})
        usage = Usage(
            prompt_tokens=usage_data.get("promptTokenCount", 0),
            completion_tokens=usage_data.get("candidatesTokenCount", 0),
            total_tokens=usage_data.get("totalTokenCount", 0),
        )

        return ModelResponse(
            choices=[
                Choice(
                    message=Message(content=content, role="assistant"),
                    index=0,
                    finish_reason=finish_reason,
                )
            ],
            model=model,
            usage=usage,
            _hidden_params={"custom_llm_provider": "gemini"},
        )

    def parse_stream_chunk(self, line: str, model: str = "") -> dict | None:
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return None

        candidates = data.get("candidates", [])
        if not candidates:
            return None

        candidate = candidates[0]
        parts = candidate.get("content", {}).get("parts", [])
        text_parts = [p.get("text", "") for p in parts if "text" in p]
        text = "".join(text_parts)

        finish_reason_raw = candidate.get("finishReason")
        finish_reason = None
        if finish_reason_raw and finish_reason_raw != "STOP":
            finish_reason = finish_reason_raw.lower()
        elif finish_reason_raw == "STOP":
            finish_reason = "stop"

        return make_stream_chunk(
            content=text if text else None,
            finish_reason=finish_reason,
            model=model,
        )
