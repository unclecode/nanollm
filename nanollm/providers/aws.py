"""AWS Bedrock provider with inline SigV4 signing.

Uses only stdlib (hashlib, hmac, datetime) for AWS Signature Version 4.
No boto3 required, but uses boto3/botocore credential chain when available.

Supports Anthropic Claude, Amazon Nova, Meta Llama, Mistral, and other
models available through Bedrock via the Converse API.

Key features:
- Inline SigV4 signing (hashlib, hmac)
- Converse API: https://bedrock-runtime.{region}.amazonaws.com/model/{model}/converse
- Convert messages to Bedrock format (content blocks, image -> to_bedrock_image)
- System -> system parameter
- inferenceConfig for max_tokens, temperature, etc.
- Parse: output.message.content[0].text
"""

from __future__ import annotations

import datetime
import hashlib
import hmac
import json
import os
from typing import Any, Optional
from urllib.parse import urlparse, quote

from . import register
from .base import BaseProvider
from .._image import extract_image_url, to_bedrock_image
from .._types import (
    ModelResponse, Choice, Message, Usage, ToolCall, FunctionCall,
    Delta, StreamChoice,
)


# -- SigV4 Signing -----------------------------------------------------------

def _sign(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def _get_signature_key(secret_key: str, date_stamp: str, region: str,
                       service: str) -> bytes:
    k_date = _sign(("AWS4" + secret_key).encode("utf-8"), date_stamp)
    k_region = _sign(k_date, region)
    k_service = _sign(k_region, service)
    k_signing = _sign(k_service, "aws4_request")
    return k_signing


def _get_credentials() -> tuple[str, str, str]:
    """Get AWS credentials from boto3 (if available) or environment variables.

    Returns:
        (access_key, secret_key, session_token)
    """
    # Try boto3 credential chain first (handles IAM roles, profiles, etc.)
    try:
        import botocore.session

        session = botocore.session.get_session()
        creds = session.get_credentials()
        if creds:
            resolved = creds.get_frozen_credentials()
            return resolved.access_key, resolved.secret_key, resolved.token or ""
    except Exception:
        pass

    access_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    session_token = os.environ.get("AWS_SESSION_TOKEN", "")
    return access_key, secret_key, session_token


def sigv4_headers(
    method: str,
    url: str,
    body: bytes,
    region: str,
    service: str = "bedrock",
    access_key: str = "",
    secret_key: str = "",
    session_token: str = "",
) -> dict[str, str]:
    """Generate AWS SigV4 signed headers for a request.

    Returns dict of headers to merge into the request.
    """
    if not access_key or not secret_key:
        access_key_resolved, secret_key_resolved, session_token_resolved = _get_credentials()
        access_key = access_key or access_key_resolved
        secret_key = secret_key or secret_key_resolved
        session_token = session_token or session_token_resolved

    if not access_key or not secret_key:
        raise ValueError(
            "AWS credentials not found. Set AWS_ACCESS_KEY_ID and "
            "AWS_SECRET_ACCESS_KEY, or configure boto3 credentials."
        )

    now = datetime.datetime.now(datetime.timezone.utc)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = now.strftime("%Y%m%d")

    parsed = urlparse(url)
    host = parsed.hostname or ""
    canonical_uri = quote(parsed.path or "/", safe="/")
    canonical_querystring = parsed.query or ""

    payload_hash = hashlib.sha256(body).hexdigest()

    # Build canonical headers
    signed_header_names = ["content-type", "host", "x-amz-date"]
    headers_map = {
        "content-type": "application/json",
        "host": host,
        "x-amz-date": amz_date,
    }
    if session_token:
        signed_header_names.append("x-amz-security-token")
        headers_map["x-amz-security-token"] = session_token

    signed_header_names.sort()
    canonical_headers = "".join(
        f"{k}:{headers_map[k]}\n" for k in signed_header_names
    )
    signed_headers = ";".join(signed_header_names)

    canonical_request = "\n".join([
        method,
        canonical_uri,
        canonical_querystring,
        canonical_headers,
        signed_headers,
        payload_hash,
    ])

    credential_scope = f"{date_stamp}/{region}/{service}/aws4_request"
    string_to_sign = "\n".join([
        "AWS4-HMAC-SHA256",
        amz_date,
        credential_scope,
        hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
    ])

    signing_key = _get_signature_key(secret_key, date_stamp, region, service)
    signature = hmac.new(
        signing_key, string_to_sign.encode("utf-8"), hashlib.sha256
    ).hexdigest()

    authorization = (
        f"AWS4-HMAC-SHA256 Credential={access_key}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )

    result = {
        "Content-Type": "application/json",
        "x-amz-date": amz_date,
        "x-amz-content-sha256": payload_hash,
        "Authorization": authorization,
    }
    if session_token:
        result["x-amz-security-token"] = session_token

    return result


# -- Bedrock Provider --------------------------------------------------------

@register("bedrock")
class BedrockProvider(BaseProvider):
    """AWS Bedrock provider with SigV4 auth.

    Uses the Converse API for all models.
    """

    name = "bedrock"
    base_url = ""  # Built dynamically from region
    api_key_env = ""  # Uses AWS credentials

    _region_env = "AWS_REGION"
    _default_region = "us-east-1"

    supported_params = frozenset({
        "temperature", "top_p", "top_k", "max_tokens", "max_completion_tokens",
        "stop", "stop_sequences", "tools", "tool_choice", "reasoning_effort",
    })

    def _get_region(self) -> str:
        return os.environ.get(self._region_env,
               os.environ.get("AWS_DEFAULT_REGION", self._default_region))

    def _get_base_url(self) -> str:
        region = self._get_region()
        return f"https://bedrock-runtime.{region}.amazonaws.com"

    # -- Headers -------------------------------------------------------------

    def build_headers(self, api_key: str) -> dict[str, str]:
        """Placeholder -- actual headers are built with SigV4 in build_signed_headers."""
        return {"Content-Type": "application/json"}

    def build_signed_headers(self, url: str, body: bytes,
                             **kwargs: Any) -> dict[str, str]:
        """Build SigV4-signed headers for Bedrock."""
        region = self._get_region()
        return sigv4_headers(
            method="POST",
            url=url,
            body=body,
            region=region,
            service="bedrock",
            access_key=kwargs.get("aws_access_key_id", ""),
            secret_key=kwargs.get("aws_secret_access_key", ""),
            session_token=kwargs.get("aws_session_token", ""),
        )

    # -- URL -----------------------------------------------------------------

    def build_url(self, model: str, endpoint: str = "",
                  base_url: Optional[str] = None, stream: bool = False,
                  **kwargs: Any) -> str:
        base = base_url or self._get_base_url()
        base = base.rstrip("/")
        action = "converse-stream" if stream else "converse"
        return f"{base}/model/{model}/{action}"

    # -- Body ----------------------------------------------------------------

    def build_body(self, model: str, messages: list, stream: bool = False,
                   **kwargs: Any) -> dict:
        """Build body for Bedrock Converse API."""
        max_tokens = kwargs.pop("max_tokens", None) or kwargs.pop(
            "max_completion_tokens", 4096)
        temperature = kwargs.pop("temperature", None)
        top_p = kwargs.pop("top_p", None)
        stop = kwargs.pop("stop", None) or kwargs.pop("stop_sequences", None)

        # Convert messages
        system_parts, converse_messages = self._convert_messages(messages)

        body: dict[str, Any] = {"messages": converse_messages}
        if system_parts:
            body["system"] = system_parts

        # inferenceConfig for max_tokens, temperature, etc.
        inference_config: dict[str, Any] = {"maxTokens": max_tokens}
        if temperature is not None:
            inference_config["temperature"] = temperature
        if top_p is not None:
            inference_config["topP"] = top_p
        if stop:
            inference_config["stopSequences"] = stop if isinstance(stop, list) else [stop]
        body["inferenceConfig"] = inference_config

        # Tools
        tools = kwargs.pop("tools", None)
        if tools:
            body["toolConfig"] = {"tools": self._convert_tools(tools)}

        tool_choice = kwargs.pop("tool_choice", None)
        if tool_choice:
            tc_config = self._convert_tool_choice(tool_choice)
            if "toolConfig" in body:
                body["toolConfig"].update(tc_config)
            else:
                body["toolConfig"] = tc_config

        return body

    def _convert_messages(self, messages: list) -> tuple[list[dict] | None, list[dict]]:
        """Convert OpenAI messages to Bedrock Converse format.

        Returns: (system_blocks, conversation_messages)
        """
        system_blocks: list[dict] = []
        conversation: list[dict] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                if isinstance(content, str):
                    system_blocks.append({"text": content})
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, str):
                            system_blocks.append({"text": block})
                        elif isinstance(block, dict):
                            system_blocks.append({"text": block.get("text", "")})
                continue

            # Map role
            bedrock_role = "assistant" if role == "assistant" else "user"

            # Convert content
            if isinstance(content, str):
                bedrock_content = [{"text": content}]
            elif isinstance(content, list):
                bedrock_content = [self._convert_content_block(b) for b in content]
            else:
                bedrock_content = [{"text": str(content)}]

            # Handle tool role -> user with toolResult
            if role == "tool":
                conversation.append({
                    "role": "user",
                    "content": [{
                        "toolResult": {
                            "toolUseId": msg.get("tool_call_id", ""),
                            "content": [{"text": content if isinstance(content, str) else str(content)}],
                        }
                    }],
                })
                continue

            # Handle assistant messages with tool_calls
            if role == "assistant" and msg.get("tool_calls"):
                blocks: list[dict] = []
                if content:
                    blocks.append({"text": content if isinstance(content, str) else str(content)})
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    try:
                        input_data = json.loads(fn.get("arguments", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        input_data = {}
                    blocks.append({
                        "toolUse": {
                            "toolUseId": tc.get("id", ""),
                            "name": fn.get("name", ""),
                            "input": input_data,
                        }
                    })
                conversation.append({"role": "assistant", "content": blocks})
                continue

            conversation.append({
                "role": bedrock_role,
                "content": bedrock_content or [{"text": ""}],
            })

        return system_blocks or None, conversation

    def _convert_content_block(self, block: dict) -> dict:
        """Convert a single OpenAI content block to Bedrock format."""
        if isinstance(block, str):
            return {"text": block}

        if block.get("type") == "text":
            return {"text": block.get("text", "")}

        url = extract_image_url(block)
        if url:
            return to_bedrock_image(url)

        # Fallback: treat as text
        return {"text": str(block.get("text", block.get("content", "")))}

    def _convert_tools(self, tools: list) -> list:
        """Convert OpenAI tools to Bedrock toolSpec format."""
        converted = []
        for tool in tools:
            if tool.get("type") == "function":
                fn = tool["function"]
                spec: dict[str, Any] = {
                    "toolSpec": {
                        "name": fn["name"],
                        "description": fn.get("description", ""),
                    }
                }
                params = fn.get("parameters")
                if params:
                    spec["toolSpec"]["inputSchema"] = {"json": params}
                converted.append(spec)
        return converted

    def _convert_tool_choice(self, choice: str | dict) -> dict:
        """Convert OpenAI tool_choice to Bedrock toolChoice format."""
        if isinstance(choice, str):
            if choice == "auto":
                return {"toolChoice": {"auto": {}}}
            elif choice == "required":
                return {"toolChoice": {"any": {}}}
            elif choice == "none":
                return {}
        if isinstance(choice, dict) and "function" in choice:
            return {
                "toolChoice": {
                    "tool": {"name": choice["function"]["name"]}
                }
            }
        return {"toolChoice": {"auto": {}}}

    # -- Response parsing ----------------------------------------------------

    def parse_response(self, data: dict, model: str = "") -> ModelResponse:
        """Parse Bedrock Converse response.

        Extracts from output.message.content[0].text.
        """
        output = data.get("output", {})
        message = output.get("message", {})
        content_blocks = message.get("content", [])

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in content_blocks:
            if "text" in block:
                text_parts.append(block["text"])
            elif "toolUse" in block:
                tu = block["toolUse"]
                tool_calls.append(ToolCall(
                    id=tu.get("toolUseId", ""),
                    type="function",
                    function=FunctionCall(
                        name=tu.get("name", ""),
                        arguments=json.dumps(tu.get("input", {})),
                    ),
                ))

        stop_reason = data.get("stopReason", "end_turn")
        finish_map = {
            "end_turn": "stop",
            "stop_sequence": "stop",
            "max_tokens": "length",
            "tool_use": "tool_calls",
            "content_filtered": "content_filter",
        }
        finish_reason = finish_map.get(stop_reason, "stop")

        msg = Message(
            role="assistant",
            content="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls or None,
        )

        usage = None
        u = data.get("usage", {})
        if u:
            usage = Usage(
                prompt_tokens=u.get("inputTokens", 0),
                completion_tokens=u.get("outputTokens", 0),
                total_tokens=u.get("totalTokens", 0),
            )

        return ModelResponse(
            model=model,
            choices=[Choice(index=0, message=msg, finish_reason=finish_reason)],
            usage=usage,
        )

    # -- Streaming -----------------------------------------------------------

    def parse_stream_line(self, data: dict, model: str = "") -> ModelResponse | None:
        """Parse Bedrock Converse streaming event.

        Event types: contentBlockDelta, messageStop, metadata.
        """
        # contentBlockDelta -> text content
        if "contentBlockDelta" in data:
            delta_block = data["contentBlockDelta"].get("delta", {})
            text = delta_block.get("text", "")
            if text:
                return ModelResponse(
                    model=model,
                    choices=[StreamChoice(delta=Delta(content=text))],
                    stream=True,
                )
            # Tool use delta
            if "toolUse" in delta_block:
                tu = delta_block["toolUse"]
                return ModelResponse(
                    model=model,
                    choices=[StreamChoice(delta=Delta(
                        tool_calls=[ToolCall(
                            function=FunctionCall(
                                arguments=tu.get("input", ""),
                            ),
                        )],
                    ))],
                    stream=True,
                )

        # contentBlockStart -> tool use start
        if "contentBlockStart" in data:
            start_block = data["contentBlockStart"].get("start", {})
            if "toolUse" in start_block:
                tu = start_block["toolUse"]
                return ModelResponse(
                    model=model,
                    choices=[StreamChoice(delta=Delta(
                        tool_calls=[ToolCall(
                            id=tu.get("toolUseId", ""),
                            type="function",
                            function=FunctionCall(
                                name=tu.get("name", ""),
                            ),
                        )],
                    ))],
                    stream=True,
                )

        # messageStop -> finish reason
        if "messageStop" in data:
            stop_reason = data["messageStop"].get("stopReason", "end_turn")
            finish_map = {
                "end_turn": "stop", "max_tokens": "length",
                "stop_sequence": "stop", "tool_use": "tool_calls",
            }
            return ModelResponse(
                model=model,
                choices=[StreamChoice(
                    delta=Delta(),
                    finish_reason=finish_map.get(stop_reason, "stop"),
                )],
                stream=True,
            )

        # metadata -> usage
        if "metadata" in data:
            u = data["metadata"].get("usage", {})
            if u:
                usage = Usage(
                    prompt_tokens=u.get("inputTokens", 0),
                    completion_tokens=u.get("outputTokens", 0),
                    total_tokens=u.get("totalTokens", 0),
                )
                return ModelResponse(
                    model=model,
                    choices=[StreamChoice(delta=Delta())],
                    usage=usage,
                    stream=True,
                )

        return None
