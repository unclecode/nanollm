"""AWS Bedrock adapter for NanoLLM.

Supports Bedrock's converse API with optional SigV4 signing.
Uses boto3 if available, otherwise falls back to minimal inline SigV4.
"""

from __future__ import annotations

import datetime
import hashlib
import hmac
import json
import os
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


def _sign_v4(
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes,
    region: str,
    service: str,
    access_key: str,
    secret_key: str,
    session_token: str | None = None,
) -> dict[str, str]:
    """Minimal AWS SigV4 signer using only stdlib."""
    now = datetime.datetime.now(datetime.timezone.utc)
    datestamp = now.strftime("%Y%m%d")
    amzdate = now.strftime("%Y%m%dT%H%M%SZ")

    # Parse URL
    from urllib.parse import urlparse
    parsed = urlparse(url)
    host = parsed.hostname
    canonical_uri = parsed.path or "/"

    headers["host"] = host
    headers["x-amz-date"] = amzdate
    if session_token:
        headers["x-amz-security-token"] = session_token

    # Canonical headers
    signed_header_keys = sorted(headers.keys())
    canonical_headers = "".join(f"{k.lower()}:{headers[k].strip()}\n" for k in signed_header_keys)
    signed_headers = ";".join(k.lower() for k in signed_header_keys)

    payload_hash = hashlib.sha256(body).hexdigest()
    headers["x-amz-content-sha256"] = payload_hash

    canonical_request = (
        f"{method}\n{canonical_uri}\n\n"
        f"{canonical_headers}\n{signed_headers}\n{payload_hash}"
    )

    credential_scope = f"{datestamp}/{region}/{service}/aws4_request"
    string_to_sign = (
        f"AWS4-HMAC-SHA256\n{amzdate}\n{credential_scope}\n"
        f"{hashlib.sha256(canonical_request.encode()).hexdigest()}"
    )

    def _hmac_sha256(key: bytes, msg: str) -> bytes:
        return hmac.new(key, msg.encode(), hashlib.sha256).digest()

    signing_key = _hmac_sha256(
        _hmac_sha256(
            _hmac_sha256(
                _hmac_sha256(f"AWS4{secret_key}".encode("utf-8"), datestamp),
                region,
            ),
            service,
        ),
        "aws4_request",
    )

    signature = hmac.new(signing_key, string_to_sign.encode(), hashlib.sha256).hexdigest()
    headers["Authorization"] = (
        f"AWS4-HMAC-SHA256 Credential={access_key}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )

    return headers


def _get_aws_credentials() -> tuple[str, str, str | None, str]:
    """Get AWS credentials from environment or boto3."""
    region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))

    # Try boto3 first for full credential chain support
    try:
        import boto3
        session = boto3.Session()
        credentials = session.get_credentials()
        if credentials:
            creds = credentials.get_frozen_credentials()
            return creds.access_key, creds.secret_key, creds.token, session.region_name or region
    except ImportError:
        pass

    # Fall back to environment variables
    access_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    session_token = os.environ.get("AWS_SESSION_TOKEN")

    if not access_key or not secret_key:
        raise ValueError(
            "AWS credentials not found. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY, "
            "or install boto3 for full credential chain support (pip install nanollm[aws])."
        )

    return access_key, secret_key, session_token, region


def _convert_messages(messages: list[dict]) -> tuple[list[dict] | None, list[dict]]:
    """Convert OpenAI messages to Bedrock Converse format.

    Returns:
        (system_messages, converse_messages)
    """
    system_messages: list[dict] = []
    converse_messages: list[dict] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            if isinstance(content, str):
                system_messages.append({"text": content})
            continue

        bedrock_role = "assistant" if role == "assistant" else "user"

        if isinstance(content, str):
            converse_content = [{"text": content}]
        elif isinstance(content, list):
            converse_content = []
            for part in content:
                if isinstance(part, str):
                    converse_content.append({"text": part})
                elif isinstance(part, dict):
                    if part.get("type") == "text":
                        converse_content.append({"text": part.get("text", "")})
                    elif part.get("type") == "image_url":
                        image_url = part.get("image_url", {})
                        url = image_url.get("url", "") if isinstance(image_url, dict) else str(image_url)
                        if url.startswith("data:"):
                            media_type, _, data = url.partition(";base64,")
                            media_type = media_type.replace("data:", "")
                            import base64
                            converse_content.append({
                                "image": {
                                    "format": media_type.split("/")[-1],
                                    "source": {"bytes": base64.b64decode(data)},
                                },
                            })
        else:
            converse_content = [{"text": str(content)}]

        converse_messages.append({
            "role": bedrock_role,
            "content": converse_content,
        })

    return system_messages if system_messages else None, converse_messages


class Adapter(BaseAdapter):
    """Adapter for AWS Bedrock Converse API."""

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
        access_key, secret_key, session_token, region = _get_aws_credentials()

        action = "converse-stream" if stream else "converse"
        url = (
            base_url
            or f"https://bedrock-runtime.{region}.amazonaws.com/model/{model}/{action}"
        )

        system, converse_messages = _convert_messages(messages)

        body: dict[str, Any] = {
            "messages": converse_messages,
        }
        if system:
            body["system"] = system

        # Build inferenceConfig
        inference_config: dict[str, Any] = {}
        if "temperature" in kwargs and kwargs["temperature"] is not None:
            inference_config["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs and kwargs["max_tokens"] is not None:
            inference_config["maxTokens"] = kwargs["max_tokens"]
        if "top_p" in kwargs and kwargs["top_p"] is not None:
            inference_config["topP"] = kwargs["top_p"]
        if "stop" in kwargs and kwargs["stop"] is not None:
            stop = kwargs["stop"]
            inference_config["stopSequences"] = stop if isinstance(stop, list) else [stop]

        if inference_config:
            body["inferenceConfig"] = inference_config

        body_bytes = json.dumps(body).encode()

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        headers = _sign_v4(
            method="POST",
            url=url,
            headers=headers,
            body=body_bytes,
            region=region,
            service="bedrock",
            access_key=access_key,
            secret_key=secret_key,
            session_token=session_token,
        )

        return url, headers, body

    def parse_response(self, data: dict, model: str = "") -> ModelResponse:
        output = data.get("output", {})
        message = output.get("message", {})
        content_blocks = message.get("content", [])

        text_parts = []
        for block in content_blocks:
            if "text" in block:
                text_parts.append(block["text"])

        content = "".join(text_parts)

        usage_data = data.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("inputTokens", 0),
            completion_tokens=usage_data.get("outputTokens", 0),
            total_tokens=usage_data.get("inputTokens", 0) + usage_data.get("outputTokens", 0),
        )

        stop_reason = data.get("stopReason", "end_turn")
        finish_reason = "stop" if stop_reason == "end_turn" else stop_reason

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
            _hidden_params={"custom_llm_provider": "bedrock"},
        )

    def parse_stream_chunk(self, line: str, model: str = "") -> dict | None:
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return None

        # Bedrock converse stream events
        if "contentBlockDelta" in data:
            delta = data["contentBlockDelta"].get("delta", {})
            text = delta.get("text", "")
            return make_stream_chunk(content=text, model=model)

        if "messageStart" in data:
            return make_stream_chunk(role="assistant", model=model)

        if "messageStop" in data:
            stop_reason = data["messageStop"].get("stopReason", "end_turn")
            finish = "stop" if stop_reason == "end_turn" else stop_reason
            return make_stream_chunk(finish_reason=finish, model=model)

        return None
