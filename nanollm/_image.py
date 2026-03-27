"""Image / multimodal utilities for NanoLLM.

Handles conversion between OpenAI's ``image_url`` content-block format
and each provider's native format.  Supports base64 data URIs,
HTTP/HTTPS URLs, and provider-specific optimizations.

All functions are module-level (no class needed) for simple import::

    from nanollm._image import to_anthropic_image, is_multimodal_message
"""

from __future__ import annotations

import base64
import re
from typing import Any

import httpx

# Supported image MIME types
_MIME_MAP: dict[str, str] = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "gif": "image/gif",
    "webp": "image/webp",
}

# data:image/jpeg;base64,... pattern
_DATA_URI_RE = re.compile(r"^data:(image/\w+);base64,(.+)$", re.DOTALL)


# ── Parsing helpers ──────────────────────────────────────────────────


def parse_data_uri(url: str) -> tuple[str, str] | None:
    """Parse a data URI into ``(mime_type, base64_data)``.

    Returns ``None`` if the string is not a data URI.
    """
    m = _DATA_URI_RE.match(url)
    if m:
        return m.group(1), m.group(2)
    return None


def guess_mime_from_url(url: str) -> str:
    """Guess MIME type from URL file extension.  Defaults to image/jpeg."""
    url_lower = url.split("?")[0].lower()
    for ext, mime in _MIME_MAP.items():
        if url_lower.endswith(f".{ext}"):
            return mime
    return "image/jpeg"


def extract_image_url(content_block: dict) -> str | None:
    """Extract the URL string from an OpenAI ``image_url`` content block."""
    if content_block.get("type") != "image_url":
        return None
    image_url = content_block.get("image_url", {})
    if isinstance(image_url, str):
        return image_url
    return image_url.get("url")


def extract_image_detail(content_block: dict) -> str | None:
    """Extract the ``detail`` level from an OpenAI ``image_url`` block."""
    image_url = content_block.get("image_url", {})
    if isinstance(image_url, dict):
        return image_url.get("detail")
    return None


# ── Download helpers ─────────────────────────────────────────────────


def download_image_as_base64(url: str) -> tuple[str, str]:
    """Download an image URL and return ``(mime_type, base64_data)``.

    Used when a provider requires base64 but the user provided a URL.
    """
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()

    content_type = response.headers.get("content-type", "")
    if content_type.startswith("image/"):
        mime = content_type.split(";")[0].strip()
    else:
        mime = guess_mime_from_url(url)

    b64 = base64.b64encode(response.content).decode("ascii")
    return mime, b64


async def async_download_image_as_base64(url: str) -> tuple[str, str]:
    """Async version of :func:`download_image_as_base64`."""
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()

    content_type = response.headers.get("content-type", "")
    if content_type.startswith("image/"):
        mime = content_type.split(";")[0].strip()
    else:
        mime = guess_mime_from_url(url)

    b64 = base64.b64encode(response.content).decode("ascii")
    return mime, b64


# ── Multimodal detection ─────────────────────────────────────────────


def is_multimodal_message(message: dict) -> bool:
    """Check if a message contains multimodal content (``image_url`` blocks)."""
    content = message.get("content")
    if not isinstance(content, list):
        return False
    return any(
        isinstance(block, dict) and block.get("type") == "image_url"
        for block in content
    )


def has_multimodal_messages(messages: list[dict]) -> bool:
    """Check if any message in the list is multimodal."""
    return any(is_multimodal_message(m) for m in messages)


# ── Provider-specific conversions ────────────────────────────────────


def to_anthropic_image(url: str) -> dict:
    """Convert an image URL to Anthropic's native format.

    Anthropic accepts:
    - HTTPS URLs directly (``type: "url"``)
    - Base64 data (``type: "base64"``)

    HTTP URLs and data URIs are converted to base64.
    """
    parsed = parse_data_uri(url)
    if parsed:
        mime, data = parsed
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": mime, "data": data},
        }

    if url.startswith("https://"):
        return {
            "type": "image",
            "source": {"type": "url", "url": url},
        }

    # HTTP or other scheme -- download and convert to base64
    mime, data = download_image_as_base64(url)
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": mime, "data": data},
    }


def to_gemini_image(url: str) -> dict:
    """Convert an image URL to Gemini's native format.

    Gemini accepts:
    - HTTPS URLs and GCS URIs as ``file_data``
    - Base64 as ``inline_data``

    HTTP URLs and data URIs are converted to ``inline_data``.
    """
    parsed = parse_data_uri(url)
    if parsed:
        mime, data = parsed
        return {"inline_data": {"mime_type": mime, "data": data}}

    if url.startswith(("https://", "gs://")):
        mime = guess_mime_from_url(url)
        return {"file_data": {"mime_type": mime, "file_uri": url}}

    # HTTP -- download and inline
    mime, data = download_image_as_base64(url)
    return {"inline_data": {"mime_type": mime, "data": data}}


def to_bedrock_image(url: str) -> dict:
    """Convert an image URL to Bedrock's native format.

    Bedrock always requires base64 bytes -- no URL passthrough.
    """
    parsed = parse_data_uri(url)
    if parsed:
        mime, data = parsed
        fmt = mime.split("/")[1]
        return {"image": {"source": {"bytes": data}, "format": fmt}}

    # Download and convert
    mime, data = download_image_as_base64(url)
    fmt = mime.split("/")[1]
    return {"image": {"source": {"bytes": data}, "format": fmt}}
