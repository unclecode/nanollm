"""HuggingFace Inference Router provider.

Uses the HuggingFace Inference Router (router.huggingface.co/v1), which is a
drop-in OpenAI-compatible API that routes to the best backend automatically.

The old api-inference.huggingface.co endpoint is no longer supported.

Usage::

    model = "huggingface/Qwen/Qwen2.5-7B-Instruct"
    # -> POST https://router.huggingface.co/v1/chat/completions
    #    body: {"model": "Qwen/Qwen2.5-7B-Instruct", "messages": [...]}

Provider selection can be controlled by appending a suffix to the model name:

    "huggingface/Qwen/Qwen2.5-7B-Instruct"           # auto (fastest)
    "huggingface/Qwen/Qwen2.5-7B-Instruct:fastest"   # fastest provider
    "huggingface/Qwen/Qwen2.5-7B-Instruct:cheapest"  # cheapest provider
    "huggingface/Qwen/Qwen2.5-7B-Instruct:sambanova" # specific provider

Dedicated Inference Endpoints (pass api_base / base_url)::

    completion(model="huggingface/tgi",
               api_base="https://my-endpoint.endpoints.huggingface.cloud/v1/")

Auth: set ``HF_TOKEN`` (preferred) or ``HUGGINGFACE_API_KEY`` environment
variable, or pass ``api_key=`` explicitly.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from . import register
from .openai import OpenAIProvider


@register("huggingface")
class HuggingFaceProvider(OpenAIProvider):
    """HuggingFace Inference Router provider.

    Drop-in OpenAI-compatible API at router.huggingface.co/v1.
    The model name (e.g. ``Qwen/Qwen2.5-7B-Instruct``) is passed in the
    request body; the router selects the backend automatically.

    Model string format (nanollm convention)::

        "huggingface/<org>/<model>"
        e.g. "huggingface/Qwen/Qwen2.5-7B-Instruct"
             "huggingface/meta-llama/Llama-3.3-70B-Instruct"
    """

    name = "huggingface"
    base_url = "https://router.huggingface.co/v1"
    api_key_env = "HF_TOKEN"

    def get_api_key(self, api_key: Optional[str] = None) -> str:
        """Resolve HF token: explicit arg > HF_TOKEN > HUGGINGFACE_API_KEY."""
        return (
            api_key
            or os.environ.get("HF_TOKEN", "")
            or os.environ.get("HUGGINGFACE_API_KEY", "")
            or ""
        )
