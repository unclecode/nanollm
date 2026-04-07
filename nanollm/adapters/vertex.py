"""Google Vertex AI adapter for NanoLLM.

Uses the Gemini generateContent format via Vertex AI endpoints
with Google Cloud authentication.
"""

from __future__ import annotations

import os
from typing import Any

from .._config import ProviderConfig
from .._types import ModelResponse
from .gemini import Adapter as GeminiAdapter


def _get_vertex_access_token() -> str:
    """Get access token for Vertex AI using google-auth."""
    try:
        import google.auth
        import google.auth.transport.requests

        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        credentials.refresh(google.auth.transport.requests.Request())
        return credentials.token
    except ImportError:
        raise ImportError(
            "google-auth is required for Vertex AI. "
            "Install with: pip install nanollm[gcp]"
        )
    except Exception as e:
        raise ValueError(
            f"Failed to get Vertex AI credentials: {e}. "
            "Ensure Application Default Credentials are configured: "
            "gcloud auth application-default login"
        ) from e


class Adapter(GeminiAdapter):
    """Adapter for Google Vertex AI (uses Gemini format with Google Cloud auth)."""

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
        project = kwargs.pop("vertex_project", None) or os.environ.get("VERTEX_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        location = kwargs.pop("vertex_location", None) or os.environ.get("VERTEX_LOCATION", "us-central1")

        if not project:
            raise ValueError(
                "Vertex AI requires a project ID. Set VERTEX_PROJECT or GOOGLE_CLOUD_PROJECT "
                "environment variable, or pass vertex_project parameter."
            )

        # Build Vertex AI endpoint
        if stream:
            endpoint = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/google/models/{model}:streamGenerateContent?alt=sse"
        else:
            endpoint = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/google/models/{model}:generateContent"

        # Get access token
        access_token = api_key or _get_vertex_access_token()

        # Use parent's message conversion but build our own request
        from .gemini import _convert_messages
        system_instruction, contents = _convert_messages(messages)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {access_token}",
        }

        body: dict[str, Any] = {"contents": contents}

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

        if "response_format" in kwargs:
            rf = kwargs["response_format"]
            if isinstance(rf, dict) and rf.get("type") == "json_object":
                gen_config["responseMimeType"] = "application/json"

        if gen_config:
            body["generationConfig"] = gen_config

        return endpoint, headers, body
