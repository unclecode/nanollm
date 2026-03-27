"""Exhaustive tests for nanollm._image -- ~40+ tests."""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from nanollm._image import (
    parse_data_uri,
    guess_mime_from_url,
    extract_image_url,
    extract_image_detail,
    download_image_as_base64,
    async_download_image_as_base64,
    is_multimodal_message,
    has_multimodal_messages,
    to_anthropic_image,
    to_gemini_image,
    to_bedrock_image,
)


# ── parse_data_uri ───────────────────────────────────────────────────


class TestParseDataUri:
    def test_valid_jpeg(self):
        uri = "data:image/jpeg;base64,abc123"
        result = parse_data_uri(uri)
        assert result == ("image/jpeg", "abc123")

    def test_valid_png(self):
        uri = "data:image/png;base64,xyz"
        result = parse_data_uri(uri)
        assert result == ("image/png", "xyz")

    def test_not_data_uri(self):
        assert parse_data_uri("https://example.com/img.png") is None

    def test_empty_string(self):
        assert parse_data_uri("") is None

    def test_partial_data_uri(self):
        assert parse_data_uri("data:image/jpeg") is None

    def test_valid_webp(self):
        uri = "data:image/webp;base64,data"
        result = parse_data_uri(uri)
        assert result == ("image/webp", "data")

    def test_long_base64(self):
        b64 = "A" * 1000
        uri = f"data:image/png;base64,{b64}"
        result = parse_data_uri(uri)
        assert result == ("image/png", b64)


# ── guess_mime_from_url ──────────────────────────────────────────────


class TestGuessMime:
    def test_jpeg(self):
        assert guess_mime_from_url("http://example.com/img.jpeg") == "image/jpeg"

    def test_jpg(self):
        assert guess_mime_from_url("http://example.com/img.jpg") == "image/jpeg"

    def test_png(self):
        assert guess_mime_from_url("http://example.com/img.png") == "image/png"

    def test_gif(self):
        assert guess_mime_from_url("http://example.com/img.gif") == "image/gif"

    def test_webp(self):
        assert guess_mime_from_url("http://example.com/img.webp") == "image/webp"

    def test_unknown_defaults_jpeg(self):
        assert guess_mime_from_url("http://example.com/file") == "image/jpeg"

    def test_with_query_params(self):
        assert guess_mime_from_url("http://example.com/img.png?w=200") == "image/png"

    def test_case_insensitive(self):
        assert guess_mime_from_url("http://example.com/IMG.PNG") == "image/png"

    def test_local_path(self):
        assert guess_mime_from_url("/tmp/test.webp") == "image/webp"


# ── extract_image_url ────────────────────────────────────────────────


class TestExtractImageUrl:
    def test_valid_block(self):
        block = {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}
        assert extract_image_url(block) == "https://example.com/img.png"

    def test_string_image_url(self):
        block = {"type": "image_url", "image_url": "https://example.com/img.png"}
        assert extract_image_url(block) == "https://example.com/img.png"

    def test_non_image_block(self):
        block = {"type": "text", "text": "hello"}
        assert extract_image_url(block) is None

    def test_missing_url(self):
        block = {"type": "image_url", "image_url": {}}
        assert extract_image_url(block) is None


# ── extract_image_detail ─────────────────────────────────────────────


class TestExtractImageDetail:
    def test_with_detail(self):
        block = {"type": "image_url", "image_url": {"url": "x", "detail": "high"}}
        assert extract_image_detail(block) == "high"

    def test_no_detail(self):
        block = {"type": "image_url", "image_url": {"url": "x"}}
        assert extract_image_detail(block) is None

    def test_string_image_url(self):
        block = {"type": "image_url", "image_url": "x"}
        assert extract_image_detail(block) is None


# ── is_multimodal_message ────────────────────────────────────────────


class TestIsMultimodal:
    def test_string_content_not_multimodal(self):
        assert is_multimodal_message({"content": "hello"}) is False

    def test_list_with_image(self):
        msg = {"content": [
            {"type": "text", "text": "hi"},
            {"type": "image_url", "image_url": {"url": "x"}},
        ]}
        assert is_multimodal_message(msg) is True

    def test_list_text_only(self):
        msg = {"content": [{"type": "text", "text": "hi"}]}
        assert is_multimodal_message(msg) is False

    def test_none_content(self):
        assert is_multimodal_message({"content": None}) is False

    def test_empty_content(self):
        assert is_multimodal_message({}) is False


class TestHasMultimodalMessages:
    def test_one_multimodal(self):
        msgs = [
            {"content": "text"},
            {"content": [{"type": "image_url", "image_url": {"url": "x"}}]},
        ]
        assert has_multimodal_messages(msgs) is True

    def test_none_multimodal(self):
        msgs = [{"content": "text"}, {"content": "more text"}]
        assert has_multimodal_messages(msgs) is False


# ── to_anthropic_image ───────────────────────────────────────────────


class TestToAnthropicImage:
    def test_data_uri(self):
        uri = "data:image/png;base64,abc123"
        result = to_anthropic_image(uri)
        assert result["type"] == "image"
        assert result["source"]["type"] == "base64"
        assert result["source"]["media_type"] == "image/png"
        assert result["source"]["data"] == "abc123"

    def test_https_url(self):
        result = to_anthropic_image("https://example.com/img.png")
        assert result["type"] == "image"
        assert result["source"]["type"] == "url"
        assert result["source"]["url"] == "https://example.com/img.png"

    def test_http_url_downloads(self):
        with patch("nanollm._image.download_image_as_base64",
                    return_value=("image/jpeg", "downloaded")):
            result = to_anthropic_image("http://example.com/img.jpg")
        assert result["source"]["type"] == "base64"
        assert result["source"]["data"] == "downloaded"


# ── to_gemini_image ──────────────────────────────────────────────────


class TestToGeminiImage:
    def test_data_uri(self):
        uri = "data:image/png;base64,abc123"
        result = to_gemini_image(uri)
        assert "inline_data" in result
        assert result["inline_data"]["mime_type"] == "image/png"

    def test_https_url(self):
        result = to_gemini_image("https://example.com/img.png")
        assert "file_data" in result
        assert result["file_data"]["file_uri"] == "https://example.com/img.png"

    def test_gs_url(self):
        result = to_gemini_image("gs://bucket/img.png")
        assert "file_data" in result

    def test_http_url_downloads(self):
        with patch("nanollm._image.download_image_as_base64",
                    return_value=("image/jpeg", "downloaded")):
            result = to_gemini_image("http://example.com/img.jpg")
        assert "inline_data" in result


# ── to_bedrock_image ─────────────────────────────────────────────────


class TestToBedrockImage:
    def test_data_uri(self):
        uri = "data:image/png;base64,abc123"
        result = to_bedrock_image(uri)
        assert result["image"]["source"]["bytes"] == "abc123"
        assert result["image"]["format"] == "png"

    def test_data_uri_jpeg(self):
        uri = "data:image/jpeg;base64,abc"
        result = to_bedrock_image(uri)
        assert result["image"]["format"] == "jpeg"

    def test_url_downloads(self):
        with patch("nanollm._image.download_image_as_base64",
                    return_value=("image/png", "downloaded")):
            result = to_bedrock_image("https://example.com/img.png")
        assert result["image"]["source"]["bytes"] == "downloaded"
        assert result["image"]["format"] == "png"


# ── download_image_as_base64 ────────────────────────────────────────


class TestDownloadImage:
    def test_download_sync(self):
        mock_response = MagicMock()
        mock_response.headers = {"content-type": "image/png"}
        mock_response.content = b"fake-image-data"
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response

        with patch("nanollm._image.httpx.Client", return_value=mock_client):
            mime, b64 = download_image_as_base64("https://example.com/img.png")
        assert mime == "image/png"
        assert len(b64) > 0

    def test_download_fallback_mime(self):
        mock_response = MagicMock()
        mock_response.headers = {"content-type": "application/octet-stream"}
        mock_response.content = b"fake"
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response

        with patch("nanollm._image.httpx.Client", return_value=mock_client):
            mime, _ = download_image_as_base64("https://example.com/img.png")
        assert mime == "image/png"
