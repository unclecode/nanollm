"""Live end-to-end tests for crawl4ai features backed by nanollm.

Covers all four crawl4ai LLM code paths:
  - LLMExtractionStrategy       — schema-guided JSON extraction from a webpage
  - JsonElementExtractionStrategy.agenerate_schema() — auto-generate CSS schemas
  - LLMTableExtraction          — intelligent table parsing
  - LLMContentFilter            — LLM-based markdown filtering/cleaning

Entire module is skipped automatically when crawl4ai is not installed.

Run with:
    pytest -m "live and crawl4ai" tests/test_live_crawl4ai.py

    # Single provider only:
    pytest -m "live and crawl4ai" tests/test_live_crawl4ai.py -k "gpt-4o-mini"

Required environment variables (set whichever providers you want to test):
  OPENAI_API_KEY          OpenAI
  ANTHROPIC_API_KEY       Anthropic (extraction only)
  GEMINI_API_KEY          Google Gemini
  GROQ_API_KEY            Groq (extraction and content filter only)
  MISTRAL_API_KEY         Mistral (extraction, schema, content filter)
  HUGGINGFACE_API_KEY     HuggingFace Inference API

  For Bedrock (APAC cross-region inference profiles, SigV4 auth):
    AWS_ACCESS_KEY_ID       IAM access key
    AWS_SECRET_ACCESS_KEY   IAM secret key
    AWS_REGION              (optional) defaults to ap-south-1

  Tests for a provider are automatically skipped if its key is not set.

Known limitations:
  - Bedrock providers use APAC cross-region inference profiles (apac.* model IDs).
    These require SigV4 (IAM key) auth — bearer tokens (AWS_BEARER_TOKEN_BEDROCK)
    issued by IAM Identity Center typically do NOT have access to apac.* profiles
    and will fail with "model identifier is invalid". Use IAM keys for Bedrock tests.
  - HuggingFace (Qwen2.5-72B-Instruct) may return empty tables in LLMTableExtraction
    because the model's context window is too small for the Wikipedia chemical
    elements page. This is a model limitation, not a nanollm bug.
  - Groq is excluded from LLMTableExtraction for the same reason.
"""

from __future__ import annotations

import asyncio
import json
import os

import pytest

crawl4ai = pytest.importorskip("crawl4ai", reason="crawl4ai not installed")

from crawl4ai import (                              # noqa: E402
    AsyncWebCrawler,
    CrawlerRunConfig,
    LLMConfig,
    LLMExtractionStrategy,
    LLMTableExtraction,
    CacheMode,
)
from crawl4ai.content_filter_strategy import LLMContentFilter          # noqa: E402
from crawl4ai.extraction_strategy import JsonElementExtractionStrategy # noqa: E402
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator  # noqa: E402

PORTFOLIO_URL = "https://sohamkukreti.github.io/portfolio"
SCHEMA_URL = "https://example.com"
WIKI_TABLE_URL = "https://en.wikipedia.org/wiki/List_of_chemical_elements"

# ── Provider lists ─────────────────────────────────────────────────────────────

EXTRACTION_PROVIDERS = [
    ("openai/gpt-4o-mini",                       "OPENAI_API_KEY",        None),
    ("anthropic/claude-3-haiku-20240307",         "ANTHROPIC_API_KEY",     None),
    ("gemini/gemini-2.5-flash",                   "GEMINI_API_KEY",        None),
    ("groq/llama-3.1-8b-instant",                 "GROQ_API_KEY",          None),
    ("mistral/mistral-small-latest",              "MISTRAL_API_KEY",       None),
    ("huggingface/Qwen/Qwen2.5-72B-Instruct",     "HUGGINGFACE_API_KEY",   None),
    ("bedrock/apac.amazon.nova-micro-v1:0",        None,                   "AWS_ACCESS_KEY_ID"),
    ("bedrock/apac.amazon.nova-lite-v1:0",         None,                   "AWS_ACCESS_KEY_ID"),
    ("bedrock/apac.anthropic.claude-3-haiku-20240307-v1:0",   None,        "AWS_ACCESS_KEY_ID"),
    ("bedrock/apac.anthropic.claude-3-5-sonnet-20241022-v2:0", None,       "AWS_ACCESS_KEY_ID"),
]

SCHEMA_PROVIDERS = [
    ("openai/gpt-4o-mini",                       "OPENAI_API_KEY",        None),
    ("gemini/gemini-2.5-flash",                   "GEMINI_API_KEY",        None),
    ("mistral/mistral-small-latest",              "MISTRAL_API_KEY",       None),
    ("huggingface/Qwen/Qwen2.5-72B-Instruct",     "HUGGINGFACE_API_KEY",   None),
    ("bedrock/apac.amazon.nova-micro-v1:0",        None,                   "AWS_ACCESS_KEY_ID"),
    ("bedrock/apac.amazon.nova-lite-v1:0",         None,                   "AWS_ACCESS_KEY_ID"),
    ("bedrock/apac.anthropic.claude-3-haiku-20240307-v1:0",   None,        "AWS_ACCESS_KEY_ID"),
    ("bedrock/apac.anthropic.claude-3-5-sonnet-20241022-v2:0", None,       "AWS_ACCESS_KEY_ID"),
]

TABLE_PROVIDERS = [
    ("openai/gpt-4o-mini",                        "OPENAI_API_KEY",       None),
    ("gemini/gemini-2.5-flash",                   "GEMINI_API_KEY",        None),
    ("huggingface/Qwen/Qwen2.5-72B-Instruct",     "HUGGINGFACE_API_KEY",   None),
    ("bedrock/apac.amazon.nova-micro-v1:0",        None,                   "AWS_ACCESS_KEY_ID"),
    ("bedrock/apac.anthropic.claude-3-haiku-20240307-v1:0",   None,        "AWS_ACCESS_KEY_ID"),
]

FILTER_PROVIDERS = [
    ("openai/gpt-4o-mini",                        "OPENAI_API_KEY",       None),
    ("gemini/gemini-2.5-flash",                   "GEMINI_API_KEY",        None),
    ("groq/llama-3.1-8b-instant",                 "GROQ_API_KEY",          None),
    ("mistral/mistral-small-latest",              "MISTRAL_API_KEY",       None),
    ("huggingface/Qwen/Qwen2.5-72B-Instruct",     "HUGGINGFACE_API_KEY",   None),
    ("bedrock/apac.amazon.nova-micro-v1:0",        None,                   "AWS_ACCESS_KEY_ID"),
]


def _ids(providers):
    return [p[0].split("/")[-1] for p in providers]


def _skip_if_no_creds(api_key_env, requires_env):
    if api_key_env and not os.getenv(api_key_env):
        pytest.skip(f"{api_key_env} not set")
    if requires_env and not os.getenv(requires_env):
        pytest.skip(f"{requires_env} not set")


def _make_llm_config(provider, api_key_env):
    if api_key_env:
        api_key = os.getenv(api_key_env)
    elif provider.startswith("bedrock/"):
        # Pass bearer token explicitly if available; otherwise None and nanollm
        # falls back to SigV4 using AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY from env
        api_key = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
    else:
        api_key = None
    return LLMConfig(provider=provider, api_token=api_key)


def _detect_llm_error(extracted_content: str) -> str | None:
    """Return error message if crawl4ai wrapped an LLM error in the JSON, else None."""
    try:
        parsed = json.loads(extracted_content)
        if isinstance(parsed, list) and parsed and parsed[0].get("error"):
            return parsed[0].get("content", "unknown LLM error")[:200]
    except (json.JSONDecodeError, AttributeError):
        pass
    return None


# ── LLMExtractionStrategy ─────────────────────────────────────────────────────


@pytest.mark.live
@pytest.mark.crawl4ai
@pytest.mark.parametrize("provider,key_env,req_env", EXTRACTION_PROVIDERS, ids=_ids(EXTRACTION_PROVIDERS))  # noqa: E501
def test_llm_extraction(provider, key_env, req_env):
    _skip_if_no_creds(key_env, req_env)

    async def _run():
        async with AsyncWebCrawler() as crawler:
            return await crawler.arun(
                url=PORTFOLIO_URL,
                config=CrawlerRunConfig(
                    extraction_strategy=LLMExtractionStrategy(
                        llm_config=_make_llm_config(provider, key_env),
                        instruction="Extract the experience name and description.",
                        schema={"name": "str", "description": "str"},
                    )
                ),
            )

    result = asyncio.run(_run())
    assert result.extracted_content, "extracted_content is empty"
    err = _detect_llm_error(result.extracted_content)
    assert err is None, f"LLM returned an error: {err}"


# ── generate_schema ───────────────────────────────────────────────────────────


@pytest.mark.live
@pytest.mark.crawl4ai
@pytest.mark.parametrize("provider,key_env,req_env", SCHEMA_PROVIDERS, ids=_ids(SCHEMA_PROVIDERS))
def test_generate_schema(provider, key_env, req_env):
    _skip_if_no_creds(key_env, req_env)

    schema = asyncio.run(JsonElementExtractionStrategy.agenerate_schema(
        url=SCHEMA_URL,
        query="Extract the title and content from the page.",
        llm_config=_make_llm_config(provider, key_env),
        validate=False,
    ))

    assert schema, "schema is None or empty"
    has_selector = bool(schema.get("baseSelector") or schema.get("base_selector"))
    has_fields   = bool(schema.get("fields") and len(schema["fields"]) > 0)
    assert has_selector, f"schema missing baseSelector: {schema}"
    assert has_fields,   f"schema missing fields: {schema}"


# ── LLMTableExtraction ────────────────────────────────────────────────────────


@pytest.mark.live
@pytest.mark.crawl4ai
@pytest.mark.parametrize("provider,key_env,req_env", TABLE_PROVIDERS, ids=_ids(TABLE_PROVIDERS))
def test_table_extraction(provider, key_env, req_env):
    _skip_if_no_creds(key_env, req_env)

    strategy = LLMTableExtraction(
        llm_config=_make_llm_config(provider, key_env),
        verbose=False,
        css_selector="div.mw-content-ltr",
        max_tries=2,
        enable_chunking=True,
        chunk_token_threshold=4000,
        min_rows_per_chunk=5,
        max_parallel_chunks=2,
    )

    async def _run():
        async with AsyncWebCrawler() as crawler:
            return await crawler.arun(
                url=WIKI_TABLE_URL,
                config=CrawlerRunConfig(cache_mode=CacheMode.BYPASS, table_extraction=strategy),
            )

    result = asyncio.run(_run())
    assert result.success, f"crawl failed: {result.error_message}"
    assert result.tables, "no tables extracted"
    table = result.tables[0]
    assert table.get("headers"), "table has no headers"
    assert table.get("rows"),    "table has no rows"


# ── LLMContentFilter ─────────────────────────────────────────────────────────


@pytest.mark.live
@pytest.mark.crawl4ai
@pytest.mark.parametrize("provider,key_env,req_env", FILTER_PROVIDERS, ids=_ids(FILTER_PROVIDERS))
def test_llm_content_filter(provider, key_env, req_env):
    _skip_if_no_creds(key_env, req_env)

    content_filter = LLMContentFilter(
        llm_config=_make_llm_config(provider, key_env),
        instruction=(
            "Extract only the work experience section. "
            "Include job titles, company names, dates, and descriptions."
        ),
        verbose=False,
    )

    async def _run():
        async with AsyncWebCrawler() as crawler:
            return await crawler.arun(
                url=PORTFOLIO_URL,
                config=CrawlerRunConfig(
                    cache_mode=CacheMode.BYPASS,
                    markdown_generator=DefaultMarkdownGenerator(content_filter=content_filter),
                ),
            )

    result = asyncio.run(_run())
    assert result.success, f"crawl failed: {result.error_message}"
    fit_md = result.markdown.fit_markdown if result.markdown else None
    assert fit_md and fit_md.strip(), "fit_markdown is empty"
