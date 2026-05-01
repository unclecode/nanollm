"""Live tests for tool/function calling across multiple providers.

Run with:
    pytest -m live tests/test_live_tool_calling.py
"""

from __future__ import annotations

import asyncio
import json
import os

import pytest

from nanollm import completion, acompletion

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    },
}

MESSAGES = [{"role": "user", "content": "What's the weather like in London right now?"}]

PROVIDERS = [
    ("openai/gpt-4o-mini",                       "OPENAI_API_KEY"),
    ("anthropic/claude-3-haiku-20240307",         "ANTHROPIC_API_KEY"),
    ("gemini/gemini-2.5-flash",                   "GEMINI_API_KEY"),
    ("groq/llama-3.1-8b-instant",                 "GROQ_API_KEY"),
    ("mistral/mistral-small-latest",              "MISTRAL_API_KEY"),
    ("huggingface/Qwen/Qwen2.5-72B-Instruct",     "HUGGINGFACE_API_KEY"),
]


def _validate_tool_call(response):
    """Assert the response contains a valid get_weather tool call."""
    assert response.choices, "no choices in response"
    msg = response.choices[0].message
    assert msg.tool_calls, (
        f"tool_calls empty (finish_reason={response.choices[0].finish_reason!r}, "
        f"content={str(msg.content)[:80]!r})"
    )
    tc = msg.tool_calls[0]
    assert tc.function.name == "get_weather", f"unexpected tool name: {tc.function.name!r}"
    args = json.loads(tc.function.arguments)
    assert "location" in args, f"'location' missing from tool arguments: {args}"


@pytest.mark.live
@pytest.mark.parametrize("model,key_env", PROVIDERS, ids=[p[0].split("/")[-1] for p in PROVIDERS])
def test_tool_call_sync(model, key_env):
    api_key = os.getenv(key_env)
    if not api_key:
        pytest.skip(f"{key_env} not set")

    response = completion(
        model=model,
        messages=MESSAGES,
        tools=[WEATHER_TOOL],
        tool_choice="auto",
        api_key=api_key,
        max_tokens=256,
    )
    _validate_tool_call(response)


@pytest.mark.live
def test_tool_call_async_openai():
    """Async path — verified with OpenAI as the representative provider."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")

    response = asyncio.run(acompletion(
        model="openai/gpt-4o-mini",
        messages=MESSAGES,
        tools=[WEATHER_TOOL],
        tool_choice="auto",
        api_key=api_key,
        max_tokens=256,
    ))
    _validate_tool_call(response)
