"""Shared pytest configuration for nanollm tests.

API keys are read from environment variables. Set them in your shell or
export them before running live tests:

    export OPENAI_API_KEY=...
    export GEMINI_API_KEY=...
    export AWS_ACCESS_KEY_ID=...
    pytest -m live
"""

from __future__ import annotations


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "live: marks tests that make real API calls (deselect with -m 'not live')",
    )
    config.addinivalue_line(
        "markers",
        "bedrock: marks tests that require AWS credentials",
    )
    config.addinivalue_line(
        "markers",
        "crawl4ai: marks tests that require crawl4ai to be installed",
    )
