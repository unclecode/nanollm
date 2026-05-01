"""Live tests for all Bedrock authentication methods.

Verifies every supported auth path works end-to-end against the real
Bedrock Converse API (APAC cross-region inference profile, ap-south-1).

Run with:
    pytest -m "live and bedrock" tests/test_live_bedrock_auth.py

Required environment variables (at least one auth method must be set):

  SigV4 (IAM) — used by tests 1, 2, 5:
    AWS_ACCESS_KEY_ID       IAM access key
    AWS_SECRET_ACCESS_KEY   IAM secret key
    AWS_REGION              (optional) defaults to ap-south-1

  Bearer token — used by tests 3, 4:
    AWS_BEARER_TOKEN_BEDROCK  IAM Identity Center / SSO bearer token

Auth priority in nanollm (mirrors litellm):
  1. Explicit api_key kwarg       → Bearer token
  2. aws_access_key_id kwarg      → SigV4 with provided key pair
  3. AWS_ACCESS_KEY_ID env        → SigV4 via env / boto3 credential chain
  4. AWS_BEARER_TOKEN_BEDROCK env → Bearer token

Known limitations:
  - The model used here (apac.amazon.nova-micro-v1:0) is an APAC cross-region
    inference profile. These profiles require the calling IAM principal to have
    the bedrock:InvokeModel permission on the inference profile ARN.
  - Bearer tokens issued by IAM Identity Center (SSO) may NOT have access to
    cross-region inference profiles, causing a "model identifier is invalid"
    BadRequestError. If this happens with your bearer token, verify that your
    SSO permission set includes bedrock:InvokeModel on apac.* profile ARNs,
    or switch to SigV4 (IAM key) auth for these models.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from nanollm import completion, acompletion

MODEL   = "bedrock/apac.amazon.nova-micro-v1:0"
MSGS    = [{"role": "user", "content": "Reply with one word: hello"}]
REGION  = os.environ.get("AWS_REGION", "ap-south-1")

ACCESS_KEY   = os.getenv("AWS_ACCESS_KEY_ID", "")
SECRET_KEY   = os.getenv("AWS_SECRET_ACCESS_KEY", "")
BEARER_TOKEN = os.getenv("AWS_BEARER_TOKEN_BEDROCK", "")


def _has_iam():
    return bool(ACCESS_KEY and SECRET_KEY)


def _has_bearer():
    return bool(BEARER_TOKEN)


def _ok(response) -> bool:
    return bool(response.choices[0].message.content)


# ── Auth method tests ─────────────────────────────────────────────────────────


@pytest.mark.live
@pytest.mark.bedrock
def test_sigv4_via_env_vars():
    """Method 1: SigV4 using AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY env vars."""
    if not _has_iam():
        pytest.skip("AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY not set")

    r = completion(MODEL, messages=MSGS, max_tokens=20)
    assert _ok(r), "empty response"


@pytest.mark.live
@pytest.mark.bedrock
def test_sigv4_via_explicit_kwargs():
    """Method 2: SigV4 using explicit aws_access_key_id / aws_secret_access_key kwargs."""
    if not _has_iam():
        pytest.skip("AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY not set")

    # Temporarily remove env vars so only the kwargs path is exercised
    saved_key    = os.environ.pop("AWS_ACCESS_KEY_ID", "")
    saved_secret = os.environ.pop("AWS_SECRET_ACCESS_KEY", "")
    try:
        r = completion(
            MODEL, messages=MSGS, max_tokens=20,
            aws_access_key_id=saved_key,
            aws_secret_access_key=saved_secret,
            aws_region_name=REGION,
        )
        assert _ok(r), "empty response"
    finally:
        if saved_key:    os.environ["AWS_ACCESS_KEY_ID"]     = saved_key
        if saved_secret: os.environ["AWS_SECRET_ACCESS_KEY"] = saved_secret


@pytest.mark.live
@pytest.mark.bedrock
def test_bearer_token_via_api_key_param():
    """Method 3: Bearer token passed as explicit api_key param."""
    if not _has_bearer():
        pytest.skip("AWS_BEARER_TOKEN_BEDROCK not set")

    saved_key    = os.environ.pop("AWS_ACCESS_KEY_ID", "")
    saved_secret = os.environ.pop("AWS_SECRET_ACCESS_KEY", "")
    try:
        r = completion(MODEL, messages=MSGS, max_tokens=20, api_key=BEARER_TOKEN)
        assert _ok(r), "empty response"
    finally:
        if saved_key:    os.environ["AWS_ACCESS_KEY_ID"]     = saved_key
        if saved_secret: os.environ["AWS_SECRET_ACCESS_KEY"] = saved_secret


@pytest.mark.live
@pytest.mark.bedrock
def test_bearer_token_via_env_var():
    """Method 4: Bearer token via AWS_BEARER_TOKEN_BEDROCK env var."""
    if not _has_bearer():
        pytest.skip("AWS_BEARER_TOKEN_BEDROCK not set")

    saved_key    = os.environ.pop("AWS_ACCESS_KEY_ID", "")
    saved_secret = os.environ.pop("AWS_SECRET_ACCESS_KEY", "")
    try:
        r = completion(MODEL, messages=MSGS, max_tokens=20)
        assert _ok(r), "empty response"
    finally:
        if saved_key:    os.environ["AWS_ACCESS_KEY_ID"]     = saved_key
        if saved_secret: os.environ["AWS_SECRET_ACCESS_KEY"] = saved_secret


@pytest.mark.live
@pytest.mark.bedrock
def test_sigv4_async():
    """Method 5: Async path using acompletion() with SigV4 env vars."""
    if not _has_iam():
        pytest.skip("AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY not set")

    r = asyncio.run(acompletion(MODEL, messages=MSGS, max_tokens=20))
    assert _ok(r), "empty response"
