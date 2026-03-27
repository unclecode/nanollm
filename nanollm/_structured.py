"""JSON extraction and validation utilities.

Extracts JSON from LLM responses (stripping markdown fences, finding
objects in mixed text) and optionally validates against a JSON schema --
all using only the standard library.
"""

from __future__ import annotations

import json
import re
from typing import Any


# ── JSON extraction ──────────────────────────────────────────────────

# Markdown code fences: ```json ... ``` or ``` ... ```
_FENCE_RE = re.compile(
    r"```(?:json|JSON)?\s*\n?(.*?)```",
    re.DOTALL,
)


def extract_json(text: str) -> Any:
    """Extract and parse JSON from an LLM response string.

    Strategy (in order):

    1. Try parsing the entire text as JSON.
    2. Look for markdown code fences and try each one.
    3. Find the first ``{...}`` or ``[...]`` block via brace matching.
    4. Raise ``ValueError`` if nothing works.
    """
    text = text.strip()

    # 1. Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Markdown fences
    for m in _FENCE_RE.finditer(text):
        candidate = m.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    # 3. Brace matching -- find outermost { } or [ ]
    result = _extract_braced(text, "{", "}")
    if result is not None:
        return result

    result = _extract_braced(text, "[", "]")
    if result is not None:
        return result

    raise ValueError(f"No valid JSON found in text: {text[:200]}...")


def _extract_braced(text: str, open_ch: str, close_ch: str) -> Any | None:
    """Find the first balanced brace pair and try to parse it as JSON."""
    start = text.find(open_ch)
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if escape:
            escape = False
            continue

        if ch == "\\":
            if in_string:
                escape = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None

    return None


# ── Schema validation ────────────────────────────────────────────────


def validate_json_response(text: str, schema: dict) -> Any:
    """Extract JSON from *text* and validate it against a JSON schema.

    This is a lightweight validator that checks:

    - Required properties
    - Type constraints (string, number, integer, boolean, array, object, null)
    - Enum values
    - Const values
    - Nested object / array schemas
    - ``anyOf`` / ``oneOf`` combinators

    For full JSON Schema compliance use a dedicated library.  This covers
    the common cases needed for structured LLM output.

    Returns the parsed + validated object, or raises ``ValueError``.
    """
    data = extract_json(text)
    _validate_value(data, schema, path="$")
    return data


def _validate_value(value: Any, schema: dict, path: str) -> None:
    """Recursively validate a value against a schema node."""
    # $ref -- not supported in this lightweight validator
    if "$ref" in schema:
        return

    # anyOf / oneOf -- pass if any sub-schema validates
    for key in ("anyOf", "oneOf"):
        if key in schema:
            errors: list[str] = []
            for sub in schema[key]:
                try:
                    _validate_value(value, sub, path)
                    return  # at least one matched
                except ValueError as e:
                    errors.append(str(e))
            raise ValueError(
                f"At {path}: value does not match any of {key}: "
                + "; ".join(errors)
            )

    # Type check
    schema_type = schema.get("type")
    if schema_type:
        _check_type(value, schema_type, path)

    # Enum check
    if "enum" in schema:
        if value not in schema["enum"]:
            raise ValueError(
                f"At {path}: {value!r} not in enum {schema['enum']}"
            )

    # Const check
    if "const" in schema:
        if value != schema["const"]:
            raise ValueError(
                f"At {path}: expected {schema['const']!r}, got {value!r}"
            )

    # Object properties
    if schema_type == "object" or "properties" in schema:
        if not isinstance(value, dict):
            return
        props = schema.get("properties", {})
        required = set(schema.get("required", []))

        for req_key in required:
            if req_key not in value:
                raise ValueError(
                    f"At {path}: missing required property '{req_key}'"
                )

        for prop_name, prop_schema in props.items():
            if prop_name in value:
                _validate_value(
                    value[prop_name], prop_schema, f"{path}.{prop_name}"
                )

    # Array items
    if schema_type == "array" or "items" in schema:
        if not isinstance(value, list):
            return
        items_schema = schema.get("items")
        if items_schema:
            for i, item in enumerate(value):
                _validate_value(item, items_schema, f"{path}[{i}]")


_TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _check_type(value: Any, schema_type: str, path: str) -> None:
    """Check that *value* matches the declared JSON Schema type."""
    if schema_type == "null":
        if value is not None:
            raise ValueError(
                f"At {path}: expected null, got {type(value).__name__}"
            )
        return

    expected = _TYPE_MAP.get(schema_type)
    if expected is None:
        return  # unknown type, skip

    if not isinstance(value, expected):
        # int is valid for "number"
        if schema_type == "number" and isinstance(value, (int, float)):
            return
        # bool is a subclass of int in Python
        if schema_type == "integer" and isinstance(value, bool):
            raise ValueError(f"At {path}: expected integer, got boolean")
        raise ValueError(
            f"At {path}: expected {schema_type}, got {type(value).__name__}"
        )


# ── Response format builder ──────────────────────────────────────────


def build_response_format(
    type: str = "json_schema",
    schema: dict | None = None,
    name: str = "response",
    strict: bool = True,
) -> dict:
    """Build a standardized ``response_format`` dict for structured output.

    Works with OpenAI's JSON Schema response format and can be adapted
    by provider code for other formats.

    Args:
        type: The response format type (e.g. ``"json_schema"``, ``"json_object"``).
        schema: The JSON Schema to validate against.
        name: A name for the schema (required by some providers).
        strict: Whether to enforce strict schema adherence.

    Returns:
        A dict suitable for passing as ``response_format`` in a request body.
    """
    if type == "json_object":
        return {"type": "json_object"}

    if type == "json_schema" and schema:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": name,
                "strict": strict,
                "schema": schema,
            },
        }

    return {"type": type}
