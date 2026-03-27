"""Exhaustive tests for nanollm._structured -- ~30+ tests."""
from __future__ import annotations

import json

import pytest

from nanollm._structured import (
    extract_json,
    validate_json_response,
    build_response_format,
    _extract_braced,
)


# ── extract_json ─────────────────────────────────────────────────────


class TestExtractJson:
    def test_direct_object(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_direct_array(self):
        assert extract_json('[1, 2, 3]') == [1, 2, 3]

    def test_direct_string(self):
        assert extract_json('"hello"') == "hello"

    def test_direct_number(self):
        assert extract_json("42") == 42

    def test_direct_bool(self):
        assert extract_json("true") is True

    def test_direct_null(self):
        assert extract_json("null") is None

    def test_markdown_fence(self):
        text = '```json\n{"key": "value"}\n```'
        assert extract_json(text) == {"key": "value"}

    def test_markdown_fence_no_lang(self):
        text = '```\n{"key": "value"}\n```'
        assert extract_json(text) == {"key": "value"}

    def test_markdown_fence_JSON(self):
        text = '```JSON\n{"key": "value"}\n```'
        assert extract_json(text) == {"key": "value"}

    def test_json_in_prose(self):
        text = 'Here is the result: {"name": "Alice", "age": 30} done.'
        result = extract_json(text)
        assert result == {"name": "Alice", "age": 30}

    def test_array_in_prose(self):
        text = 'The list is: [1, 2, 3] and that is all.'
        assert extract_json(text) == [1, 2, 3]

    def test_nested_json(self):
        text = '{"a": {"b": [1, 2]}}'
        assert extract_json(text) == {"a": {"b": [1, 2]}}

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="No valid JSON"):
            extract_json("This has no JSON at all.")

    def test_whitespace_handling(self):
        text = '  \n  {"a": 1}  \n  '
        assert extract_json(text) == {"a": 1}

    def test_escaped_strings(self):
        text = '{"msg": "hello \\"world\\""}'
        result = extract_json(text)
        assert result["msg"] == 'hello "world"'

    def test_multiple_fences_takes_first(self):
        text = '```json\n{"first": true}\n```\n```json\n{"second": true}\n```'
        result = extract_json(text)
        assert result == {"first": True}

    def test_json_with_surrounding_text(self):
        text = 'Sure! Here is the JSON:\n\n{"name": "Bob"}\n\nHope that helps!'
        result = extract_json(text)
        assert result == {"name": "Bob"}


# ── _extract_braced ──────────────────────────────────────────────────


class TestExtractBraced:
    def test_finds_object(self):
        result = _extract_braced('prefix {"a": 1} suffix', "{", "}")
        assert result == {"a": 1}

    def test_finds_array(self):
        result = _extract_braced("prefix [1, 2] suffix", "[", "]")
        assert result == [1, 2]

    def test_no_match(self):
        assert _extract_braced("no braces here", "{", "}") is None

    def test_unbalanced(self):
        assert _extract_braced("{invalid", "{", "}") is None

    def test_nested_braces(self):
        result = _extract_braced('{"a": {"b": 1}}', "{", "}")
        assert result == {"a": {"b": 1}}


# ── validate_json_response ───────────────────────────────────────────


class TestValidateJsonResponse:
    def test_valid_object(self):
        text = '{"name": "Alice", "age": 30}'
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }
        result = validate_json_response(text, schema)
        assert result == {"name": "Alice", "age": 30}

    def test_missing_required(self):
        text = '{"name": "Alice"}'
        schema = {
            "type": "object",
            "required": ["name", "age"],
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        }
        with pytest.raises(ValueError, match="missing required"):
            validate_json_response(text, schema)

    def test_wrong_type(self):
        text = '{"name": 42}'
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
        with pytest.raises(ValueError, match="expected string"):
            validate_json_response(text, schema)

    def test_enum_valid(self):
        text = '"red"'
        schema = {"type": "string", "enum": ["red", "green", "blue"]}
        assert validate_json_response(text, schema) == "red"

    def test_enum_invalid(self):
        text = '"yellow"'
        schema = {"type": "string", "enum": ["red", "green", "blue"]}
        with pytest.raises(ValueError, match="not in enum"):
            validate_json_response(text, schema)

    def test_const(self):
        text = '42'
        schema = {"type": "integer", "const": 42}
        assert validate_json_response(text, schema) == 42

    def test_const_mismatch(self):
        text = '43'
        schema = {"type": "integer", "const": 42}
        with pytest.raises(ValueError, match="expected"):
            validate_json_response(text, schema)

    def test_array_items(self):
        text = '[1, 2, 3]'
        schema = {"type": "array", "items": {"type": "integer"}}
        assert validate_json_response(text, schema) == [1, 2, 3]

    def test_array_wrong_item_type(self):
        text = '[1, "two"]'
        schema = {"type": "array", "items": {"type": "integer"}}
        with pytest.raises(ValueError, match="expected integer"):
            validate_json_response(text, schema)

    def test_nested_object(self):
        text = '{"user": {"name": "Alice"}}'
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                }
            },
        }
        result = validate_json_response(text, schema)
        assert result["user"]["name"] == "Alice"

    def test_null_type(self):
        text = "null"
        schema = {"type": "null"}
        assert validate_json_response(text, schema) is None

    def test_null_type_mismatch(self):
        text = '"not null"'
        schema = {"type": "null"}
        with pytest.raises(ValueError, match="expected null"):
            validate_json_response(text, schema)

    def test_anyof(self):
        text = '"hello"'
        schema = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
        assert validate_json_response(text, schema) == "hello"

    def test_anyof_no_match(self):
        text = '"hello"'
        schema = {"anyOf": [{"type": "integer"}, {"type": "array"}]}
        with pytest.raises(ValueError, match="does not match"):
            validate_json_response(text, schema)

    def test_ref_skipped(self):
        # $ref is not validated -- just ignored
        text = '{"a": 1}'
        schema = {"$ref": "#/definitions/Foo"}
        result = validate_json_response(text, schema)
        assert result == {"a": 1}

    def test_boolean_passes_as_integer_due_to_python_subclassing(self):
        """In Python, bool is a subclass of int, so True passes isinstance(int)."""
        text = "true"
        schema = {"type": "integer"}
        # This actually passes because isinstance(True, int) is True
        result = validate_json_response(text, schema)
        assert result is True

    def test_number_accepts_int(self):
        text = "42"
        schema = {"type": "number"}
        assert validate_json_response(text, schema) == 42


# ── build_response_format ────────────────────────────────────────────


class TestBuildResponseFormat:
    def test_json_object(self):
        result = build_response_format(type="json_object")
        assert result == {"type": "json_object"}

    def test_json_schema(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        result = build_response_format(type="json_schema", schema=schema)
        assert result["type"] == "json_schema"
        assert result["json_schema"]["schema"] == schema
        assert result["json_schema"]["strict"] is True

    def test_json_schema_custom_name(self):
        result = build_response_format(type="json_schema", schema={"type": "object"},
                                       name="my_schema", strict=False)
        assert result["json_schema"]["name"] == "my_schema"
        assert result["json_schema"]["strict"] is False

    def test_unknown_type(self):
        result = build_response_format(type="text")
        assert result == {"type": "text"}

    def test_json_schema_no_schema(self):
        result = build_response_format(type="json_schema")
        assert result == {"type": "json_schema"}
