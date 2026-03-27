"""Exhaustive tests for nanollm._types -- ~100+ tests."""
from __future__ import annotations

import json
import time

import pytest

from nanollm._types import (
    _DictAccessMixin,
    _AttrDict,
    PromptTokensDetails,
    CompletionTokensDetails,
    Usage,
    FunctionCall,
    ToolCall,
    Message,
    Choice,
    Delta,
    StreamChoice,
    ModelResponse,
    TextChoice,
    TextCompletionResponse,
    EmbeddingData,
    EmbeddingResponse,
    make_model_response,
    make_embedding_response,
    make_stream_chunk,
    stream_chunk_builder,
    _generate_id,
)


# ── _generate_id ──────────────────────────────────────────────────────


class TestGenerateId:
    def test_default_prefix(self):
        r = _generate_id()
        assert r.startswith("chatcmpl-")

    def test_custom_prefix(self):
        r = _generate_id(prefix="cmpl-")
        assert r.startswith("cmpl-")

    def test_unique(self):
        ids = {_generate_id() for _ in range(100)}
        assert len(ids) == 100


# ── _DictAccessMixin ──────────────────────────────────────────────────


class TestDictAccessMixin:
    """Tests using a real slots class (FunctionCall)."""

    def test_getitem(self):
        fc = FunctionCall(name="fn", arguments='{"a":1}')
        assert fc["name"] == "fn"

    def test_getitem_missing_raises_keyerror(self):
        fc = FunctionCall()
        with pytest.raises(KeyError):
            fc["nonexistent"]

    def test_setitem(self):
        fc = FunctionCall()
        fc["name"] = "hello"
        assert fc.name == "hello"

    def test_get_present(self):
        fc = FunctionCall(name="fn")
        assert fc.get("name") == "fn"

    def test_get_missing_default(self):
        fc = FunctionCall()
        assert fc.get("missing", 42) == 42

    def test_get_missing_none(self):
        fc = FunctionCall()
        assert fc.get("missing") is None

    def test_contains_true(self):
        fc = FunctionCall(name="fn")
        assert "name" in fc

    def test_contains_false_slot_exists_but_attr_exists(self):
        fc = FunctionCall()
        # slot exists, attribute exists even if empty
        assert "name" in fc

    def test_keys(self):
        fc = FunctionCall()
        assert set(fc.keys()) == {"name", "arguments"}

    def test_items(self):
        fc = FunctionCall(name="fn", arguments="args")
        items = dict(fc.items())
        assert items == {"name": "fn", "arguments": "args"}

    def test_to_dict(self):
        fc = FunctionCall(name="fn", arguments='{"a":1}')
        d = fc.to_dict()
        assert d == {"name": "fn", "arguments": '{"a":1}'}

    def test_to_dict_omits_none(self):
        msg = Message(content=None)
        d = msg.to_dict()
        assert "content" not in d
        assert "role" in d

    def test_to_dict_recursive(self):
        tc = ToolCall(id="t1", function=FunctionCall(name="fn", arguments="{}"))
        d = tc.to_dict()
        assert isinstance(d["function"], dict)
        assert d["function"]["name"] == "fn"

    def test_to_dict_list_recursive(self):
        msg = Message(tool_calls=[ToolCall(id="t1")])
        d = msg.to_dict()
        assert isinstance(d["tool_calls"], list)
        assert isinstance(d["tool_calls"][0], dict)

    def test_model_dump(self):
        fc = FunctionCall(name="fn")
        assert fc.model_dump() == fc.to_dict()

    def test_json_method(self):
        fc = FunctionCall(name="fn", arguments="{}")
        j = fc.json()
        data = json.loads(j)
        assert data["name"] == "fn"

    def test_repr(self):
        fc = FunctionCall(name="fn")
        r = repr(fc)
        assert "FunctionCall" in r
        assert "fn" in r

    def test_repr_omits_none(self):
        msg = Message(content=None)
        r = repr(msg)
        assert "content" not in r

    def test_keys_hides_private(self):
        mr = ModelResponse()
        assert "_hidden_params" not in mr.keys()


# ── _AttrDict ─────────────────────────────────────────────────────────


class TestAttrDict:
    def test_attr_access(self):
        d = _AttrDict({"cached_tokens": 10})
        assert d.cached_tokens == 10

    def test_dict_access(self):
        d = _AttrDict({"cached_tokens": 10})
        assert d["cached_tokens"] == 10

    def test_missing_attr_raises(self):
        d = _AttrDict({})
        with pytest.raises(AttributeError):
            d.missing_key

    def test_setattr(self):
        d = _AttrDict({})
        d.foo = 42
        assert d["foo"] == 42

    def test_dunder_dict(self):
        d = _AttrDict({"a": 1, "b": 2})
        assert d.__dict__ == {"a": 1, "b": 2}

    def test_dunder_dict_is_plain_dict(self):
        d = _AttrDict({"x": 1})
        assert type(d.__dict__) is dict

    def test_is_dict_subclass(self):
        d = _AttrDict({"a": 1})
        assert isinstance(d, dict)


# ── PromptTokensDetails ──────────────────────────────────────────────


class TestPromptTokensDetails:
    def test_defaults(self):
        ptd = PromptTokensDetails()
        assert ptd.cached_tokens == 0
        assert ptd.audio_tokens == 0

    def test_custom(self):
        ptd = PromptTokensDetails(cached_tokens=100, audio_tokens=50)
        assert ptd.cached_tokens == 100

    def test_slots(self):
        ptd = PromptTokensDetails()
        assert hasattr(ptd, "__slots__")

    def test_dict_access(self):
        ptd = PromptTokensDetails(cached_tokens=5)
        assert ptd["cached_tokens"] == 5


# ── CompletionTokensDetails ──────────────────────────────────────────


class TestCompletionTokensDetails:
    def test_defaults(self):
        ctd = CompletionTokensDetails()
        assert ctd.reasoning_tokens == 0
        assert ctd.audio_tokens == 0
        assert ctd.accepted_prediction_tokens == 0
        assert ctd.rejected_prediction_tokens == 0

    def test_custom(self):
        ctd = CompletionTokensDetails(reasoning_tokens=100)
        assert ctd.reasoning_tokens == 100

    def test_slots(self):
        assert "reasoning_tokens" in CompletionTokensDetails.__slots__


# ── Usage ─────────────────────────────────────────────────────────────


class TestUsage:
    def test_defaults(self):
        u = Usage()
        assert u.prompt_tokens == 0
        assert u.completion_tokens == 0
        assert u.total_tokens == 0

    def test_auto_total(self):
        u = Usage(prompt_tokens=10, completion_tokens=20)
        assert u.total_tokens == 30

    def test_explicit_total(self):
        u = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=99)
        assert u.total_tokens == 99

    def test_wraps_dict_ptd_to_attrdict(self):
        u = Usage(prompt_tokens_details={"cached_tokens": 10})
        assert isinstance(u.prompt_tokens_details, _AttrDict)
        assert u.prompt_tokens_details.cached_tokens == 10
        assert u.prompt_tokens_details.__dict__ == {"cached_tokens": 10}

    def test_wraps_dict_ctd_to_attrdict(self):
        u = Usage(completion_tokens_details={"reasoning_tokens": 5})
        assert isinstance(u.completion_tokens_details, _AttrDict)
        assert u.completion_tokens_details.reasoning_tokens == 5

    def test_passes_through_ptd_object(self):
        ptd = PromptTokensDetails(cached_tokens=10)
        u = Usage(prompt_tokens_details=ptd)
        assert u.prompt_tokens_details is ptd

    def test_none_details(self):
        u = Usage()
        assert u.prompt_tokens_details is None
        assert u.completion_tokens_details is None

    def test_kwargs_ignored(self):
        # Should not raise on unknown kwargs
        u = Usage(extra_field=42)
        assert u.prompt_tokens == 0


# ── FunctionCall ──────────────────────────────────────────────────────


class TestFunctionCall:
    def test_defaults(self):
        fc = FunctionCall()
        assert fc.name == ""
        assert fc.arguments == ""

    def test_custom(self):
        fc = FunctionCall(name="fn", arguments='{"a":1}')
        assert fc.name == "fn"

    def test_slots(self):
        assert "name" in FunctionCall.__slots__


# ── ToolCall ──────────────────────────────────────────────────────────


class TestToolCall:
    def test_defaults(self):
        tc = ToolCall()
        assert tc.id == ""
        assert tc.type == "function"
        assert isinstance(tc.function, FunctionCall)

    def test_custom(self):
        tc = ToolCall(id="tc1", function=FunctionCall(name="f"))
        assert tc.id == "tc1"
        assert tc.function.name == "f"

    def test_to_dict(self):
        tc = ToolCall(id="tc1", function=FunctionCall(name="f", arguments="{}"))
        d = tc.to_dict()
        assert d["id"] == "tc1"
        assert d["function"]["name"] == "f"


# ── Message ───────────────────────────────────────────────────────────


class TestMessage:
    def test_defaults(self):
        m = Message()
        assert m.role == "assistant"
        assert m.content is None

    def test_reasoning_content(self):
        m = Message(reasoning_content="thinking...")
        assert m.reasoning_content == "thinking..."

    def test_tool_calls(self):
        m = Message(tool_calls=[ToolCall(id="t1")])
        assert len(m.tool_calls) == 1

    def test_function_call(self):
        m = Message(function_call=FunctionCall(name="fn"))
        assert m.function_call.name == "fn"

    def test_slots(self):
        assert "reasoning_content" in Message.__slots__


# ── Choice ────────────────────────────────────────────────────────────


class TestChoice:
    def test_defaults(self):
        c = Choice()
        assert c.index == 0
        assert isinstance(c.message, Message)
        assert c.finish_reason is None

    def test_custom(self):
        c = Choice(index=1, message=Message(content="hi"), finish_reason="stop")
        assert c.message.content == "hi"
        assert c.finish_reason == "stop"


# ── Delta ─────────────────────────────────────────────────────────────


class TestDelta:
    def test_defaults(self):
        d = Delta()
        assert d.role is None
        assert d.content is None
        assert d.reasoning_content is None

    def test_custom(self):
        d = Delta(content="hello", reasoning_content="think")
        assert d.content == "hello"


# ── StreamChoice ──────────────────────────────────────────────────────


class TestStreamChoice:
    def test_defaults(self):
        sc = StreamChoice()
        assert sc.index == 0
        assert isinstance(sc.delta, Delta)

    def test_custom(self):
        sc = StreamChoice(index=1, delta=Delta(content="x"), finish_reason="stop")
        assert sc.finish_reason == "stop"


# ── ModelResponse ─────────────────────────────────────────────────────


class TestModelResponse:
    def test_defaults(self):
        mr = ModelResponse()
        assert mr.object == "chat.completion"
        assert mr.id.startswith("chatcmpl-")
        assert isinstance(mr.created, int)
        assert len(mr.choices) == 1

    def test_stream_mode(self):
        mr = ModelResponse(stream=True)
        assert mr.object == "chat.completion.chunk"
        assert isinstance(mr.choices[0], StreamChoice)

    def test_custom_id(self):
        mr = ModelResponse(id="custom-id")
        assert mr.id == "custom-id"

    def test_hidden_params(self):
        mr = ModelResponse(_hidden_params={"custom_llm_provider": "openai"})
        assert mr._hidden_params["custom_llm_provider"] == "openai"

    def test_hidden_params_default(self):
        mr = ModelResponse()
        assert mr._hidden_params == {}

    def test_dict_access_choices(self):
        mr = ModelResponse()
        assert mr["choices"] is mr.choices

    def test_to_dict(self):
        mr = ModelResponse(model="gpt-4")
        d = mr.to_dict()
        assert d["model"] == "gpt-4"
        assert "choices" in d

    def test_created_is_recent(self):
        mr = ModelResponse()
        assert abs(mr.created - int(time.time())) < 2


# ── TextChoice ────────────────────────────────────────────────────────


class TestTextChoice:
    def test_defaults(self):
        tc = TextChoice()
        assert tc.text == ""
        assert tc.index == 0
        assert tc.finish_reason is None
        assert tc.logprobs is None

    def test_custom(self):
        tc = TextChoice(text="hello", index=1, finish_reason="stop")
        assert tc.text == "hello"

    def test_slots(self):
        assert "text" in TextChoice.__slots__


# ── TextCompletionResponse ────────────────────────────────────────────


class TestTextCompletionResponse:
    def test_defaults(self):
        tcr = TextCompletionResponse()
        assert tcr.object == "text_completion"
        assert tcr.id.startswith("cmpl-")
        assert tcr.choices == []

    def test_custom(self):
        tcr = TextCompletionResponse(
            model="gpt-4",
            choices=[TextChoice(text="hello")],
        )
        assert tcr.model == "gpt-4"
        assert tcr.choices[0].text == "hello"


# ── EmbeddingData ─────────────────────────────────────────────────────


class TestEmbeddingData:
    def test_defaults(self):
        ed = EmbeddingData()
        assert ed.object == "embedding"
        assert ed.embedding == []
        assert ed.index == 0

    def test_custom(self):
        ed = EmbeddingData(embedding=[0.1, 0.2], index=1)
        assert ed.embedding == [0.1, 0.2]

    def test_to_dict(self):
        ed = EmbeddingData(embedding=[0.1])
        d = ed.to_dict()
        assert d["embedding"] == [0.1]


# ── EmbeddingResponse ────────────────────────────────────────────────


class TestEmbeddingResponse:
    def test_defaults(self):
        er = EmbeddingResponse()
        assert er.object == "list"
        assert er.data == []
        assert isinstance(er.usage, Usage)

    def test_custom(self):
        er = EmbeddingResponse(
            model="text-embedding-ada-002",
            data=[EmbeddingData(embedding=[0.1])],
        )
        assert er.model == "text-embedding-ada-002"
        assert len(er.data) == 1


# ── Factory helpers ──────────────────────────────────────────────────


class TestMakeModelResponse:
    def test_basic(self):
        r = make_model_response("hello", model="gpt-4")
        assert r.choices[0].message.content == "hello"
        assert r.model == "gpt-4"

    def test_with_usage(self):
        r = make_model_response("hi", usage={"prompt_tokens": 10, "completion_tokens": 5})
        assert r.usage.prompt_tokens == 10

    def test_with_tool_calls(self):
        tc = [ToolCall(id="t1")]
        r = make_model_response("", tool_calls=tc)
        assert r.choices[0].message.tool_calls is tc

    def test_with_reasoning(self):
        r = make_model_response("hi", reasoning_content="think")
        assert r.choices[0].message.reasoning_content == "think"

    def test_with_provider(self):
        r = make_model_response("hi", provider="openai")
        assert r._hidden_params["custom_llm_provider"] == "openai"


class TestMakeEmbeddingResponse:
    def test_basic(self):
        r = make_embedding_response([[0.1, 0.2]], model="ada-002")
        assert len(r.data) == 1
        assert r.data[0].embedding == [0.1, 0.2]
        assert r.data[0].index == 0

    def test_multiple(self):
        r = make_embedding_response([[0.1], [0.2]])
        assert len(r.data) == 2
        assert r.data[1].index == 1


class TestMakeStreamChunk:
    def test_basic(self):
        c = make_stream_chunk(content="hi")
        assert c.stream is True
        assert c.choices[0].delta.content == "hi"

    def test_with_role(self):
        c = make_stream_chunk(role="assistant")
        assert c.choices[0].delta.role == "assistant"

    def test_with_finish_reason(self):
        c = make_stream_chunk(finish_reason="stop")
        assert c.choices[0].finish_reason == "stop"

    def test_with_reasoning(self):
        c = make_stream_chunk(reasoning_content="think")
        assert c.choices[0].delta.reasoning_content == "think"


# ── stream_chunk_builder ─────────────────────────────────────────────


class TestStreamChunkBuilder:
    def test_empty(self):
        r = stream_chunk_builder([])
        assert isinstance(r, ModelResponse)
        assert r.choices[0].message.content is None

    def test_single_chunk(self):
        chunks = [make_stream_chunk(content="hello")]
        r = stream_chunk_builder(chunks)
        assert r.choices[0].message.content == "hello"

    def test_multi_chunks(self):
        chunks = [
            make_stream_chunk(content="hello "),
            make_stream_chunk(content="world"),
        ]
        r = stream_chunk_builder(chunks)
        assert r.choices[0].message.content == "hello world"

    def test_role_preserved(self):
        chunks = [
            make_stream_chunk(role="assistant"),
            make_stream_chunk(content="hi"),
        ]
        r = stream_chunk_builder(chunks)
        assert r.choices[0].message.role == "assistant"

    def test_finish_reason(self):
        chunks = [
            make_stream_chunk(content="hi"),
            make_stream_chunk(finish_reason="stop"),
        ]
        r = stream_chunk_builder(chunks)
        assert r.choices[0].finish_reason == "stop"

    def test_usage_from_last_chunk(self):
        c1 = make_stream_chunk(content="hi")
        c2 = make_stream_chunk(content="!")
        c2.usage = Usage(prompt_tokens=10, completion_tokens=5)
        r = stream_chunk_builder([c1, c2])
        assert r.usage.prompt_tokens == 10

    def test_model_preserved(self):
        chunks = [make_stream_chunk(content="hi", model="gpt-4")]
        r = stream_chunk_builder(chunks)
        assert r.model == "gpt-4"

    def test_reasoning_content(self):
        chunks = [
            make_stream_chunk(reasoning_content="think "),
            make_stream_chunk(reasoning_content="more"),
            make_stream_chunk(content="answer"),
        ]
        r = stream_chunk_builder(chunks)
        assert r.choices[0].message.reasoning_content == "think more"
        assert r.choices[0].message.content == "answer"

    def test_tool_calls_merged(self):
        tc1 = ToolCall(id="t1", function=FunctionCall(name="fn"))
        tc2 = ToolCall(function=FunctionCall(arguments='{"a":'))
        tc3 = ToolCall(function=FunctionCall(arguments="1}"))
        chunks = [
            make_stream_chunk(tool_calls=[tc1]),
            make_stream_chunk(tool_calls=[tc2]),
            make_stream_chunk(tool_calls=[tc3]),
        ]
        r = stream_chunk_builder(chunks)
        assert r.choices[0].message.tool_calls is not None
        assert len(r.choices[0].message.tool_calls) == 1
        assert r.choices[0].message.tool_calls[0].function.arguments == '{"a":1}'

    def test_not_stream(self):
        r = stream_chunk_builder([make_stream_chunk(content="hi")])
        assert r.stream is False
        assert r.object == "chat.completion"

    def test_id_preserved(self):
        c = make_stream_chunk(content="hi")
        c.id = "test-id"
        r = stream_chunk_builder([c])
        assert r.id == "test-id"

    def test_none_content_when_empty(self):
        chunks = [make_stream_chunk(finish_reason="stop")]
        r = stream_chunk_builder(chunks)
        assert r.choices[0].message.content is None

    def test_messages_arg_ignored(self):
        chunks = [make_stream_chunk(content="hi")]
        r = stream_chunk_builder(chunks, messages=[{"role": "user", "content": "hello"}])
        assert r.choices[0].message.content == "hi"

    def test_multiple_choices_stream(self):
        """Test tool calls across multiple indices."""
        tc0 = ToolCall(id="t0", function=FunctionCall(name="fn0"))
        tc0.index = 0
        tc1 = ToolCall(id="t1", function=FunctionCall(name="fn1"))
        tc1.index = 1
        c1 = make_stream_chunk(tool_calls=[tc0])
        c2 = make_stream_chunk(tool_calls=[tc1])
        r = stream_chunk_builder([c1, c2])
        # Both should be merged
        assert r.choices[0].message.tool_calls is not None


# ── Additional slot / mixin edge cases ───────────────────────────────


class TestAdditionalEdgeCases:
    def test_model_response_contains(self):
        mr = ModelResponse()
        assert "model" in mr
        assert "nonexistent" not in mr

    def test_model_response_setitem(self):
        mr = ModelResponse()
        mr["model"] = "new-model"
        assert mr.model == "new-model"

    def test_choice_to_dict_nested(self):
        c = Choice(message=Message(content="hi", tool_calls=[ToolCall(id="t1")]))
        d = c.to_dict()
        assert d["message"]["content"] == "hi"
        assert d["message"]["tool_calls"][0]["id"] == "t1"

    def test_delta_to_dict(self):
        d = Delta(content="hi", role="assistant")
        result = d.to_dict()
        assert result["content"] == "hi"
        assert result["role"] == "assistant"

    def test_stream_choice_to_dict(self):
        sc = StreamChoice(delta=Delta(content="hi"))
        d = sc.to_dict()
        assert "delta" in d

    def test_embedding_response_to_dict(self):
        er = EmbeddingResponse(
            model="ada",
            data=[EmbeddingData(embedding=[0.1])],
        )
        d = er.to_dict()
        assert d["model"] == "ada"
        assert len(d["data"]) == 1

    def test_usage_to_dict_with_details(self):
        u = Usage(
            prompt_tokens=10,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=5),
        )
        d = u.to_dict()
        assert d["prompt_tokens_details"]["cached_tokens"] == 5

    def test_model_response_json(self):
        mr = ModelResponse(model="gpt-4")
        j = mr.json()
        data = json.loads(j)
        assert data["model"] == "gpt-4"

    def test_text_completion_response_to_dict(self):
        tcr = TextCompletionResponse(
            model="gpt-4",
            choices=[TextChoice(text="hello")],
        )
        d = tcr.to_dict()
        assert d["choices"][0]["text"] == "hello"

    def test_function_call_to_dict_empty(self):
        fc = FunctionCall()
        d = fc.to_dict()
        assert d == {"name": "", "arguments": ""}

    def test_attrdict_in_usage_via_dict_access(self):
        u = Usage(prompt_tokens_details={"cached_tokens": 5})
        assert u["prompt_tokens_details"]["cached_tokens"] == 5

    def test_model_response_keys(self):
        mr = ModelResponse()
        keys = mr.keys()
        assert "id" in keys
        assert "model" in keys
        assert "choices" in keys

    def test_model_response_items(self):
        mr = ModelResponse(model="gpt-4")
        items = dict(mr.items())
        assert items["model"] == "gpt-4"

    def test_usage_model_dump(self):
        u = Usage(prompt_tokens=10)
        d = u.model_dump()
        assert d["prompt_tokens"] == 10

    def test_message_get_default(self):
        m = Message()
        assert m.get("content") is None
        # content attr exists (set to None), so get returns None not default
        assert m.get("content", "default") is None
        # But truly missing attribute returns default
        assert m.get("nonexistent_attr", "default") == "default"
