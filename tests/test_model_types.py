"""Message types keep what the API needs back, and the scripted model plays its script."""

import pytest

from observable_analytics_agent.models.base import (
    ContentBlock,
    Message,
    ModelRequest,
    ModelUnavailable,
    TokenUsage,
    text_message,
    tool_result_block,
)
from observable_analytics_agent.models.scripted import (
    ScriptedChatModel,
    ScriptExhausted,
    text_reply,
    tool_call,
    tool_use_reply,
)


def a_request(text: str = "question") -> ModelRequest:
    return ModelRequest(
        model="test-model",
        system="system",
        messages=[text_message("user", text)],
        tools=[],
        max_tokens=100,
    )


def test_unknown_block_fields_survive_a_round_trip() -> None:
    wire = {"type": "thinking", "thinking": "plan", "signature": "abc"}

    assert ContentBlock.model_validate(wire).to_wire() == wire


def test_empty_fields_are_left_out_of_the_wire_form() -> None:
    block = ContentBlock(type="text", text="hello", citations=None)

    assert block.to_wire() == {"type": "text", "text": "hello"}


def test_a_tool_result_block_carries_the_call_id_and_the_error_flag() -> None:
    block = tool_result_block("toolu_1", '{"error":"x"}', True)

    assert block.to_wire() == {
        "type": "tool_result",
        "tool_use_id": "toolu_1",
        "content": '{"error":"x"}',
        "is_error": True,
    }


def test_a_message_lists_its_tool_calls_in_order() -> None:
    reply = tool_use_reply(
        tool_call("describe_schema", call_id="toolu_a"),
        tool_call("run_read_only_query", {"sql": "SELECT 1"}, "toolu_b"),
    )
    message = Message(role="assistant", content=reply.content)

    assert [(call.id, call.name) for call in message.tool_calls] == [
        ("toolu_a", "describe_schema"),
        ("toolu_b", "run_read_only_query"),
    ]
    assert message.tool_calls[1].arguments == {"sql": "SELECT 1"}
    assert reply.tool_calls == message.tool_calls


def test_a_text_message_has_no_tool_calls() -> None:
    assert text_message("assistant", "done").tool_calls == []


def test_token_usage_adds_every_field_and_totals_the_billed_tokens() -> None:
    first = TokenUsage(input_tokens=10, output_tokens=5, cache_read_input_tokens=2)
    second = TokenUsage(input_tokens=1, output_tokens=1, cache_creation_input_tokens=4)

    combined = first + second

    assert combined.input_tokens == 11
    assert combined.cache_creation_input_tokens == 4
    assert combined.total_tokens == 11 + 6 + 4 + 2


def test_the_scripted_model_returns_the_replies_in_order() -> None:
    first, second = text_reply("one"), text_reply("two")
    model = ScriptedChatModel([first, second])

    assert model.complete(a_request()) is first
    assert model.complete(a_request()) is second


def test_the_scripted_model_keeps_an_independent_copy_of_each_request() -> None:
    model = ScriptedChatModel([text_reply("one")])
    request = a_request("before")
    model.complete(request)

    request.messages.append(text_message("user", "after"))

    assert len(model.requests[0].messages) == 1


def test_the_scripted_model_raises_the_errors_in_its_script() -> None:
    model = ScriptedChatModel([ModelUnavailable("connection lost")])

    with pytest.raises(ModelUnavailable, match="connection lost"):
        model.complete(a_request())


def test_the_scripted_model_reports_an_exhausted_script() -> None:
    model = ScriptedChatModel([])

    with pytest.raises(ScriptExhausted, match="request 1"):
        model.complete(a_request())
