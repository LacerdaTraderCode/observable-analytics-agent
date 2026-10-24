"""The Anthropic adapter converts requests and replies; no test touches the network."""

from typing import Any

import anthropic
import httpx2
import pytest
from anthropic import types

from observable_analytics_agent.models.anthropic_model import AnthropicChatModel
from observable_analytics_agent.models.base import (
    ModelRequest,
    ModelUnavailable,
    text_message,
    tool_result_block,
)
from observable_analytics_agent.tools.definition import ToolDefinition

MODEL_NAME = "claude-haiku-5-5"


class StubMessages:
    def __init__(self, response: types.Message | None = None, error: Exception | None = None):
        self._response = response
        self._error = error
        self.calls: list[dict[str, Any]] = []

    def create(self, **arguments: Any) -> types.Message:
        self.calls.append(arguments)
        if self._error is not None:
            raise self._error
        return self._response


class StubClient:
    def __init__(self, messages: StubMessages) -> None:
        self.messages = messages


def sdk_message(content: list[Any], usage: types.Usage | None = None) -> types.Message:
    return types.Message(
        id="msg_1",
        type="message",
        role="assistant",
        model=MODEL_NAME,
        content=content,
        stop_reason="tool_use",
        stop_sequence=None,
        usage=usage or types.Usage(input_tokens=10, output_tokens=5),
    )


def a_request() -> ModelRequest:
    return ModelRequest(
        model=MODEL_NAME,
        system="You are an analyst.",
        messages=[
            text_message("user", "How many orders?"),
            text_message("assistant", "Let me check."),
            ModelRequest.model_construct(messages=[]).messages
            or text_message("user", "unused"),
        ][:2]
        + [
            text_message("user", "continue"),
        ],
        tools=[
            ToolDefinition(
                name="run_read_only_query",
                description="Run SQL.",
                input_schema={"type": "object", "properties": {"sql": {"type": "string"}}},
            )
        ],
        max_tokens=256,
    )


def adapter_for(messages: StubMessages) -> AnthropicChatModel:
    return AnthropicChatModel(StubClient(messages))


def test_sends_the_request_in_the_messages_api_format() -> None:
    messages = StubMessages(sdk_message([types.TextBlock(type="text", text="ok")]))

    adapter_for(messages).complete(a_request())

    sent = messages.calls[0]
    assert sent["model"] == MODEL_NAME
    assert sent["max_tokens"] == 256
    assert sent["system"] == "You are an analyst."
    assert sent["messages"][0] == {
        "role": "user",
        "content": [{"type": "text", "text": "How many orders?"}],
    }
    assert sent["tools"] == [
        {
            "name": "run_read_only_query",
            "description": "Run SQL.",
            "input_schema": {"type": "object", "properties": {"sql": {"type": "string"}}},
        }
    ]


def test_sends_tool_results_as_blocks_of_a_user_message() -> None:
    messages = StubMessages(sdk_message([types.TextBlock(type="text", text="ok")]))
    request = a_request()
    request.messages.append(
        text_message("user", "x").model_copy(
            update={"content": [tool_result_block("toolu_1", "{}", False)]}
        )
    )

    adapter_for(messages).complete(request)

    assert messages.calls[0]["messages"][-1]["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "toolu_1",
        "content": "{}",
        "is_error": False,
    }


def test_converts_text_and_tool_use_blocks() -> None:
    response = sdk_message(
        [
            types.TextBlock(type="text", text="Checking."),
            types.ToolUseBlock(
                type="tool_use", id="toolu_1", name="run_read_only_query", input={"sql": "SELECT 1"}
            ),
        ]
    )

    reply = adapter_for(StubMessages(response)).complete(a_request())

    assert reply.model == MODEL_NAME
    assert reply.stop_reason == "tool_use"
    assert [block.type for block in reply.content] == ["text", "tool_use"]
    assert reply.tool_calls[0].arguments == {"sql": "SELECT 1"}


def test_keeps_thinking_blocks_so_they_can_go_back_unchanged() -> None:
    response = sdk_message(
        [types.ThinkingBlock(type="thinking", thinking="plan", signature="sig-1")]
    )

    reply = adapter_for(StubMessages(response)).complete(a_request())

    assert reply.content[0].to_wire() == {
        "type": "thinking",
        "thinking": "plan",
        "signature": "sig-1",
    }


def test_maps_the_usage_including_cache_tokens() -> None:
    usage = types.Usage(
        input_tokens=10,
        output_tokens=5,
        cache_creation_input_tokens=3,
        cache_read_input_tokens=2,
        cache_creation=types.CacheCreation(ephemeral_1h_input_tokens=1, ephemeral_5m_input_tokens=2),
    )

    reply = adapter_for(StubMessages(sdk_message([], usage))).complete(a_request())

    assert reply.usage.input_tokens == 10
    assert reply.usage.output_tokens == 5
    assert reply.usage.cache_creation_input_tokens == 3
    assert reply.usage.cache_creation_1h_input_tokens == 1
    assert reply.usage.cache_read_input_tokens == 2


def test_usage_without_cache_fields_counts_zero_cache_tokens() -> None:
    reply = adapter_for(StubMessages(sdk_message([]))).complete(a_request())

    assert reply.usage.cache_creation_input_tokens == 0
    assert reply.usage.cache_creation_1h_input_tokens == 0
    assert reply.usage.cache_read_input_tokens == 0


def test_an_api_failure_becomes_a_model_error() -> None:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    failure = anthropic.APIConnectionError(message="connection reset", request=request)

    with pytest.raises(ModelUnavailable, match="connection reset"):
        adapter_for(StubMessages(error=failure)).complete(a_request())
