"""A model that plays a fixed script, for deterministic tests and offline demos."""

from collections import deque
from collections.abc import Iterable
from typing import Any

from observable_analytics_agent.models.base import (
    ContentBlock,
    ModelError,
    ModelReply,
    ModelRequest,
    TokenUsage,
    ToolCall,
)

SCRIPTED_MODEL_NAME = "scripted-model"
DEFAULT_SCRIPTED_USAGE = TokenUsage(input_tokens=100, output_tokens=20)


class ScriptExhausted(ModelError):
    """The agent asked for more replies than the script holds."""


class ScriptedChatModel:
    """Returns the scripted replies in order and keeps the requests it received."""

    def __init__(self, replies: Iterable[ModelReply | ModelError]) -> None:
        self._replies = deque(replies)
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelReply:
        self.requests.append(request.model_copy(deep=True))
        if not self._replies:
            raise ScriptExhausted(f"The script has no reply left for request {len(self.requests)}.")
        reply = self._replies.popleft()
        if isinstance(reply, ModelError):
            raise reply
        return reply


def tool_call(name: str, arguments: dict[str, Any] | None = None, call_id: str = "") -> ToolCall:
    return ToolCall(id=call_id or f"toolu_{name}", name=name, arguments=arguments or {})


def tool_use_reply(*calls: ToolCall, usage: TokenUsage | None = None) -> ModelReply:
    blocks = [
        ContentBlock(type="tool_use", id=call.id, name=call.name, input=call.arguments)
        for call in calls
    ]
    return _reply(blocks, "tool_use", usage)


def text_reply(text: str, usage: TokenUsage | None = None) -> ModelReply:
    return _reply([ContentBlock(type="text", text=text)], "end_turn", usage)


def submit_answer_call(
    value: Any, explanation: str = "Computed from the data.", call_id: str = ""
) -> ToolCall:
    arguments = {"status": "answered", "value": value, "explanation": explanation}
    return tool_call("submit_answer", arguments, call_id)


def _reply(blocks: list[ContentBlock], stop_reason: str, usage: TokenUsage | None) -> ModelReply:
    return ModelReply(
        content=blocks,
        model=SCRIPTED_MODEL_NAME,
        stop_reason=stop_reason,
        usage=usage or DEFAULT_SCRIPTED_USAGE,
    )
