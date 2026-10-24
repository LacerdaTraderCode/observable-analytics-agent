"""The model interface the agent depends on, and the data that crosses it."""

from collections.abc import Sequence
from typing import Any, Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field

from observable_analytics_agent.tools.definition import ToolDefinition


class ContentBlock(BaseModel):
    """One block of a message.

    Fields this code does not know, such as the signature of a thinking block, are kept because
    the API requires the blocks it returned to come back unchanged.
    """

    model_config = ConfigDict(extra="allow")

    type: str

    def get(self, name: str, default: Any = None) -> Any:
        return (self.model_extra or {}).get(name, default)

    def to_wire(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]


def extract_tool_calls(blocks: Sequence[ContentBlock]) -> list[ToolCall]:
    return [
        ToolCall(id=block.get("id"), name=block.get("name"), arguments=block.get("input") or {})
        for block in blocks
        if block.type == "tool_use"
    ]


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: list[ContentBlock]

    @property
    def tool_calls(self) -> list[ToolCall]:
        return extract_tool_calls(self.content)

    def to_wire(self) -> dict[str, Any]:
        return {"role": self.role, "content": [block.to_wire() for block in self.content]}


def text_message(role: Literal["user", "assistant"], text: str) -> Message:
    return Message(role=role, content=[ContentBlock(type="text", text=text)])


def tool_result_block(tool_use_id: str, content: str, is_error: bool) -> ContentBlock:
    return ContentBlock(
        type="tool_result", tool_use_id=tool_use_id, content=content, is_error=is_error
    )


class TokenUsage(BaseModel):
    """Tokens the API billed for one call. Cache fields are zero when caching is not used."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_creation_1h_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )

    def __add__(self, other: Self) -> Self:
        return type(self)(
            **{name: getattr(self, name) + getattr(other, name) for name in type(self).model_fields}
        )


class ModelRequest(BaseModel):
    model: str
    system: str
    messages: list[Message]
    tools: list[ToolDefinition]
    max_tokens: int


class ModelReply(BaseModel):
    content: list[ContentBlock]
    model: str
    stop_reason: str | None = None
    usage: TokenUsage = Field(default_factory=TokenUsage)

    @property
    def tool_calls(self) -> list[ToolCall]:
        return extract_tool_calls(self.content)


class ChatModel(Protocol):
    def complete(self, request: ModelRequest) -> ModelReply: ...


class ModelError(Exception):
    """A model call failed; the run ends with the status model_failed instead of crashing."""


class ModelUnavailable(ModelError):
    """The provider could not be reached or rejected the call."""
