"""Adapter from the ChatModel interface to the Anthropic Messages API."""

from typing import Any

import anthropic
from anthropic.types import Message as AnthropicMessage
from anthropic.types import Usage as AnthropicUsage

from observable_analytics_agent.models.base import (
    ContentBlock,
    ModelReply,
    ModelRequest,
    ModelUnavailable,
    TokenUsage,
)
from observable_analytics_agent.tools.definition import ToolDefinition


class AnthropicChatModel:
    """Sends one request to the Messages API and converts the reply.

    Retries for rate limits and transient failures belong to the SDK client, which does them
    before an error ever reaches this class.
    """

    def __init__(self, client: anthropic.Anthropic) -> None:
        self._client = client

    def complete(self, request: ModelRequest) -> ModelReply:
        try:
            response = self._client.messages.create(
                model=request.model,
                max_tokens=request.max_tokens,
                system=request.system,
                messages=[message.to_wire() for message in request.messages],
                tools=[_to_tool_param(tool) for tool in request.tools],
            )
        except anthropic.APIError as error:
            raise ModelUnavailable(f"The Anthropic API call failed: {error}") from error
        return _to_reply(response)


def _to_tool_param(tool: ToolDefinition) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema,
    }


def _to_reply(response: AnthropicMessage) -> ModelReply:
    return ModelReply(
        content=[
            ContentBlock.model_validate(block.model_dump(exclude_none=True))
            for block in response.content
        ],
        model=response.model,
        stop_reason=response.stop_reason,
        usage=_to_usage(response.usage),
    )


def _to_usage(usage: AnthropicUsage) -> TokenUsage:
    cache_creation = usage.cache_creation
    return TokenUsage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_creation_input_tokens=usage.cache_creation_input_tokens or 0,
        cache_creation_1h_input_tokens=cache_creation.ephemeral_1h_input_tokens
        if cache_creation
        else 0,
        cache_read_input_tokens=usage.cache_read_input_tokens or 0,
    )
