"""The tool contract and the registry that dispatches the tool calls a model makes."""

import json
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import Any, ClassVar

from pydantic import BaseModel, ValidationError

from observable_analytics_agent.tools.definition import ToolDefinition, compact_json_schema
from observable_analytics_agent.tools.errors import InvalidArguments, ToolError, UnknownTool
from observable_analytics_agent.tools.final_answer import FinalAnswer


class ToolResult(BaseModel):
    """What a tool hands back: text for the model plus a machine-readable outcome."""

    content: str
    is_error: bool = False
    error_code: str | None = None
    answer: FinalAnswer | None = None


def to_compact_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def error_result(error: ToolError) -> ToolResult:
    content = to_compact_json({"error": error.code, "message": str(error)})
    return ToolResult(content=content, is_error=True, error_code=error.code)


class Tool(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    arguments_model: ClassVar[type[BaseModel]]

    def definition(self) -> ToolDefinition:
        schema = compact_json_schema(self.arguments_model.model_json_schema())
        return ToolDefinition(name=self.name, description=self.description, input_schema=schema)

    def invoke(self, raw_arguments: Mapping[str, Any]) -> ToolResult:
        """Validate the arguments, then run. Raises ToolError for failures the model can fix."""
        try:
            arguments = self.arguments_model.model_validate(dict(raw_arguments))
        except ValidationError as error:
            raise InvalidArguments(_describe_validation_error(error)) from error
        return self.run(arguments)

    @abstractmethod
    def run(self, arguments: Any) -> ToolResult:
        """Execute with arguments already validated against `arguments_model`."""


class ToolExecutor(Protocol):
    """What the agent needs from its tools, so a wrapper can observe every call."""

    def definitions(self) -> list[ToolDefinition]: ...

    def execute(self, name: str, raw_arguments: Mapping[str, Any]) -> ToolResult: ...


class ToolRegistry:
    """Dispatches tool calls by name and turns every ToolError into a result the model can read."""

    def __init__(self, tools: Iterable[Tool]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    def definitions(self) -> list[ToolDefinition]:
        return [tool.definition() for tool in self._tools.values()]

    def execute(self, name: str, raw_arguments: Mapping[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        try:
            if tool is None:
                raise UnknownTool(f"No tool named '{name}'. Available: {', '.join(self._tools)}.")
            return tool.invoke(raw_arguments)
        except ToolError as error:
            return error_result(error)


def _describe_validation_error(error: ValidationError) -> str:
    problems = [
        f"{'.'.join(str(part) for part in problem['loc']) or 'arguments'}: {problem['msg']}"
        for problem in error.errors()
    ]
    return "; ".join(problems)
