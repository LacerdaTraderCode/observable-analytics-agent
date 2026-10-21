"""The description of a tool as it is offered to a model."""

from typing import Any

from pydantic import BaseModel, ConfigDict


class ToolDefinition(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    input_schema: dict[str, Any]


def compact_json_schema(schema: Any) -> Any:
    """Drop the generated `title` keys: they cost tokens and say nothing the name does not."""
    if isinstance(schema, dict):
        return {
            key: compact_json_schema(value)
            for key, value in schema.items()
            if not (key == "title" and isinstance(value, str))
        }
    if isinstance(schema, list):
        return [compact_json_schema(item) for item in schema]
    return schema
