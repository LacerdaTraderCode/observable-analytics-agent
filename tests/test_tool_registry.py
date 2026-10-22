"""The registry offers the tools to the model and turns failures into readable results."""

import json

from observable_analytics_agent.tools.base import ToolRegistry
from observable_analytics_agent.tools.definition import compact_json_schema

EXPECTED_TOOL_NAMES = [
    "describe_schema",
    "run_read_only_query",
    "search_business_glossary",
    "submit_answer",
]


def test_offers_the_four_tools(tool_registry: ToolRegistry) -> None:
    definitions = tool_registry.definitions()

    assert [definition.name for definition in definitions] == EXPECTED_TOOL_NAMES
    assert all(definition.description for definition in definitions)
    assert all(definition.input_schema["type"] == "object" for definition in definitions)


def test_an_unknown_tool_is_reported_to_the_model(tool_registry: ToolRegistry) -> None:
    outcome = tool_registry.execute("drop_everything", {})

    assert outcome.is_error
    assert outcome.error_code == "unknown_tool"
    assert "run_read_only_query" in outcome.content


def test_missing_arguments_are_reported_to_the_model(tool_registry: ToolRegistry) -> None:
    outcome = tool_registry.execute("run_read_only_query", {})

    assert outcome.is_error
    assert outcome.error_code == "invalid_arguments"
    assert "sql" in json.loads(outcome.content)["message"]


def test_a_failing_query_is_a_result_not_an_exception(tool_registry: ToolRegistry) -> None:
    outcome = tool_registry.execute("run_read_only_query", {"sql": "SELECT missing FROM orders"})

    assert outcome.is_error
    assert outcome.error_code == "query_failed"


def test_a_successful_call_is_not_flagged(tool_registry: ToolRegistry) -> None:
    outcome = tool_registry.execute("run_read_only_query", {"sql": "SELECT count(*) FROM orders"})

    assert not outcome.is_error
    assert outcome.error_code is None
    assert outcome.answer is None


def test_submit_answer_carries_the_structured_answer(tool_registry: ToolRegistry) -> None:
    outcome = tool_registry.execute(
        "submit_answer", {"status": "answered", "value": 7, "explanation": "Counted."}
    )

    assert outcome.answer is not None
    assert outcome.answer.value == 7.0


def test_compacting_a_schema_keeps_a_property_that_is_named_title() -> None:
    schema = {"title": "Generated", "properties": {"title": {"type": "string", "title": "Title"}}}

    assert compact_json_schema(schema) == {"properties": {"title": {"type": "string"}}}
