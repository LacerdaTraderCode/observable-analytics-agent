"""submit_answer validates the structured answer that the evaluation scores."""

import pytest

from observable_analytics_agent.tools.errors import InvalidArguments
from observable_analytics_agent.tools.final_answer import AnswerStatus
from observable_analytics_agent.tools.submit_answer import SubmitAnswerTool

EXPLANATION = "Counted the rows."


def test_accepts_a_numeric_answer() -> None:
    outcome = SubmitAnswerTool().invoke(
        {
            "status": "answered",
            "value": 42,
            "explanation": EXPLANATION,
            "queries_used": ["SELECT 42"],
        }
    )

    assert not outcome.is_error
    assert outcome.answer.status is AnswerStatus.ANSWERED
    assert outcome.answer.value == 42.0
    assert outcome.answer.queries_used == ["SELECT 42"]


def test_accepts_a_ranking() -> None:
    outcome = SubmitAnswerTool().invoke(
        {"status": "answered", "value": ["web", "mobile_app"], "explanation": EXPLANATION}
    )

    assert outcome.answer.value == ["web", "mobile_app"]


def test_accepts_cannot_answer_without_a_value() -> None:
    outcome = SubmitAnswerTool().invoke(
        {"status": "cannot_answer", "explanation": "The dataset has no costs."}
    )

    assert outcome.answer.status is AnswerStatus.CANNOT_ANSWER
    assert outcome.answer.value is None


@pytest.mark.parametrize(
    "arguments",
    [
        {"status": "answered", "explanation": EXPLANATION},
        {"status": "answered", "value": 1, "explanation": ""},
        {"status": "maybe", "value": 1, "explanation": EXPLANATION},
        {"value": 1, "explanation": EXPLANATION},
    ],
    ids=["answered without value", "empty explanation", "unknown status", "missing status"],
)
def test_rejects_an_invalid_answer(arguments: dict[str, object]) -> None:
    with pytest.raises(InvalidArguments):
        SubmitAnswerTool().invoke(arguments)


def test_the_schema_offered_to_the_model_is_compact() -> None:
    schema = SubmitAnswerTool().definition().input_schema

    assert schema["required"] == ["status", "explanation"]
    assert "title" not in schema
    assert all("title" not in property_schema for property_schema in schema["properties"].values())
