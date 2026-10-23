"""The structured answer that ends an agent run."""

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, Field, model_validator

MAX_EXPLANATION_CHARS = 600

AnswerValue = float | str | list[str | float] | None


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    CANNOT_ANSWER = "cannot_answer"


class FinalAnswer(BaseModel):
    status: AnswerStatus = Field(
        description="'answered' when the data answers the question, 'cannot_answer' otherwise."
    )
    value: AnswerValue = Field(
        default=None,
        description="The answer: a number, a text, or a list for rankings. Null for cannot_answer.",
    )
    explanation: str = Field(
        min_length=1,
        max_length=MAX_EXPLANATION_CHARS,
        description="At most two sentences: how the value was obtained, or why it cannot be.",
    )
    queries_used: list[str] = Field(
        default_factory=list, description="The SQL queries that produced the value."
    )

    @model_validator(mode="after")
    def require_a_value_when_answered(self) -> Self:
        if self.status is AnswerStatus.ANSWERED and self.value is None:
            raise ValueError("value is required when status is 'answered'")
        return self
