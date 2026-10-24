"""The typed state that flows through the agent graph, and the outcome of a run."""

import operator
from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel, Field

from observable_analytics_agent.models.base import Message, TokenUsage
from observable_analytics_agent.tools.final_answer import FinalAnswer


class RunStatus(StrEnum):
    """How a run ended. `completed` means submit_answer was called, whatever the answer says."""

    COMPLETED = "completed"
    STEP_BUDGET_EXCEEDED = "step_budget_exceeded"
    TOKEN_BUDGET_EXCEEDED = "token_budget_exceeded"
    TIMED_OUT = "timed_out"
    NO_ANSWER = "no_answer"
    MODEL_FAILED = "model_failed"


class AgentState(BaseModel):
    run_id: str
    question: str
    started_at: float
    messages: Annotated[list[Message], operator.add] = Field(default_factory=list)
    steps_taken: int = 0
    usage: TokenUsage = Field(default_factory=TokenUsage)
    reminders_sent: int = 0
    answer: FinalAnswer | None = None
    status: RunStatus | None = None
    error: str | None = None


class RunOutcome(BaseModel):
    run_id: str
    question: str
    status: RunStatus
    answer: FinalAnswer | None
    steps_taken: int
    usage: TokenUsage
    error: str | None
    messages: list[Message]

    @classmethod
    def from_state(cls, state: AgentState) -> Self:
        if state.status is None:
            raise ValueError("The run ended without a final status.")
        return cls(
            run_id=state.run_id,
            question=state.question,
            status=state.status,
            answer=state.answer,
            steps_taken=state.steps_taken,
            usage=state.usage,
            error=state.error,
            messages=state.messages,
        )
