"""Limits that end a run gracefully instead of letting it spend without bound."""

from collections.abc import Callable
from dataclasses import dataclass

from observable_analytics_agent.agent.state import AgentState, RunStatus

DEFAULT_MAX_STEPS = 10
DEFAULT_MAX_TOTAL_TOKENS = 60_000
DEFAULT_TIMEOUT_SECONDS = 120.0

Clock = Callable[[], float]


@dataclass(frozen=True)
class RunBudget:
    """Model calls allowed, tokens billed across all calls, and wall-clock seconds."""

    max_steps: int = DEFAULT_MAX_STEPS
    max_total_tokens: int = DEFAULT_MAX_TOTAL_TOKENS
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS


def exceeded_budget(
    state: AgentState, budget: RunBudget, elapsed_seconds: float
) -> RunStatus | None:
    """Return the status that matches the first limit already reached, or None."""
    if state.steps_taken >= budget.max_steps:
        return RunStatus.STEP_BUDGET_EXCEEDED
    if state.usage.total_tokens >= budget.max_total_tokens:
        return RunStatus.TOKEN_BUDGET_EXCEEDED
    if elapsed_seconds >= budget.timeout_seconds:
        return RunStatus.TIMED_OUT
    return None
