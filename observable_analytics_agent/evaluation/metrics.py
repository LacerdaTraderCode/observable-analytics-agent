"""Aggregate metrics over the results of an evaluation run."""

import math
from collections import defaultdict
from collections.abc import Sequence

from pydantic import BaseModel

from observable_analytics_agent.agent.state import RunStatus
from observable_analytics_agent.evaluation.results import TaskResult
from observable_analytics_agent.models.recording import MISSING_RECORDING_PREFIX

LATENCY_TAIL = 0.95


class EvaluationMetrics(BaseModel):
    task_count: int
    passed_count: int
    success_rate: float
    average_cost_usd: float | None
    latency_p95_seconds: float
    average_steps: float
    glossary_use_rate: float | None
    missing_recordings: int
    success_rate_by_category: dict[str, float]


def percentile(values: Sequence[float], share: float) -> float:
    """Linear interpolation between ranks, the same definition DuckDB's quantile_cont uses."""
    ordered = sorted(values)
    position = share * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(results: Sequence[TaskResult]) -> EvaluationMetrics:
    if not results:
        raise ValueError("Cannot summarize an evaluation without results.")
    passed = [result for result in results if result.score.passed]
    costs = [result.record.cost_usd for result in results]
    needing_glossary = [result for result in results if result.task.requires_glossary]
    return EvaluationMetrics(
        task_count=len(results),
        passed_count=len(passed),
        success_rate=len(passed) / len(results),
        average_cost_usd=None if None in costs else sum(costs) / len(costs),
        latency_p95_seconds=percentile([r.latency_seconds for r in results], LATENCY_TAIL),
        average_steps=sum(result.outcome.steps_taken for result in results) / len(results),
        glossary_use_rate=(
            sum(result.score.used_glossary for result in needing_glossary) / len(needing_glossary)
            if needing_glossary
            else None
        ),
        missing_recordings=sum(_lacked_a_recording(result) for result in results),
        success_rate_by_category=_success_rate_by_category(results),
    )


def _lacked_a_recording(result: TaskResult) -> bool:
    outcome = result.outcome
    return outcome.status is RunStatus.MODEL_FAILED and (outcome.error or "").startswith(
        MISSING_RECORDING_PREFIX
    )


def _success_rate_by_category(results: Sequence[TaskResult]) -> dict[str, float]:
    outcomes: dict[str, list[bool]] = defaultdict(list)
    for result in results:
        outcomes[result.score.category.value].append(result.score.passed)
    return {category: sum(passed) / len(passed) for category, passed in sorted(outcomes.items())}
