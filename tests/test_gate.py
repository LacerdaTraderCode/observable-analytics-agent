"""The gate fails the pull request when quality drops, and passes when it does not."""

from datetime import date

import pytest

from observable_analytics_agent.evaluation.baseline import Baseline
from observable_analytics_agent.evaluation.gate import (
    AVERAGE_STEP_BUDGET,
    LATENCY_P95_BUDGET_SECONDS,
    MAX_COST_INCREASE_RATIO,
    MAX_SUCCESS_RATE_DROP,
    RERECORD_COMMAND,
    GateResult,
    GateThresholds,
    evaluate_gate,
)
from observable_analytics_agent.evaluation.metrics import EvaluationMetrics, summarize
from tests.support import a_task_result

PROMPT_VERSION = "2.0.0"
MODEL = "claude-haiku-5-5"
TASK_COUNT = 32
BASELINE_COST = 0.001


def a_baseline(**changes: object) -> Baseline:
    fields = {
        "prompt_name": "analyst_system",
        "prompt_version": PROMPT_VERSION,
        "model": MODEL,
        "task_count": TASK_COUNT,
        "success_rate": 30 / TASK_COUNT,
        "average_cost_usd": BASELINE_COST,
        "latency_p95_seconds": 20.0,
        "average_steps": 4.0,
        "recorded_on": date(2026, 10, 24),
    }
    return Baseline(**{**fields, **changes})


def metrics_like_the_baseline(**changes: object) -> EvaluationMetrics:
    fields = {
        "task_count": TASK_COUNT,
        "passed_count": 30,
        "success_rate": 30 / TASK_COUNT,
        "average_cost_usd": BASELINE_COST,
        "latency_p95_seconds": 20.0,
        "average_steps": 4.0,
        "glossary_use_rate": 1.0,
        "missing_recordings": 0,
        "success_rate_by_category": {},
    }
    return EvaluationMetrics(**{**fields, **changes})


def gate_for(metrics: EvaluationMetrics, baseline: Baseline | None = None, **options) -> GateResult:
    return evaluate_gate(
        metrics, baseline or a_baseline(), prompt_version=PROMPT_VERSION, model=MODEL, **options
    )


def failed_checks(result: GateResult) -> list[str]:
    return [check.name for check in result.checks if not check.passed]


def test_the_documented_thresholds() -> None:
    assert MAX_SUCCESS_RATE_DROP == 0.05
    assert MAX_COST_INCREASE_RATIO == 0.15
    assert LATENCY_P95_BUDGET_SECONDS == 60.0
    assert AVERAGE_STEP_BUDGET == 7.0


def test_results_equal_to_the_baseline_pass_every_check() -> None:
    result = gate_for(metrics_like_the_baseline())

    assert result.passed
    assert [check.name for check in result.checks] == [
        "recordings",
        "baseline",
        "success rate",
        "cost",
        "latency",
        "steps",
    ]


def test_losing_one_task_is_inside_the_tolerance_and_losing_two_is_not() -> None:
    one_less = metrics_like_the_baseline(passed_count=29, success_rate=29 / TASK_COUNT)
    two_less = metrics_like_the_baseline(passed_count=28, success_rate=28 / TASK_COUNT)

    assert gate_for(one_less).passed
    assert failed_checks(gate_for(two_less)) == ["success rate"]


def test_the_success_rate_boundary_is_inclusive() -> None:
    baseline = a_baseline(success_rate=0.9)
    thresholds = GateThresholds(max_success_rate_drop=0.1)

    assert gate_for(metrics_like_the_baseline(success_rate=0.8), baseline, thresholds=thresholds).passed
    assert not gate_for(
        metrics_like_the_baseline(success_rate=0.79), baseline, thresholds=thresholds
    ).passed


def test_the_cost_may_rise_by_the_documented_percentage_and_no_more() -> None:
    at_the_limit = metrics_like_the_baseline(average_cost_usd=BASELINE_COST * 1.15)
    over_the_limit = metrics_like_the_baseline(average_cost_usd=BASELINE_COST * 1.16)

    assert gate_for(at_the_limit).passed
    assert failed_checks(gate_for(over_the_limit)) == ["cost"]


def test_an_unknown_cost_fails_the_gate() -> None:
    result = gate_for(metrics_like_the_baseline(average_cost_usd=None))

    assert failed_checks(result) == ["cost"]
    assert "no listed price" in result.checks[3].detail


def test_the_p95_latency_budget() -> None:
    at_the_budget = metrics_like_the_baseline(latency_p95_seconds=LATENCY_P95_BUDGET_SECONDS)
    over_the_budget = metrics_like_the_baseline(latency_p95_seconds=LATENCY_P95_BUDGET_SECONDS + 1)

    assert gate_for(at_the_budget).passed
    assert failed_checks(gate_for(over_the_budget)) == ["latency"]


def test_the_average_step_budget() -> None:
    at_the_budget = metrics_like_the_baseline(average_steps=AVERAGE_STEP_BUDGET)
    over_the_budget = metrics_like_the_baseline(average_steps=AVERAGE_STEP_BUDGET + 0.5)

    assert gate_for(at_the_budget).passed
    assert failed_checks(gate_for(over_the_budget)) == ["steps"]


def test_a_missing_recording_fails_with_the_command_to_record_again() -> None:
    result = gate_for(metrics_like_the_baseline(missing_recordings=3))

    assert failed_checks(result) == ["recordings"]
    detail = result.checks[0].detail
    assert "3 task(s)" in detail
    assert RERECORD_COMMAND in detail
    assert "evaluation/recordings/" in detail


@pytest.mark.parametrize(
    "changes",
    [
        {"prompt_version": "1.0.0"},
        {"model": "claude-sonnet-5-5"},
        {"task_count": TASK_COUNT + 1},
    ],
)
def test_a_baseline_for_another_setup_is_not_comparable(changes: dict[str, object]) -> None:
    result = gate_for(metrics_like_the_baseline(), a_baseline(**changes))

    assert failed_checks(result) == ["baseline"]
    assert RERECORD_COMMAND in result.checks[1].detail


def test_custom_thresholds_replace_the_defaults() -> None:
    metrics = metrics_like_the_baseline(average_cost_usd=BASELINE_COST * 1.5)

    assert not gate_for(metrics).passed
    assert gate_for(metrics, thresholds=GateThresholds(max_cost_increase_ratio=0.6)).passed


def test_a_degraded_set_of_results_fails_the_gate() -> None:
    healthy = [a_task_result(f"task-{index}") for index in range(TASK_COUNT)]
    degraded = [
        a_task_result(f"task-{index}", passed=index >= 10, cost_usd=BASELINE_COST)
        for index in range(TASK_COUNT)
    ]
    baseline = a_baseline(success_rate=summarize(healthy).success_rate, average_cost_usd=BASELINE_COST)

    assert gate_for(summarize(healthy), baseline).passed
    result = gate_for(summarize(degraded), baseline)

    assert not result.passed
    assert failed_checks(result) == ["success rate"]
    assert "68.8%" in result.checks[2].detail


def test_a_result_with_any_failed_check_does_not_pass() -> None:
    result = gate_for(metrics_like_the_baseline(missing_recordings=1, average_steps=99.0))

    assert not result.passed
    assert failed_checks(result) == ["recordings", "steps"]
