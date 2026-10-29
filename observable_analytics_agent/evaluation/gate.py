"""The CI gate: compares an evaluation with the committed baseline and with fixed budgets.

The thresholds are constants in this file on purpose. Changing one is a visible code change in a
pull request, never a quiet edit made to get a number to pass.
"""

from dataclasses import dataclass

from observable_analytics_agent.evaluation.baseline import Baseline
from observable_analytics_agent.evaluation.metrics import EvaluationMetrics

MAX_SUCCESS_RATE_DROP = 0.05
MAX_COST_INCREASE_RATIO = 0.15
LATENCY_P95_BUDGET_SECONDS = 60.0
AVERAGE_STEP_BUDGET = 7.0
COMPARISON_EPSILON = 1e-9
RERECORD_COMMAND = "python -m observable_analytics_agent evaluate --mode record --write-baseline"


@dataclass(frozen=True)
class GateThresholds:
    max_success_rate_drop: float = MAX_SUCCESS_RATE_DROP
    max_cost_increase_ratio: float = MAX_COST_INCREASE_RATIO
    latency_p95_budget_seconds: float = LATENCY_P95_BUDGET_SECONDS
    average_step_budget: float = AVERAGE_STEP_BUDGET


@dataclass(frozen=True)
class GateCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class GateResult:
    checks: tuple[GateCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)


def evaluate_gate(
    metrics: EvaluationMetrics,
    baseline: Baseline,
    *,
    prompt_version: str,
    model: str,
    thresholds: GateThresholds | None = None,
) -> GateResult:
    limits = thresholds or GateThresholds()
    return GateResult(
        checks=(
            _recordings_check(metrics),
            _comparability_check(metrics, baseline, prompt_version, model),
            _success_rate_check(metrics, baseline, limits),
            _cost_check(metrics, baseline, limits),
            _latency_check(metrics, limits),
            _steps_check(metrics, limits),
        )
    )


def _recordings_check(metrics: EvaluationMetrics) -> GateCheck:
    if metrics.missing_recordings == 0:
        return GateCheck("recordings", True, "every model call was replayed from a recording")
    return GateCheck(
        "recordings",
        False,
        f"{metrics.missing_recordings} task(s) hit a model call with no recording. A prompt, a "
        "tool definition or the dataset changed. Record again against the real model with "
        f"`{RERECORD_COMMAND}` and commit evaluation/recordings/ and evaluation/baseline.json.",
    )


def _comparability_check(
    metrics: EvaluationMetrics, baseline: Baseline, prompt_version: str, model: str
) -> GateCheck:
    recorded = (baseline.prompt_version, baseline.model, baseline.task_count)
    evaluated = (prompt_version, model, metrics.task_count)
    if recorded == evaluated:
        return GateCheck("baseline", True, "the baseline was recorded for the same setup")
    return GateCheck(
        "baseline",
        False,
        f"the baseline covers prompt {baseline.prompt_version} on {baseline.model} with "
        f"{baseline.task_count} tasks, but this run is prompt {prompt_version} on {model} with "
        f"{metrics.task_count} tasks. Record again with `{RERECORD_COMMAND}`.",
    )


def _success_rate_check(
    metrics: EvaluationMetrics, baseline: Baseline, limits: GateThresholds
) -> GateCheck:
    minimum = baseline.success_rate - limits.max_success_rate_drop
    detail = (
        f"success rate {metrics.success_rate:.1%} against a baseline of {baseline.success_rate:.1%} "
        f"(allowed drop {limits.max_success_rate_drop:.1%}, minimum {minimum:.1%})"
    )
    return GateCheck("success rate", metrics.success_rate >= minimum - COMPARISON_EPSILON, detail)


def _cost_check(
    metrics: EvaluationMetrics, baseline: Baseline, limits: GateThresholds
) -> GateCheck:
    if metrics.average_cost_usd is None:
        return GateCheck("cost", False, "the cost is unknown: the model has no listed price")
    maximum = baseline.average_cost_usd * (1 + limits.max_cost_increase_ratio)
    detail = (
        f"average cost {metrics.average_cost_usd:.6f} USD against a baseline of "
        f"{baseline.average_cost_usd:.6f} (allowed increase "
        f"{limits.max_cost_increase_ratio:.0%}, maximum {maximum:.6f})"
    )
    return GateCheck("cost", metrics.average_cost_usd <= maximum + COMPARISON_EPSILON, detail)


def _latency_check(metrics: EvaluationMetrics, limits: GateThresholds) -> GateCheck:
    detail = (
        f"p95 latency {metrics.latency_p95_seconds:.2f} s against a budget of "
        f"{limits.latency_p95_budget_seconds:.1f} s"
    )
    passed = metrics.latency_p95_seconds <= limits.latency_p95_budget_seconds + COMPARISON_EPSILON
    return GateCheck("latency", passed, detail)


def _steps_check(metrics: EvaluationMetrics, limits: GateThresholds) -> GateCheck:
    detail = (
        f"average of {metrics.average_steps:.2f} model calls per task against a budget of "
        f"{limits.average_step_budget:.1f}"
    )
    passed = metrics.average_steps <= limits.average_step_budget + COMPARISON_EPSILON
    return GateCheck("steps", passed, detail)
