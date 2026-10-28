"""Turns an evaluation into text for the terminal and into JSON for the report artifact."""

from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from observable_analytics_agent.evaluation.gate import GateResult
from observable_analytics_agent.evaluation.metrics import EvaluationMetrics
from observable_analytics_agent.evaluation.results import TaskResult
from observable_analytics_agent.models.recording import ModelMode
from observable_analytics_agent.observability.records import PromptStamp
from observable_analytics_agent.text_table import render_table

NOT_AVAILABLE = "n/a"


def render_evaluation(
    results: Sequence[TaskResult], metrics: EvaluationMetrics, gate: GateResult | None
) -> str:
    sections = [_results_table(results), _summary(metrics)]
    if gate is not None:
        sections.append(_gate(gate))
    return "\n\n".join(sections) + "\n"


def report_payload(
    results: Sequence[TaskResult],
    metrics: EvaluationMetrics,
    gate: GateResult | None,
    *,
    prompt: PromptStamp,
    model: str,
    mode: ModelMode,
) -> dict[str, Any]:
    return {
        "mode": mode.value,
        "model": model,
        "prompt": prompt.model_dump(),
        "metrics": metrics.model_dump(),
        "gate": None if gate is None else _gate_payload(gate),
        "tasks": [_task_payload(result) for result in results],
    }


def _results_table(results: Sequence[TaskResult]) -> str:
    rows = [
        [
            result.task.task_id,
            "pass" if result.score.passed else "FAIL",
            result.outcome.steps_taken,
            _money(result.record.cost_usd),
            f"{result.latency_seconds:.2f}",
            "; ".join(result.score.failures),
        ]
        for result in results
    ]
    return render_table(
        "Task results", ["task", "result", "calls", "cost USD", "latency s", "notes"], rows
    )


def _summary(metrics: EvaluationMetrics) -> str:
    categories = ", ".join(
        f"{name} {rate:.1%}" for name, rate in metrics.success_rate_by_category.items()
    )
    glossary = (
        NOT_AVAILABLE
        if metrics.glossary_use_rate is None
        else f"{metrics.glossary_use_rate:.1%} of the tasks that need it"
    )
    lines = [
        "Summary",
        f"  tasks passed        {metrics.passed_count} of {metrics.task_count} "
        f"({metrics.success_rate:.1%})",
        f"  average cost        {_money(metrics.average_cost_usd)} USD per task",
        f"  p95 latency         {metrics.latency_p95_seconds:.2f} s",
        f"  average model calls {metrics.average_steps:.2f} per task",
        f"  glossary used       {glossary}",
        f"  missing recordings  {metrics.missing_recordings}",
        f"  by category         {categories}",
    ]
    return "\n".join(lines)


def _gate(gate: GateResult) -> str:
    lines = [f"Gate: {'PASS' if gate.passed else 'FAIL'}"]
    lines += [
        f"  {'PASS' if check.passed else 'FAIL'} {check.name}: {check.detail}"
        for check in gate.checks
    ]
    return "\n".join(lines)


def _gate_payload(gate: GateResult) -> dict[str, Any]:
    return {"passed": gate.passed, "checks": [asdict(check) for check in gate.checks]}


def _task_payload(result: TaskResult) -> dict[str, Any]:
    answer = result.outcome.answer
    return {
        "task_id": result.task.task_id,
        "category": result.task.category.value,
        "question": result.task.question,
        "passed": result.score.passed,
        "failures": list(result.score.failures),
        "run_status": result.outcome.status.value,
        "expected": {
            "kind": result.expected.kind.value,
            "number": result.expected.number,
            "text": result.expected.text,
            "labels": list(result.expected.labels),
        },
        "answer": answer.model_dump(mode="json") if answer else None,
        "model_calls": result.outcome.steps_taken,
        "cost_usd": result.record.cost_usd,
        "latency_seconds": result.latency_seconds,
    }


def _money(amount: float | None) -> str:
    return NOT_AVAILABLE if amount is None else f"{amount:.6f}"
