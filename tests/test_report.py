"""The report aggregates the ledger by prompt version, task and tool."""

from pathlib import Path

import pytest

from observable_analytics_agent.agent.state import RunStatus
from observable_analytics_agent.observability.ledger import RunLedger
from observable_analytics_agent.observability.records import StepOutcome
from observable_analytics_agent.observability.report import build_report, render_report
from tests.support import a_run_record

FIRST_VERSION = "analyst_system@1.0.0"
SECOND_VERSION = "analyst_system@2.0.0"


@pytest.fixture
def ledger(tmp_path: Path) -> RunLedger:
    ledger = RunLedger(tmp_path / "ledger.duckdb")
    runs = [
        a_run_record(
            "run-a",
            cost_usd=0.001,
            latency_seconds=1.0,
            task_id="t1",
            tool_outcomes=[StepOutcome.OK, StepOutcome.ERROR],
        ),
        a_run_record("run-b", cost_usd=0.002, latency_seconds=2.0, task_id="t1"),
        a_run_record(
            "run-c",
            status=RunStatus.STEP_BUDGET_EXCEEDED,
            cost_usd=0.003,
            latency_seconds=3.0,
            task_id="t2",
        ),
        a_run_record(
            "run-d", prompt_version="2.0.0", cost_usd=0.0005, latency_seconds=1.5, task_id="t1"
        ),
    ]
    for run in runs:
        ledger.write(run)
    return ledger


def test_summarizes_runs_by_prompt_version(ledger: RunLedger) -> None:
    first, second = build_report(ledger).by_prompt

    assert first.prompt == FIRST_VERSION
    assert first.runs == 3
    assert first.success_rate == pytest.approx(2 / 3)
    assert first.average_cost_usd == pytest.approx(0.002)
    assert first.total_cost_usd == pytest.approx(0.006)
    assert first.latency_p50_seconds == pytest.approx(2.0)
    assert first.latency_p95_seconds == pytest.approx(2.9)
    assert second.prompt == SECOND_VERSION
    assert (second.runs, second.success_rate) == (1, 1.0)
    assert second.latency_p50_seconds == second.latency_p95_seconds == 1.5


def test_summarizes_cost_and_latency_by_task(ledger: RunLedger) -> None:
    by_task = {(row.task_id, row.prompt): row for row in build_report(ledger).by_task}

    assert set(by_task) == {("t1", FIRST_VERSION), ("t1", SECOND_VERSION), ("t2", FIRST_VERSION)}
    busiest = by_task[("t1", FIRST_VERSION)]
    assert busiest.runs == 2
    assert busiest.average_cost_usd == pytest.approx(0.0015)
    assert busiest.average_latency_seconds == pytest.approx(1.5)


def test_summarizes_the_tool_error_rate_by_prompt_version(ledger: RunLedger) -> None:
    first, second = build_report(ledger).tool_errors

    assert (first.prompt, first.tool, first.calls, first.errors) == (
        FIRST_VERSION,
        "run_read_only_query",
        4,
        1,
    )
    assert first.error_rate == pytest.approx(0.25)
    assert (second.prompt, second.calls, second.errors, second.error_rate) == (
        SECOND_VERSION,
        1,
        0,
        0.0,
    )


def test_runs_without_a_prompt_are_grouped_apart(tmp_path: Path) -> None:
    ledger = RunLedger(tmp_path / "ledger.duckdb")
    ledger.write(a_run_record().model_copy(update={"prompt": None}))

    assert [row.prompt for row in build_report(ledger).by_prompt] == ["unversioned"]


def test_the_rendered_report_shows_each_section(ledger: RunLedger) -> None:
    text = render_report(build_report(ledger))

    assert "Runs by prompt version" in text
    assert "Cost and latency by task" in text
    assert "Tool error rate by prompt version" in text
    assert FIRST_VERSION in text
    assert "66.7%" in text
    assert "0.002000" in text


def test_a_section_without_data_says_so(tmp_path: Path) -> None:
    ledger = RunLedger(tmp_path / "ledger.duckdb")
    ledger.write(a_run_record(task_id=None))

    text = render_report(build_report(ledger))

    assert "Cost and latency by task\n  (no data)" in text


def test_an_unknown_cost_is_shown_as_not_available(tmp_path: Path) -> None:
    ledger = RunLedger(tmp_path / "ledger.duckdb")
    ledger.write(a_run_record(cost_usd=None))

    assert "n/a" in render_report(build_report(ledger))
