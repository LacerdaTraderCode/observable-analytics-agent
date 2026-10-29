"""The evaluate command: record, replay, baseline and gate, end to end with an oracle model."""

import json
import logging
from pathlib import Path

import pytest

from observable_analytics_agent.cli import main
from observable_analytics_agent.evaluation import runner
from observable_analytics_agent.evaluation.golden_tasks import GOLDEN_TASKS
from observable_analytics_agent.models.anthropic_model import API_KEY_VARIABLE
from observable_analytics_agent.tools.sql_runner import ReadOnlyDatabase
from tests.oracle import OracleModel

REGRESSED_TASK_COUNT = 10


@pytest.fixture
def mistakes(monkeypatch: pytest.MonkeyPatch, database: ReadOnlyDatabase) -> set[str]:
    """Make record and live runs talk to the oracle instead of the Anthropic API.

    The returned set holds the task ids the oracle should answer wrongly.
    """
    wrong: set[str] = set()
    monkeypatch.setattr(
        runner,
        "build_anthropic_model",
        lambda environment, **_: OracleModel(database, wrong),
    )
    return wrong


@pytest.fixture
def evaluate(tmp_path: Path, dataset_path: Path):
    def run(*extra: str) -> int:
        return main(
            [
                "evaluate",
                "--dataset",
                str(dataset_path),
                "--ledger",
                str(tmp_path / "ledger.duckdb"),
                "--recordings-dir",
                str(tmp_path / "recordings"),
                "--baseline",
                str(tmp_path / "baseline.json"),
                *extra,
            ]
        )

    return run


def test_record_a_baseline_then_replay_it_through_the_gate(
    tmp_path: Path, mistakes: set[str], evaluate, capsys: pytest.CaptureFixture[str]
) -> None:
    assert evaluate("--mode", "record", "--write-baseline") == 0
    assert (tmp_path / "baseline.json").is_file()
    assert list((tmp_path / "recordings").glob("*.json"))
    capsys.readouterr()

    exit_code = evaluate("--mode", "replay", "--gate")

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Gate: PASS" in output
    assert f"{len(GOLDEN_TASKS)} of {len(GOLDEN_TASKS)} (100.0%)" in output


def test_replay_needs_no_api_key_and_never_calls_the_model(
    tmp_path: Path, mistakes: set[str], evaluate, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluate("--mode", "record", "--write-baseline")
    monkeypatch.delenv(API_KEY_VARIABLE, raising=False)
    monkeypatch.setattr(runner, "build_anthropic_model", lambda *_, **__: pytest.fail("API used"))

    assert evaluate("--mode", "replay", "--gate") == 0


def test_a_regression_recorded_against_the_model_fails_the_gate(
    mistakes: set[str], evaluate, capsys: pytest.CaptureFixture[str]
) -> None:
    evaluate("--mode", "record", "--write-baseline")
    mistakes.update(task.task_id for task in GOLDEN_TASKS[:REGRESSED_TASK_COUNT])
    assert evaluate("--mode", "record") == 0
    capsys.readouterr()

    exit_code = evaluate("--mode", "replay", "--gate")

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "Gate: FAIL" in output
    assert "FAIL success rate" in output
    assert "PASS recordings" in output


def test_a_changed_prompt_has_no_recordings_and_the_gate_says_how_to_record(
    mistakes: set[str], evaluate, capsys: pytest.CaptureFixture[str]
) -> None:
    evaluate("--mode", "record", "--write-baseline")
    capsys.readouterr()

    exit_code = evaluate("--mode", "replay", "--gate", "--prompt-version", "1.0.0")

    output = capsys.readouterr().out
    assert exit_code == 1
    assert f"{len(GOLDEN_TASKS)} task(s) hit a model call with no recording" in output
    assert "evaluate --mode record" in output
    assert "evaluation/recordings/" in output


def test_the_gate_without_a_baseline_fails_and_says_how_to_create_one(
    evaluate, caplog: pytest.LogCaptureFixture
) -> None:
    exit_code = evaluate("--mode", "replay", "--gate")

    assert exit_code == 1
    assert "No baseline" in caplog.text
    assert "evaluate --mode record --write-baseline" in caplog.text


def test_live_mode_does_not_store_recordings(
    tmp_path: Path, mistakes: set[str], evaluate
) -> None:
    assert evaluate("--mode", "live", "--task", "count_customers") == 0

    assert not (tmp_path / "recordings").exists()


@pytest.mark.parametrize("mode", ["record", "live"])
def test_record_and_live_modes_need_an_api_key(
    mode: str, evaluate, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv(API_KEY_VARIABLE, raising=False)

    exit_code = evaluate("--mode", mode)

    assert exit_code == 1
    assert API_KEY_VARIABLE in caplog.text


def test_a_subset_of_tasks_can_run_but_cannot_be_gated_or_stored(
    mistakes: set[str], evaluate, capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    assert evaluate("--mode", "record", "--task", "count_customers", "--task", "total_orders") == 0
    output = capsys.readouterr().out
    assert "count_customers" in output
    assert "total_orders" in output
    assert "enterprise_customers" not in output

    assert evaluate("--mode", "replay", "--task", "count_customers", "--gate") == 2
    assert evaluate("--mode", "record", "--task", "count_customers", "--write-baseline") == 2
    assert "subset" in caplog.text


def test_an_unknown_task_is_a_usage_error(evaluate, caplog: pytest.LogCaptureFixture) -> None:
    assert evaluate("--mode", "replay", "--task", "no_such_task") == 2
    assert "Unknown task(s): no_such_task" in caplog.text


def test_a_new_baseline_cannot_gate_itself(evaluate, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.ERROR):
        assert evaluate("--mode", "replay", "--gate", "--write-baseline") == 2

    assert "cannot be combined" in caplog.text


def test_the_full_report_can_be_written_as_json(
    tmp_path: Path, mistakes: set[str], evaluate
) -> None:
    report_path = tmp_path / "report.json"
    evaluate("--mode", "record", "--write-baseline")

    assert evaluate("--mode", "replay", "--gate", "--output", str(report_path)) == 0

    report = json.loads(report_path.read_text())
    assert report["mode"] == "replay"
    assert report["prompt"]["version"] == "2.0.0"
    assert report["gate"]["passed"] is True
    assert report["metrics"]["success_rate"] == 1.0
    assert len(report["tasks"]) == len(GOLDEN_TASKS)
    assert {"task_id", "passed", "expected", "answer", "cost_usd"} <= set(report["tasks"][0])


def test_evaluation_runs_show_up_in_the_ledger_report(
    mistakes: set[str], evaluate, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    evaluate("--mode", "record")
    capsys.readouterr()

    assert main(["report", "--ledger", str(tmp_path / "ledger.duckdb")]) == 0

    output = capsys.readouterr().out
    assert "analyst_system@2.0.0" in output
    assert "count_customers" in output
