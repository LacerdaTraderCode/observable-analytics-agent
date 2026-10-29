"""The evaluate command: record, replay, baseline and gate, end to end with an oracle model."""

import json
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from observable_analytics_agent.cli import main
from observable_analytics_agent.evaluation import runner
from observable_analytics_agent.evaluation.golden_tasks import GOLDEN_TASKS
from observable_analytics_agent.models.anthropic_model import API_KEY_VARIABLE
from observable_analytics_agent.tools.sql_runner import ReadOnlyDatabase
from tests.oracle import OracleModel

REGRESSED_TASK_COUNT = 10
Evaluate = Callable[..., int]


def evaluate_in(workspace: Path, dataset_path: Path) -> Evaluate:
    def run(*extra: str) -> int:
        return main(
            [
                "evaluate",
                "--dataset",
                str(dataset_path),
                "--ledger",
                str(workspace / "ledger.duckdb"),
                "--recordings-dir",
                str(workspace / "recordings"),
                "--baseline",
                str(workspace / "baseline.json"),
                *extra,
            ]
        )

    return run


@pytest.fixture(scope="module")
def recorded(tmp_path_factory: pytest.TempPathFactory, dataset_path: Path) -> Path:
    """A workspace with recordings, a baseline and a ledger, made once by the oracle."""
    workspace = tmp_path_factory.mktemp("recorded")
    oracle = OracleModel(ReadOnlyDatabase(dataset_path))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runner, "build_anthropic_model", lambda environment, **_: oracle)
        assert evaluate_in(workspace, dataset_path)("--mode", "record", "--write-baseline") == 0
    return workspace


@pytest.fixture
def workspace(tmp_path: Path, recorded: Path) -> Path:
    copy = tmp_path / "workspace"
    shutil.copytree(recorded, copy)
    return copy


@pytest.fixture
def evaluate(workspace: Path, dataset_path: Path) -> Evaluate:
    return evaluate_in(workspace, dataset_path)


@pytest.fixture
def fresh_evaluate(tmp_path: Path, dataset_path: Path) -> Evaluate:
    return evaluate_in(tmp_path / "fresh", dataset_path)


@pytest.fixture
def mistakes(monkeypatch: pytest.MonkeyPatch, database: ReadOnlyDatabase) -> set[str]:
    """Make record and live runs talk to an oracle that answers these task ids wrongly."""
    wrong: set[str] = set()
    monkeypatch.setattr(
        runner, "build_anthropic_model", lambda environment, **_: OracleModel(database, wrong)
    )
    return wrong


def test_recording_writes_the_replies_and_the_baseline(recorded: Path) -> None:
    baseline = json.loads((recorded / "baseline.json").read_text())

    assert len(list((recorded / "recordings").glob("*.json"))) >= len(GOLDEN_TASKS)
    assert baseline["task_count"] == len(GOLDEN_TASKS)
    assert baseline["success_rate"] == 1.0
    assert (baseline["prompt_name"], baseline["prompt_version"]) == ("analyst_system", "2.0.0")
    assert baseline["average_cost_usd"] > 0


def test_replaying_the_recordings_passes_the_gate(
    evaluate: Evaluate, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = evaluate("--mode", "replay", "--gate")

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Gate: PASS" in output
    assert f"{len(GOLDEN_TASKS)} of {len(GOLDEN_TASKS)} (100.0%)" in output


def test_replay_needs_no_api_key_and_never_calls_the_model(
    evaluate: Evaluate, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(API_KEY_VARIABLE, raising=False)
    monkeypatch.setattr(runner, "build_anthropic_model", lambda *_, **__: pytest.fail("API used"))

    assert evaluate("--mode", "replay", "--gate") == 0


def test_a_regression_recorded_against_the_model_fails_the_gate(
    evaluate: Evaluate, mistakes: set[str], capsys: pytest.CaptureFixture[str]
) -> None:
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
    evaluate: Evaluate, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = evaluate("--mode", "replay", "--gate", "--prompt-version", "1.0.0")

    output = capsys.readouterr().out
    assert exit_code == 1
    assert f"{len(GOLDEN_TASKS)} task(s) hit a model call with no recording" in output
    assert "evaluate --mode record" in output
    assert "evaluation/recordings/" in output


def test_the_gate_without_a_baseline_fails_fast_and_says_how_to_create_one(
    workspace: Path,
    evaluate: Evaluate,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    (workspace / "baseline.json").unlink()

    exit_code = evaluate("--mode", "replay", "--gate")

    assert exit_code == 1
    assert capsys.readouterr().out == ""
    assert "No baseline" in caplog.text
    assert "evaluate --mode record --write-baseline" in caplog.text


def test_the_full_report_can_be_written_as_json(evaluate: Evaluate, workspace: Path) -> None:
    report_path = workspace / "report.json"

    assert evaluate("--mode", "replay", "--gate", "--output", str(report_path)) == 0

    report = json.loads(report_path.read_text())
    assert report["mode"] == "replay"
    assert report["prompt"]["version"] == "2.0.0"
    assert report["gate"]["passed"] is True
    assert report["metrics"]["success_rate"] == 1.0
    assert len(report["tasks"]) == len(GOLDEN_TASKS)
    assert {"task_id", "passed", "expected", "answer", "cost_usd"} <= set(report["tasks"][0])


def test_evaluation_runs_show_up_in_the_ledger_report(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["report", "--ledger", str(workspace / "ledger.duckdb")]) == 0

    output = capsys.readouterr().out
    assert "analyst_system@2.0.0" in output
    assert "count_customers" in output


def test_live_mode_does_not_store_recordings(
    tmp_path: Path, fresh_evaluate: Evaluate, mistakes: set[str]
) -> None:
    assert fresh_evaluate("--mode", "live", "--task", "count_customers") == 0

    assert not (tmp_path / "fresh" / "recordings").exists()


@pytest.mark.parametrize("mode", ["record", "live"])
def test_record_and_live_modes_need_an_api_key(
    mode: str,
    fresh_evaluate: Evaluate,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.delenv(API_KEY_VARIABLE, raising=False)

    exit_code = fresh_evaluate("--mode", mode)

    assert exit_code == 1
    assert API_KEY_VARIABLE in caplog.text


def test_a_subset_of_tasks_can_run_but_cannot_be_gated_or_stored(
    fresh_evaluate: Evaluate,
    mistakes: set[str],
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    subset = ["--task", "count_customers", "--task", "total_orders"]
    assert fresh_evaluate("--mode", "record", *subset) == 0
    output = capsys.readouterr().out
    assert "count_customers" in output
    assert "total_orders" in output
    assert "enterprise_customers" not in output

    assert fresh_evaluate("--mode", "replay", *subset, "--gate") == 2
    assert fresh_evaluate("--mode", "record", *subset, "--write-baseline") == 2
    assert "subset" in caplog.text


def test_an_unknown_task_is_a_usage_error(
    fresh_evaluate: Evaluate, caplog: pytest.LogCaptureFixture
) -> None:
    assert fresh_evaluate("--mode", "replay", "--task", "no_such_task") == 2
    assert "Unknown task(s): no_such_task" in caplog.text


def test_a_new_baseline_cannot_gate_itself(
    fresh_evaluate: Evaluate, caplog: pytest.LogCaptureFixture
) -> None:
    assert fresh_evaluate("--mode", "replay", "--gate", "--write-baseline") == 2
    assert "cannot be combined" in caplog.text
