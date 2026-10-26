"""The command line generates the dataset and reports what it wrote."""

import logging
import runpy
import sys
from pathlib import Path

import pytest

from observable_analytics_agent.cli import main
from observable_analytics_agent.dataset.generator import (
    DEFAULT_SEED,
    DatasetGenerationError,
    _latest_resolved_billing,
    _oldest_open_high_priority,
    summarize_database,
)
from observable_analytics_agent.observability.ledger import RunLedger
from observable_analytics_agent.prompts.registry import DEFAULT_PROMPTS_DIR, hash_content
from tests.support import a_run_record


def test_generate_dataset_writes_the_file_and_reports_the_tables(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "analytics.duckdb"

    with caplog.at_level(logging.INFO):
        exit_code = main(["generate-dataset", "--path", str(path)])

    assert exit_code == 0
    assert summarize_database(path).row_counts["customers"] > 0
    assert "support_tickets" in caplog.text
    assert summarize_database(path).fingerprint in caplog.text


def test_the_seed_option_changes_the_data(tmp_path: Path) -> None:
    default_path = tmp_path / "default.duckdb"
    other_path = tmp_path / "other.duckdb"

    main(["generate-dataset", "--path", str(default_path), "--seed", str(DEFAULT_SEED)])
    main(["generate-dataset", "--path", str(other_path), "--seed", str(DEFAULT_SEED + 1)])

    assert (
        summarize_database(default_path).fingerprint != summarize_database(other_path).fingerprint
    )


def test_the_module_can_be_run_with_python_dash_m(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "analytics.duckdb"
    monkeypatch.setattr(
        sys, "argv", ["observable_analytics_agent", "generate-dataset", "--path", str(path)]
    )

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("observable_analytics_agent", run_name="__main__")

    assert exit_info.value.code == 0
    assert path.is_file()


def test_report_prints_the_ledger_summaries(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ledger = RunLedger(tmp_path / "ledger.duckdb")
    ledger.write(a_run_record())

    exit_code = main(["report", "--ledger", str(ledger.path)])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Runs by prompt version" in output
    assert "analyst_system@1.0.0" in output


def test_report_without_a_ledger_fails_with_a_message(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    exit_code = main(["report", "--ledger", str(tmp_path / "missing.duckdb")])

    assert exit_code == 1
    assert "Nothing has been recorded" in caplog.text


def test_a_command_is_required() -> None:
    with pytest.raises(SystemExit) as exit_info:
        main([])

    assert exit_info.value.code != 0


def test_planting_fails_loudly_when_a_seed_has_no_open_high_priority_ticket() -> None:
    with pytest.raises(DatasetGenerationError, match="open high-priority"):
        _oldest_open_high_priority([])


def test_planting_fails_loudly_when_a_seed_has_no_resolved_billing_ticket() -> None:
    with pytest.raises(DatasetGenerationError, match="resolved billing"):
        _latest_resolved_billing([])
