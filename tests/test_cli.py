"""The command line generates the dataset and reports what it wrote."""

import logging
import runpy
import shutil
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


def test_prompts_list_shows_each_version_with_a_short_hash(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["prompts", "list"])

    lines = capsys.readouterr().out.splitlines()
    assert exit_code == 0
    assert [line.split()[:2] for line in lines] == [
        ["analyst_system", "1.0.0"],
        ["analyst_system", "2.0.0"],
    ]
    assert all(len(line.split()[2]) == 12 for line in lines)


def test_prompts_verify_accepts_the_shipped_prompts(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["prompts", "verify"])

    assert exit_code == 0
    assert "2 prompt versions verified." in capsys.readouterr().out


def test_prompts_verify_fails_when_a_prompt_was_edited_in_place(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    copied = tmp_path / "prompts"
    shutil.copytree(DEFAULT_PROMPTS_DIR, copied)
    baseline = copied / "analyst_system" / "1.0.0.md"
    baseline.write_text("An edited prompt.\n", encoding="utf-8")

    exit_code = main(["prompts", "verify", "--directory", str(copied)])

    assert exit_code == 1
    assert "no longer matches its declared hash" in caplog.text


def test_prompts_list_fails_without_a_manifest(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    exit_code = main(["prompts", "list", "--directory", str(tmp_path)])

    assert exit_code == 1
    assert "manifest" in caplog.text


def test_prompts_hash_prints_the_hash_to_declare(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    new_prompt = tmp_path / "3.0.0.md"
    new_prompt.write_text("A new version.\n", encoding="utf-8")

    exit_code = main(["prompts", "hash", str(new_prompt)])

    assert exit_code == 0
    assert capsys.readouterr().out.strip() == hash_content(b"A new version.\n")


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
