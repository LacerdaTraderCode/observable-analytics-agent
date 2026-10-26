"""The ledger stores runs and steps in DuckDB and reads them back without loss."""

from pathlib import Path

import duckdb
import pytest

from observable_analytics_agent.agent.state import RunStatus
from observable_analytics_agent.observability.ledger import LedgerNotFoundError, RunLedger
from observable_analytics_agent.observability.records import StepOutcome
from tests.support import a_run_record


@pytest.fixture
def ledger(tmp_path: Path) -> RunLedger:
    return RunLedger(tmp_path / "nested" / "ledger.duckdb")


def test_a_run_reads_back_identically(ledger: RunLedger) -> None:
    record = a_run_record(task_id="task-1", tool_outcomes=[StepOutcome.OK, StepOutcome.ERROR])

    ledger.write(record)

    assert ledger.fetch_run("run-1") == record


def test_the_ledger_creates_its_directory(ledger: RunLedger) -> None:
    ledger.write(a_run_record())

    assert ledger.path.is_file()


def test_an_unknown_run_is_none(ledger: RunLedger) -> None:
    ledger.write(a_run_record("run-1"))

    assert ledger.fetch_run("other") is None


def test_a_missing_ledger_is_reported(tmp_path: Path) -> None:
    with pytest.raises(LedgerNotFoundError, match="Nothing has been recorded"):
        RunLedger(tmp_path / "none.duckdb").fetch_run("run-1")


def test_a_run_cannot_be_stored_twice(ledger: RunLedger) -> None:
    ledger.write(a_run_record())

    with pytest.raises(duckdb.ConstraintException):
        ledger.write(a_run_record())


def test_a_failed_write_leaves_no_partial_run(ledger: RunLedger) -> None:
    record = a_run_record()
    record.steps.append(record.steps[0])

    with pytest.raises(duckdb.ConstraintException):
        ledger.write(record)

    assert ledger.fetch_run("run-1") is None


def test_runs_can_be_queried_with_sql(ledger: RunLedger) -> None:
    ledger.write(a_run_record("run-1"))
    ledger.write(a_run_record("run-2", status=RunStatus.STEP_BUDGET_EXCEEDED))

    with ledger.read_connection() as connection:
        rows = connection.execute(
            "SELECT status, count(*) FROM runs GROUP BY 1 ORDER BY 1"
        ).fetchall()

    assert rows == [("completed", 1), ("step_budget_exceeded", 1)]


def test_steps_keep_the_model_and_tool_details(ledger: RunLedger) -> None:
    ledger.write(a_run_record(tool_outcomes=[StepOutcome.ERROR]))

    with ledger.read_connection() as connection:
        rows = connection.execute(
            "SELECT kind, name, outcome, error_code, input_tokens, prompt_version "
            "FROM steps ORDER BY step_index"
        ).fetchall()

    assert rows == [
        ("llm", "priced-model", "ok", None, 100, "1.0.0"),
        ("tool", "run_read_only_query", "error", "query_failed", None, None),
    ]
