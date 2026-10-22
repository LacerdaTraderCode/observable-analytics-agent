"""The runner returns JSON-safe rows within the configured limits."""

from pathlib import Path

import pytest

from observable_analytics_agent.tools.errors import QueryFailed, QueryTimedOut
from observable_analytics_agent.tools.sql_runner import (
    TRUNCATION_MARKER,
    DatasetNotFoundError,
    ReadOnlyDatabase,
    SqlLimits,
    SqlRunner,
)

SHORT_TIMEOUT_SECONDS = 0.3


def runner_with(dataset_path: Path, **limits: float | int | str) -> SqlRunner:
    return SqlRunner(ReadOnlyDatabase(dataset_path, SqlLimits(**limits)))


def test_returns_columns_and_rows(dataset_path: Path) -> None:
    outcome = runner_with(dataset_path).run(
        "SELECT status, count(*) AS orders FROM orders GROUP BY status ORDER BY status"
    )

    assert outcome.columns == ["status", "orders"]
    assert [row[0] for row in outcome.rows] == ["cancelled", "completed", "refunded"]
    assert not outcome.truncated


def test_truncates_to_the_row_limit(dataset_path: Path) -> None:
    outcome = runner_with(dataset_path, row_limit=5).run(
        "SELECT customer_id FROM customers ORDER BY customer_id"
    )

    assert [row[0] for row in outcome.rows] == [1, 2, 3, 4, 5]
    assert outcome.truncated


def test_a_result_that_fits_the_limit_is_not_flagged(dataset_path: Path) -> None:
    outcome = runner_with(dataset_path, row_limit=3).run("SELECT * FROM range(3)")

    assert len(outcome.rows) == 3
    assert not outcome.truncated


def test_truncates_long_text_cells(dataset_path: Path) -> None:
    outcome = runner_with(dataset_path, max_cell_chars=20).run("SELECT repeat('x', 100)")

    assert outcome.rows == [["x" * 20 + TRUNCATION_MARKER]]


def test_converts_database_types_to_json_values(dataset_path: Path) -> None:
    outcome = runner_with(dataset_path).run(
        "SELECT 12.50::DECIMAL(10, 2), DATE '2025-01-31', TIMESTAMP '2025-01-31 08:30:00', "
        "[1, 2], NULL"
    )

    assert outcome.rows == [[12.5, "2025-01-31", "2025-01-31T08:30:00", [1, 2], None]]


def test_non_finite_numbers_become_text(dataset_path: Path) -> None:
    outcome = runner_with(dataset_path).run("SELECT 'nan'::DOUBLE")

    assert outcome.rows == [["nan"]]


def test_a_binder_error_names_the_columns_that_exist(dataset_path: Path) -> None:
    with pytest.raises(QueryFailed, match="ordered_at"):
        runner_with(dataset_path).run("SELECT order_date FROM orders")


def test_stops_a_query_that_exceeds_the_timeout(dataset_path: Path) -> None:
    runner = runner_with(dataset_path, timeout_seconds=SHORT_TIMEOUT_SECONDS)

    with pytest.raises(QueryTimedOut):
        runner.run("SELECT count(*) FROM range(10000000000) a, range(10) b")


def test_a_missing_dataset_is_reported(tmp_path: Path) -> None:
    with pytest.raises(DatasetNotFoundError):
        ReadOnlyDatabase(tmp_path / "missing.duckdb")
