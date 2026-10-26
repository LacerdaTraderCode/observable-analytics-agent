"""The DuckDB run ledger: every run and every step, queryable with SQL.

Connections are opened per operation and closed right after. DuckDB allows one writer process
per file, so holding the file open would block a report run from another process.
"""

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import duckdb

from observable_analytics_agent.agent.state import RunStatus
from observable_analytics_agent.models.base import TokenUsage
from observable_analytics_agent.observability.records import (
    PromptStamp,
    RunRecord,
    StepKind,
    StepOutcome,
    StepRecord,
)

DEFAULT_LEDGER_PATH = Path("var/ledger.duckdb")
USAGE_COLUMNS = tuple(TokenUsage.model_fields)

SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS runs (
        run_id VARCHAR PRIMARY KEY,
        task_id VARCHAR,
        started_at TIMESTAMP NOT NULL,
        question VARCHAR NOT NULL,
        model VARCHAR NOT NULL,
        prompt_name VARCHAR,
        prompt_version VARCHAR,
        prompt_hash VARCHAR,
        status VARCHAR NOT NULL,
        answer_status VARCHAR,
        answer_value VARCHAR,
        error VARCHAR,
        step_count INTEGER NOT NULL,
        input_tokens BIGINT NOT NULL,
        output_tokens BIGINT NOT NULL,
        cache_creation_input_tokens BIGINT NOT NULL,
        cache_creation_1h_input_tokens BIGINT NOT NULL,
        cache_read_input_tokens BIGINT NOT NULL,
        cost_usd DOUBLE,
        latency_seconds DOUBLE NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS steps (
        run_id VARCHAR NOT NULL,
        step_index INTEGER NOT NULL,
        kind VARCHAR NOT NULL,
        name VARCHAR NOT NULL,
        started_offset_seconds DOUBLE NOT NULL,
        duration_seconds DOUBLE NOT NULL,
        outcome VARCHAR NOT NULL,
        error_code VARCHAR,
        arguments VARCHAR,
        finish_reason VARCHAR,
        prompt_name VARCHAR,
        prompt_version VARCHAR,
        prompt_hash VARCHAR,
        input_tokens BIGINT,
        output_tokens BIGINT,
        cache_creation_input_tokens BIGINT,
        cache_creation_1h_input_tokens BIGINT,
        cache_read_input_tokens BIGINT,
        cost_usd DOUBLE,
        PRIMARY KEY (run_id, step_index)
    )
    """,
)


class LedgerNotFoundError(FileNotFoundError):
    """No ledger file exists yet: nothing has been recorded."""


class RunLedger:
    def __init__(self, path: Path = DEFAULT_LEDGER_PATH) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def write(self, record: RunRecord) -> None:
        """Store a run and its steps in one transaction, creating the tables when needed."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        connection = duckdb.connect(str(self._path))
        try:
            for statement in SCHEMA_STATEMENTS:
                connection.execute(statement)
            connection.begin()
            _insert(connection, "runs", _run_row(record))
            for step in record.steps:
                _insert(connection, "steps", _step_row(record.run_id, step))
            connection.commit()
        finally:
            connection.close()

    def fetch_run(self, run_id: str) -> RunRecord | None:
        with self.read_connection() as connection:
            runs = _fetch_rows(connection, "SELECT * FROM runs WHERE run_id = ?", [run_id])
            if not runs:
                return None
            steps = _fetch_rows(
                connection, "SELECT * FROM steps WHERE run_id = ? ORDER BY step_index", [run_id]
            )
        return _to_record(runs[0], steps)

    @contextmanager
    def read_connection(self) -> Iterator[duckdb.DuckDBPyConnection]:
        if not self._path.is_file():
            raise LedgerNotFoundError(f"No ledger at {self._path}. Nothing has been recorded yet.")
        connection = duckdb.connect(str(self._path), read_only=True)
        try:
            yield connection
        finally:
            connection.close()


def _insert(connection: duckdb.DuckDBPyConnection, table: str, row: Mapping[str, Any]) -> None:
    placeholders = ", ".join("?" for _ in row)
    connection.execute(
        f"INSERT INTO {table} ({', '.join(row)}) VALUES ({placeholders})", list(row.values())
    )


def _prompt_columns(prompt: PromptStamp | None) -> dict[str, str | None]:
    return {
        "prompt_name": prompt.name if prompt else None,
        "prompt_version": prompt.version if prompt else None,
        "prompt_hash": prompt.content_hash if prompt else None,
    }


def _run_row(record: RunRecord) -> dict[str, Any]:
    return {
        "run_id": record.run_id,
        "task_id": record.task_id,
        "started_at": record.started_at,
        "question": record.question,
        "model": record.model,
        **_prompt_columns(record.prompt),
        "status": record.status.value,
        "answer_status": record.answer_status,
        "answer_value": record.answer_value,
        "error": record.error,
        "step_count": len(record.steps),
        **record.usage.model_dump(),
        "cost_usd": record.cost_usd,
        "latency_seconds": record.latency_seconds,
    }


def _step_row(run_id: str, step: StepRecord) -> dict[str, Any]:
    usage = step.usage.model_dump() if step.usage else dict.fromkeys(USAGE_COLUMNS)
    return {
        "run_id": run_id,
        "step_index": step.step_index,
        "kind": step.kind.value,
        "name": step.name,
        "started_offset_seconds": step.started_offset_seconds,
        "duration_seconds": step.duration_seconds,
        "outcome": step.outcome.value,
        "error_code": step.error_code,
        "arguments": step.arguments,
        "finish_reason": step.finish_reason,
        **_prompt_columns(step.prompt),
        **usage,
        "cost_usd": step.cost_usd,
    }


def _fetch_rows(
    connection: duckdb.DuckDBPyConnection, sql: str, parameters: Sequence[Any]
) -> list[dict[str, Any]]:
    cursor = connection.execute(sql, list(parameters))
    names = [column[0] for column in cursor.description]
    return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]


def _stamp_from(row: Mapping[str, Any]) -> PromptStamp | None:
    if row["prompt_name"] is None:
        return None
    return PromptStamp(
        name=row["prompt_name"], version=row["prompt_version"], content_hash=row["prompt_hash"]
    )


def _to_step(row: Mapping[str, Any]) -> StepRecord:
    has_usage = row["input_tokens"] is not None
    return StepRecord(
        step_index=row["step_index"],
        kind=StepKind(row["kind"]),
        name=row["name"],
        started_offset_seconds=row["started_offset_seconds"],
        duration_seconds=row["duration_seconds"],
        outcome=StepOutcome(row["outcome"]),
        error_code=row["error_code"],
        arguments=row["arguments"],
        finish_reason=row["finish_reason"],
        usage=TokenUsage(**{name: row[name] for name in USAGE_COLUMNS}) if has_usage else None,
        cost_usd=row["cost_usd"],
        prompt=_stamp_from(row),
    )


def _to_record(run: Mapping[str, Any], steps: Sequence[Mapping[str, Any]]) -> RunRecord:
    return RunRecord(
        run_id=run["run_id"],
        task_id=run["task_id"],
        started_at=run["started_at"],
        question=run["question"],
        model=run["model"],
        prompt=_stamp_from(run),
        status=RunStatus(run["status"]),
        answer_status=run["answer_status"],
        answer_value=run["answer_value"],
        error=run["error"],
        steps=[_to_step(step) for step in steps],
        usage=TokenUsage(**{name: run[name] for name in USAGE_COLUMNS}),
        cost_usd=run["cost_usd"],
        latency_seconds=run["latency_seconds"],
    )
