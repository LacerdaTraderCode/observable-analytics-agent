"""Hardened, read-only DuckDB access: the connection settings and the guarded query runner."""

import contextlib
import math
import threading
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb

from observable_analytics_agent.tools.errors import (
    QueryFailed,
    QueryRejected,
    QueryTimedOut,
    summarize_error,
)
from observable_analytics_agent.tools.sql_guard import validate_read_only_query

DEFAULT_ROW_LIMIT = 50
DEFAULT_TIMEOUT_SECONDS = 5.0
DEFAULT_MEMORY_LIMIT = "512MB"
DEFAULT_MAX_CELL_CHARS = 400
TRUNCATION_MARKER = "...[truncated]"

# Order matters in the DuckDB docs: external access is switched off before the lock.
HARDENED_SETTINGS = {
    "enable_external_access": "false",
    "autoinstall_known_extensions": "false",
    "autoload_known_extensions": "false",
    "threads": "1",
}


@dataclass(frozen=True)
class SqlLimits:
    row_limit: int = DEFAULT_ROW_LIMIT
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    memory_limit: str = DEFAULT_MEMORY_LIMIT
    max_cell_chars: int = DEFAULT_MAX_CELL_CHARS


@dataclass(frozen=True)
class QueryResult:
    columns: list[str]
    rows: list[list[Any]]
    truncated: bool


class DatasetNotFoundError(FileNotFoundError):
    """The dataset file does not exist; generate it before querying."""


class ReadOnlyDatabase:
    """Opens the dataset read-only, with external access disabled and the configuration locked."""

    def __init__(self, database_path: Path, limits: SqlLimits | None = None) -> None:
        if not database_path.is_file():
            raise DatasetNotFoundError(f"No dataset at {database_path}. Generate it first.")
        self._database_path = database_path
        self.limits = limits or SqlLimits()

    def connect(self) -> duckdb.DuckDBPyConnection:
        settings = {**HARDENED_SETTINGS, "memory_limit": self.limits.memory_limit}
        connection = duckdb.connect(str(self._database_path), read_only=True, config=settings)
        connection.execute("SET lock_configuration = true")
        return connection


class SqlRunner:
    """Runs one validated query with a row limit and a timeout."""

    def __init__(self, database: ReadOnlyDatabase) -> None:
        self._database = database

    def run(self, sql: str) -> QueryResult:
        statement = validate_read_only_query(sql)
        limits = self._database.limits
        connection = self._database.connect()
        timer = threading.Timer(limits.timeout_seconds, _interrupt, args=(connection,))
        timer.start()
        try:
            cursor = connection.execute(statement)
            columns = [column[0] for column in cursor.description]
            fetched = cursor.fetchmany(limits.row_limit + 1)
        except duckdb.InterruptException as error:
            raise QueryTimedOut(
                f"The query ran longer than {limits.timeout_seconds:g}s."
            ) from error
        except duckdb.PermissionException as error:
            raise QueryRejected(f"Blocked by the sandbox: {summarize_error(error)}") from error
        except duckdb.Error as error:
            raise QueryFailed(summarize_error(error)) from error
        finally:
            timer.cancel()
            connection.close()
        kept = fetched[: limits.row_limit]
        rows = [[_to_json_value(cell, limits.max_cell_chars) for cell in row] for row in kept]
        return QueryResult(columns=columns, rows=rows, truncated=len(fetched) > limits.row_limit)


def _interrupt(connection: duckdb.DuckDBPyConnection) -> None:
    with contextlib.suppress(duckdb.ConnectionException):
        connection.interrupt()


def _to_json_value(value: Any, max_chars: int) -> Any:
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime | date | time):
        return value.isoformat()
    if isinstance(value, list | tuple):
        return [_to_json_value(item, max_chars) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_json_value(item, max_chars) for key, item in value.items()}
    text = value if isinstance(value, str) else str(value)
    return text if len(text) <= max_chars else text[:max_chars] + TRUNCATION_MARKER
