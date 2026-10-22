"""Attack attempts against the SQL tool: none may write, read files or loosen the sandbox."""

import json
from pathlib import Path

import duckdb
import pytest

from observable_analytics_agent.dataset import vocabulary
from observable_analytics_agent.dataset.generator import summarize_database
from observable_analytics_agent.tools.base import ToolRegistry
from observable_analytics_agent.tools.sql_runner import ReadOnlyDatabase

ATTACKS = {
    "multiple statements": "SELECT 1; DROP TABLE orders",
    "multiple statements after a comment": "SELECT 1; -- harmless\nDELETE FROM orders",
    "COPY to a file": "COPY orders TO '/tmp/stolen_orders.csv'",
    "ATTACH another database": "ATTACH '/tmp/other.duckdb' AS other",
    "INSTALL an extension": "INSTALL httpfs",
    "LOAD an extension": "LOAD httpfs",
    "change a setting": "SET enable_external_access = true",
    "unlock the configuration": "SET lock_configuration = false",
    "PRAGMA": "PRAGMA database_list",
    "INSERT": "INSERT INTO customers VALUES (0, 'x', 'x', 'x', 'x', DATE '2025-01-01', NULL)",
    "UPDATE": "UPDATE orders SET status = 'completed'",
    "DELETE": "DELETE FROM orders",
    "DROP": "DROP TABLE orders",
    "CREATE TABLE AS": "CREATE TABLE stolen AS SELECT * FROM customers",
    "EXPORT DATABASE": "EXPORT DATABASE '/tmp/export'",
    "writing CTE": "WITH moved AS (SELECT 1) INSERT INTO orders SELECT * FROM moved",
    "read a file with read_csv": "SELECT * FROM read_csv('/etc/passwd')",
    "read a file with read_text": "SELECT * FROM read_text('/etc/hostname')",
    "list files with glob": "SELECT * FROM glob('/etc/*')",
    "read a parquet file": "SELECT * FROM read_parquet('/etc/passwd')",
    "fetch a URL": "SELECT * FROM read_csv('https://example.com/data.csv')",
}

WITHOUT_THE_GUARD = {
    "INSERT": "INSERT INTO customers VALUES (0, 'x', 'x', 'x', 'x', DATE '2025-01-01', NULL)",
    "CREATE TABLE": "CREATE TABLE stolen (id INTEGER)",
    "DELETE": "DELETE FROM orders",
    "COPY to a file": "COPY orders TO '/tmp/stolen_orders.csv'",
    "ATTACH": "ATTACH '/tmp/other.duckdb' AS other",
    "INSTALL": "INSTALL httpfs",
    "LOAD": "LOAD httpfs",
    "enable external access": "SET enable_external_access = true",
    "unlock the configuration": "SET lock_configuration = false",
    "raise the memory limit": "SET memory_limit = '64GB'",
    "read a file": "SELECT * FROM read_csv('/etc/passwd')",
    "EXPORT DATABASE": "EXPORT DATABASE '/tmp/export'",
}


@pytest.mark.parametrize("sql", ATTACKS.values(), ids=ATTACKS.keys())
def test_the_tool_rejects_the_attack(tool_registry: ToolRegistry, sql: str) -> None:
    outcome = tool_registry.execute("run_read_only_query", {"sql": sql})

    assert outcome.is_error
    assert outcome.error_code == "query_rejected"


def test_the_dataset_is_unchanged_after_every_attack(
    tool_registry: ToolRegistry, dataset_path: Path
) -> None:
    before = summarize_database(dataset_path)

    for sql in ATTACKS.values():
        tool_registry.execute("run_read_only_query", {"sql": sql})

    assert summarize_database(dataset_path) == before


@pytest.mark.parametrize("sql", WITHOUT_THE_GUARD.values(), ids=WITHOUT_THE_GUARD.keys())
def test_the_database_blocks_the_statement_even_without_the_guard(
    database: ReadOnlyDatabase, sql: str
) -> None:
    connection = database.connect()
    try:
        with pytest.raises(duckdb.Error):
            connection.execute(sql)
    finally:
        connection.close()


def test_the_connection_is_read_only_with_a_locked_configuration(
    database: ReadOnlyDatabase,
) -> None:
    connection = database.connect()
    try:
        settings = dict(
            connection.execute(
                "SELECT name, value FROM duckdb_settings() "
                "WHERE name IN ('access_mode', 'enable_external_access', 'lock_configuration')"
            ).fetchall()
        )
    finally:
        connection.close()

    assert settings == {
        "access_mode": "read_only",
        "enable_external_access": "false",
        "lock_configuration": "true",
    }


def test_instructions_stored_in_the_data_come_back_as_plain_text(
    tool_registry: ToolRegistry,
) -> None:
    outcome = tool_registry.execute(
        "run_read_only_query",
        {"sql": "SELECT body FROM support_tickets WHERE body LIKE '%REDACTED%'"},
    )

    assert not outcome.is_error
    assert json.loads(outcome.content)["rows"] == [[vocabulary.OVERRIDE_ANSWER_INJECTION]]
