"""The static guard accepts one SELECT or WITH statement and rejects everything else."""

import time

import pytest

from observable_analytics_agent.tools.errors import QueryFailed, QueryRejected
from observable_analytics_agent.tools.sql_guard import leading_keyword, validate_read_only_query

MAX_SCAN_SECONDS = 2.0

ACCEPTED_QUERIES = {
    "plain select": "SELECT 1",
    "lowercase": "select count(*) from orders",
    "with clause": "WITH recent AS (SELECT * FROM orders) SELECT count(*) FROM recent",
    "leading line comment": "-- total orders\nSELECT count(*) FROM orders",
    "leading block comment": "/* total */ SELECT 1",
    "trailing semicolon": "SELECT 1;",
}

REJECTED_QUERIES = {
    "two statements": "SELECT 1; SELECT 2",
    "stacked drop": "SELECT 1; DROP TABLE orders",
    "comment hiding a second statement": "SELECT 1; -- harmless\nDELETE FROM orders",
    "copy": "COPY orders TO '/tmp/orders.csv'",
    "attach": "ATTACH '/tmp/other.duckdb' AS other",
    "install": "INSTALL httpfs",
    "load": "LOAD httpfs",
    "set": "SET enable_external_access = true",
    "unlock the configuration": "SET lock_configuration = false",
    "pragma": "PRAGMA database_list",
    "insert": "INSERT INTO customers VALUES (1)",
    "update": "UPDATE orders SET status = 'completed'",
    "delete": "DELETE FROM orders",
    "drop": "DROP TABLE orders",
    "create table as": "CREATE TABLE copy_of_orders AS SELECT * FROM orders",
    "export": "EXPORT DATABASE '/tmp/export'",
    "call": "CALL pragma_version()",
    "explain": "EXPLAIN SELECT 1",
    "describe": "DESCRIBE orders",
    "show": "SHOW TABLES",
    "empty": "",
    "only a comment": "-- nothing to run",
    "writing cte": "WITH moved AS (SELECT 1) INSERT INTO orders SELECT * FROM moved",
}


@pytest.mark.parametrize("sql", ACCEPTED_QUERIES.values(), ids=ACCEPTED_QUERIES.keys())
def test_accepts_a_single_select_or_with(sql: str) -> None:
    assert validate_read_only_query(sql)


@pytest.mark.parametrize("sql", REJECTED_QUERIES.values(), ids=REJECTED_QUERIES.keys())
def test_rejects_anything_else(sql: str) -> None:
    with pytest.raises(QueryRejected):
        validate_read_only_query(sql)


def test_reports_a_syntax_error_as_a_failed_query() -> None:
    with pytest.raises(QueryFailed, match="syntax error"):
        validate_read_only_query("SELEC 1")


def test_names_the_offending_keyword() -> None:
    with pytest.raises(QueryRejected, match="PRAGMA"):
        validate_read_only_query("PRAGMA database_list")


@pytest.mark.parametrize(
    ("text", "keyword"),
    [
        ("select 1", "SELECT"),
        ("  \n\t WITH x AS (SELECT 1) SELECT * FROM x", "WITH"),
        ("-- note\nSELECT 1", "SELECT"),
        ("/* a */ /* b */ -- c\n  SELECT 1", "SELECT"),
        ("-- only a comment", ""),
        ("/* never closed SELECT 1", ""),
        ("", ""),
        ("(SELECT 1)", ""),
    ],
)
def test_the_leading_keyword_skips_whitespace_and_comments(text: str, keyword: str) -> None:
    assert leading_keyword(text) == keyword


PATHOLOGICAL_TEXTS = {
    "overlapping comment markers": "/*" + "*//*" * 20_000,
    "many empty block comments": "/**/" * 20_000 + "SELECT 1",
    "many line comments": "--\n" * 20_000 + "SELECT 1",
    "long whitespace": " " * 100_000 + "SELECT 1",
    "unterminated comments": "/*" * 50_000,
}


@pytest.mark.parametrize("text", PATHOLOGICAL_TEXTS.values(), ids=PATHOLOGICAL_TEXTS.keys())
def test_the_keyword_scan_stays_linear_on_pathological_text(text: str) -> None:
    started = time.monotonic()

    leading_keyword(text)

    assert time.monotonic() - started < MAX_SCAN_SECONDS


def test_a_nested_comment_cannot_make_a_write_look_like_a_select() -> None:
    with pytest.raises((QueryRejected, QueryFailed)):
        validate_read_only_query("/* outer /* inner */ SELECT 1 */ DROP TABLE orders")
