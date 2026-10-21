"""Static checks that turn a model-written string into exactly one read-only statement.

This is the first of two layers. The second is the database itself, opened read-only with
external access disabled, so a gap in these checks still cannot write or read files.
"""

import re

import duckdb

from observable_analytics_agent.tools.errors import QueryFailed, QueryRejected, summarize_error

ALLOWED_LEADING_KEYWORDS = frozenset({"SELECT", "WITH"})

_LEADING_KEYWORD = re.compile(r"\s*(?:(?:--[^\n]*(?:\n|$)|/\*.*?\*/)\s*)*([A-Za-z_]+)", re.DOTALL)


def validate_read_only_query(sql: str) -> str:
    """Return the statement when `sql` is a single SELECT or WITH query, otherwise raise."""
    try:
        statements = duckdb.extract_statements(sql)
    except duckdb.Error as error:
        raise QueryFailed(summarize_error(error)) from error
    if not statements:
        raise QueryRejected("The query is empty.")
    if len(statements) > 1:
        raise QueryRejected("Only one statement is allowed per call.")
    statement = statements[0]
    if statement.type != duckdb.StatementType.SELECT:
        raise QueryRejected(f"Only SELECT queries are allowed, not {statement.type.name}.")
    keyword = _leading_keyword(statement.query)
    if keyword not in ALLOWED_LEADING_KEYWORDS:
        raise QueryRejected(f"A query must start with SELECT or WITH, not {keyword or 'nothing'}.")
    return statement.query


def _leading_keyword(statement_text: str) -> str:
    match = _LEADING_KEYWORD.match(statement_text)
    return match.group(1).upper() if match else ""
