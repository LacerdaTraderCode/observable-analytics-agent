"""Static checks that turn a model-written string into exactly one read-only statement.

This is the first of two layers. The second is the database itself, opened read-only with
external access disabled, so a gap in these checks still cannot write or read files.
"""

import re

import duckdb

from observable_analytics_agent.tools.errors import QueryFailed, QueryRejected, summarize_error

ALLOWED_LEADING_KEYWORDS = frozenset({"SELECT", "WITH"})

_KEYWORD = re.compile(r"[A-Za-z_]+")


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
    keyword = leading_keyword(sql)
    if keyword not in ALLOWED_LEADING_KEYWORDS:
        found = keyword or sql.strip()[:1]
        raise QueryRejected(f"A query must start with SELECT or WITH, found '{found}'.")
    statement = statements[0]
    if statement.type != duckdb.StatementType.SELECT:
        raise QueryRejected(f"Only SELECT queries are allowed, not {statement.type.name}.")
    return statement.query


def leading_keyword(statement_text: str) -> str:
    """Return the first keyword after whitespace and comments, or an empty string.

    A scanner is used instead of one regular expression: comment patterns with nested
    quantifiers backtrack exponentially on crafted input, and this text comes from a model.
    """
    match = _KEYWORD.match(statement_text, _skip_whitespace_and_comments(statement_text))
    return match.group().upper() if match else ""


def _skip_whitespace_and_comments(text: str) -> int:
    position = 0
    while position < len(text):
        if text[position].isspace():
            position += 1
        elif text.startswith("--", position):
            line_end = text.find("\n", position)
            position = len(text) if line_end == -1 else line_end + 1
        elif text.startswith("/*", position):
            comment_end = text.find("*/", position + 2)
            if comment_end == -1:
                return len(text)
            position = comment_end + 2
        else:
            break
    return position
