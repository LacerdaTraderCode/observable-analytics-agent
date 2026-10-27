"""Computes the expected answer of a golden task by running its reference query."""

from typing import Any

from observable_analytics_agent.evaluation.tasks import AnswerKind, ExpectedAnswer, GoldenTask
from observable_analytics_agent.tools.sql_runner import ReadOnlyDatabase


class GoldenTaskError(Exception):
    """A golden task is malformed or ambiguous. Fix the task, never the thresholds."""


def compute_expected(task: GoldenTask, database: ReadOnlyDatabase) -> ExpectedAnswer:
    if task.kind is AnswerKind.CANNOT_ANSWER:
        return ExpectedAnswer(kind=task.kind)
    if task.reference_sql is None:
        raise GoldenTaskError(f"{task.task_id}: a task that expects an answer needs a query.")
    rows = _run_reference_query(database, task.reference_sql)
    if not rows:
        raise GoldenTaskError(f"{task.task_id}: the reference query returned no rows.")
    if task.kind is AnswerKind.NUMBER:
        return ExpectedAnswer(kind=task.kind, number=float(rows[0][0]))
    if task.kind is AnswerKind.RANKING:
        return ExpectedAnswer(kind=task.kind, labels=_ranked_labels(task, rows))
    return ExpectedAnswer(kind=task.kind, text=str(rows[0][0]))


def _run_reference_query(database: ReadOnlyDatabase, sql: str) -> list[tuple[Any, ...]]:
    connection = database.connect()
    try:
        return connection.execute(sql).fetchall()
    finally:
        connection.close()


def _ranked_labels(task: GoldenTask, rows: list[tuple[Any, ...]]) -> tuple[str, ...]:
    """Return the leading labels, refusing a ranking whose order or cutoff is a tie."""
    if len(rows) <= task.top_n:
        raise GoldenTaskError(
            f"{task.task_id}: a ranking query must return one row beyond top_n to expose ties."
        )
    metrics = [row[1] for row in rows[: task.top_n + 1]]
    if len(set(metrics)) != len(metrics):
        raise GoldenTaskError(f"{task.task_id}: the ranking has a tie, so its order is ambiguous.")
    return tuple(str(row[0]) for row in rows[: task.top_n])
