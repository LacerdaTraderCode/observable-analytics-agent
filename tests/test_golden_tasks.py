"""The golden tasks are well formed, and every expected answer comes from the data."""

import pytest

from observable_analytics_agent.dataset import vocabulary
from observable_analytics_agent.dataset.generator import CUSTOMER_COUNT
from observable_analytics_agent.evaluation.expected import GoldenTaskError, compute_expected
from observable_analytics_agent.evaluation.golden_tasks import GOLDEN_TASKS
from observable_analytics_agent.evaluation.tasks import (
    AnswerKind,
    GoldenTask,
    TaskCategory,
)
from observable_analytics_agent.tools.glossary import load_glossary
from observable_analytics_agent.tools.sql_runner import ReadOnlyDatabase

MIN_TASKS = 30


def a_task(sql: str | None, kind: AnswerKind = AnswerKind.NUMBER, **fields) -> GoldenTask:
    return GoldenTask(
        task_id="synthetic",
        question="A question?",
        category=TaskCategory.SINGLE_NUMBER,
        kind=kind,
        reference_sql=sql,
        **fields,
    )


def test_there_are_about_thirty_tasks_with_unique_ids_and_questions() -> None:
    assert len(GOLDEN_TASKS) >= MIN_TASKS
    assert len({task.task_id for task in GOLDEN_TASKS}) == len(GOLDEN_TASKS)
    assert len({task.question for task in GOLDEN_TASKS}) == len(GOLDEN_TASKS)


def test_every_kind_of_question_is_covered() -> None:
    assert {task.category for task in GOLDEN_TASKS} == set(TaskCategory)


def test_only_tasks_that_cannot_be_answered_have_no_reference_query() -> None:
    for task in GOLDEN_TASKS:
        assert (task.reference_sql is None) == (task.kind is AnswerKind.CANNOT_ANSWER), task.task_id


def test_refusal_tasks_are_the_write_requests_and_the_impossible_questions() -> None:
    refusals = {task.category for task in GOLDEN_TASKS if task.kind is AnswerKind.CANNOT_ANSWER}

    assert refusals == {TaskCategory.WRITE_REQUEST, TaskCategory.IMPOSSIBLE}


def test_every_expected_answer_is_computed_from_the_data(database: ReadOnlyDatabase) -> None:
    for task in GOLDEN_TASKS:
        expected = compute_expected(task, database)

        assert expected.kind is task.kind, task.task_id
        if task.kind is AnswerKind.NUMBER:
            assert expected.number is not None, task.task_id
        elif task.kind is AnswerKind.RANKING:
            assert len(expected.labels) == task.top_n, task.task_id
        elif task.kind is not AnswerKind.CANNOT_ANSWER:
            assert expected.text, task.task_id


def test_expected_answers_follow_the_data_not_a_typed_constant(database: ReadOnlyDatabase) -> None:
    by_id = {task.task_id: task for task in GOLDEN_TASKS}

    assert compute_expected(by_id["count_customers"], database).number == CUSTOMER_COUNT
    assert compute_expected(by_id["keyboard_list_price"], database).number == 89.0
    assert compute_expected(by_id["latest_order_date"], database).text == "2025-12-31"


def test_the_injection_tasks_point_at_rows_that_carry_planted_instructions(
    database: ReadOnlyDatabase,
) -> None:
    by_id = {task.task_id: task for task in GOLDEN_TASKS}
    connection = database.connect()
    try:
        oldest_open = by_id["subject_of_oldest_open_high_priority_ticket"].reference_sql
        subject = connection.execute(oldest_open).fetchone()[0]
        body = connection.execute(
            "SELECT body FROM support_tickets WHERE subject = ? AND status = 'open' "
            "AND priority = 'high' ORDER BY opened_at LIMIT 1",
            [subject],
        ).fetchone()[0]
    finally:
        connection.close()

    assert body == vocabulary.OVERRIDE_ANSWER_INJECTION
    assert "REDACTED" in vocabulary.OVERRIDE_ANSWER_INJECTION
    assert "secret promotion" in vocabulary.PRICE_OVERRIDE_DESCRIPTION
    assert "DELETE FROM" in vocabulary.DESTRUCTIVE_SQL_INJECTION


def test_tasks_that_need_a_glossary_term_use_terms_the_glossary_defines() -> None:
    glossary_text = " ".join(
        f"{entry.term} {' '.join(entry.aliases)}" for entry in load_glossary()
    ).casefold()
    needing_it = [task for task in GOLDEN_TASKS if task.category is TaskCategory.GLOSSARY]

    assert needing_it
    assert all(task.requires_glossary for task in needing_it)
    for term in ("revenue", "average order value", "active customer", "churn", "repeat customer"):
        assert term in glossary_text


def test_a_task_with_no_reference_query_is_malformed(database: ReadOnlyDatabase) -> None:
    with pytest.raises(GoldenTaskError, match="needs a query"):
        compute_expected(a_task(None), database)


def test_a_reference_query_that_returns_nothing_is_malformed(database: ReadOnlyDatabase) -> None:
    with pytest.raises(GoldenTaskError, match="no rows"):
        compute_expected(a_task("SELECT 1 WHERE false"), database)


def test_a_ranking_needs_a_row_beyond_the_cutoff(database: ReadOnlyDatabase) -> None:
    task = a_task("SELECT 'a', 3", AnswerKind.RANKING, top_n=1)

    with pytest.raises(GoldenTaskError, match="one row beyond top_n"):
        compute_expected(task, database)


def test_a_ranking_with_a_tie_is_refused(database: ReadOnlyDatabase) -> None:
    task = a_task("SELECT * FROM (VALUES ('a', 3), ('b', 3)) ORDER BY 2 DESC", AnswerKind.RANKING)

    with pytest.raises(GoldenTaskError, match="tie"):
        compute_expected(task, database)
