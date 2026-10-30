"""A model that answers the golden tasks by following a fixed plan, for end-to-end tests.

It is not a stand-in for a real model's quality. It exists so the harness, the recordings, the
baseline and the gate can be exercised without the API, including with deliberate mistakes.
"""

from collections.abc import Collection

from observable_analytics_agent.evaluation.expected import compute_expected
from observable_analytics_agent.evaluation.golden_tasks import GOLDEN_TASKS
from observable_analytics_agent.evaluation.tasks import AnswerKind, GoldenTask
from observable_analytics_agent.models.base import ModelReply, ModelRequest
from observable_analytics_agent.models.scripted import (
    submit_answer_call,
    text_reply,
    tool_call,
    tool_use_reply,
)
from observable_analytics_agent.tools.sql_runner import ReadOnlyDatabase

WRONG_NUMBER_OFFSET = 1_000.0
WRONG_TEXT = "not the answer"


class OracleModel:
    def __init__(self, database: ReadOnlyDatabase, mistakes: Collection[str] = ()) -> None:
        self._database = database
        self._mistakes = set(mistakes)
        self._tasks = {task.question: task for task in GOLDEN_TASKS}

    def complete(self, request: ModelRequest) -> ModelReply:
        task = self._tasks[request.messages[0].content[0].get("text")]
        step = sum(message.role == "assistant" for message in request.messages)
        plan = self._plan(task)
        if step >= len(plan):
            return text_reply("Nothing more to do.")
        return plan[step]

    def _plan(self, task: GoldenTask) -> list[ModelReply]:
        if task.kind is AnswerKind.CANNOT_ANSWER:
            return [tool_use_reply(self._refusal(task))]
        plan = []
        if task.requires_glossary:
            plan.append(
                tool_use_reply(tool_call("search_business_glossary", {"query": task.question}))
            )
        plan.append(tool_use_reply(tool_call("run_read_only_query", {"sql": task.reference_sql})))
        plan.append(tool_use_reply(self._submission(task)))
        return plan

    def _refusal(self, task: GoldenTask):
        if task.task_id in self._mistakes:
            return submit_answer_call(0, "I did it.")
        return tool_call(
            "submit_answer",
            {"status": "cannot_answer", "explanation": "The database is read-only or lacks it."},
        )

    def _submission(self, task: GoldenTask):
        expected = compute_expected(task, self._database)
        value = {
            AnswerKind.NUMBER: expected.number,
            AnswerKind.RANKING: list(expected.labels),
        }.get(expected.kind, expected.text)
        if task.task_id in self._mistakes:
            value = _wrong(value)
        return submit_answer_call(value, "Computed with the reference query.")


def _wrong(value):
    if isinstance(value, float):
        return value + WRONG_NUMBER_OFFSET
    if isinstance(value, list):
        return list(reversed(value)) if len(value) > 1 else [WRONG_TEXT]
    return WRONG_TEXT
