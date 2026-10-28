"""Runs the golden tasks through an observed agent and scores each run."""

from collections.abc import Sequence

from observable_analytics_agent.evaluation.expected import compute_expected
from observable_analytics_agent.evaluation.results import TaskResult
from observable_analytics_agent.evaluation.scoring import score_task
from observable_analytics_agent.evaluation.tasks import GoldenTask
from observable_analytics_agent.observability.observed_agent import ObservedAnalystAgent
from observable_analytics_agent.tools.sql_runner import ReadOnlyDatabase


class EvaluationHarness:
    def __init__(self, agent: ObservedAnalystAgent, database: ReadOnlyDatabase) -> None:
        self._agent = agent
        self._database = database

    def run(self, tasks: Sequence[GoldenTask]) -> list[TaskResult]:
        """Run every task. The expected answers are computed first, so a malformed task fails
        before any model call is made or paid for."""
        expectations = [(task, compute_expected(task, self._database)) for task in tasks]
        results = []
        for task, expected in expectations:
            observed = self._agent.run(task.question, task_id=task.task_id)
            score = score_task(task, expected, observed.outcome, observed.record)
            results.append(TaskResult(task, expected, observed.outcome, observed.record, score))
        return results
