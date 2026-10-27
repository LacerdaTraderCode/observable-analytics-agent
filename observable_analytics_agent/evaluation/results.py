"""What the harness keeps for every task it ran."""

from dataclasses import dataclass

from observable_analytics_agent.agent.state import RunOutcome
from observable_analytics_agent.evaluation.scoring import TaskScore
from observable_analytics_agent.evaluation.tasks import ExpectedAnswer, GoldenTask
from observable_analytics_agent.observability.records import RunRecord


@dataclass(frozen=True)
class TaskResult:
    task: GoldenTask
    expected: ExpectedAnswer
    outcome: RunOutcome
    record: RunRecord
    score: TaskScore

    @property
    def latency_seconds(self) -> float:
        """Time spent in model and tool calls. A replayed model call counts its recorded time."""
        return sum(step.duration_seconds for step in self.record.steps)
