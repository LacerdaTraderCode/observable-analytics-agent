"""Test doubles shared by several test modules."""

from collections.abc import Sequence
from datetime import date, datetime

from observable_analytics_agent.agent.state import RunOutcome, RunStatus
from observable_analytics_agent.evaluation.results import TaskResult
from observable_analytics_agent.evaluation.scoring import TaskScore
from observable_analytics_agent.evaluation.tasks import (
    AnswerKind,
    ExpectedAnswer,
    GoldenTask,
    TaskCategory,
)
from observable_analytics_agent.models.base import ChatModel, ModelReply, ModelRequest, TokenUsage
from observable_analytics_agent.observability.pricing import ModelPricing, PriceTier, PricingTable
from observable_analytics_agent.observability.records import (
    PromptStamp,
    RunRecord,
    StepKind,
    StepOutcome,
    StepRecord,
)

PRICED_MODEL = "priced-model"
FAKE_INPUT_PRICE = 1.0
FAKE_OUTPUT_PRICE = 2.0
FAKE_CACHE_WRITE_5M_PRICE = 1.25
FAKE_CACHE_WRITE_1H_PRICE = 2.0
FAKE_CACHE_READ_PRICE = 0.1


class FakeClock:
    def __init__(self) -> None:
        self.seconds = 0.0

    def __call__(self) -> float:
        return self.seconds

    def advance(self, seconds: float) -> None:
        self.seconds += seconds


class ClockAdvancingModel:
    """Makes every call to the wrapped model take `seconds` on the fake clock."""

    def __init__(self, inner: ChatModel, clock: FakeClock, seconds: float) -> None:
        self._inner = inner
        self._clock = clock
        self._seconds = seconds

    def complete(self, request: ModelRequest) -> ModelReply:
        self._clock.advance(self._seconds)
        return self._inner.complete(request)


def fake_pricing() -> PricingTable:
    tier = PriceTier(
        max_prompt_tokens=None,
        input=FAKE_INPUT_PRICE,
        output=FAKE_OUTPUT_PRICE,
        cache_write_5m=FAKE_CACHE_WRITE_5M_PRICE,
        cache_write_1h=FAKE_CACHE_WRITE_1H_PRICE,
        cache_read=FAKE_CACHE_READ_PRICE,
    )
    return PricingTable(
        as_of=date(2026, 1, 1),
        source="test fixture",
        currency="USD",
        models={PRICED_MODEL: ModelPricing(tiers=[tier])},
    )


def a_run_record(
    run_id: str = "run-1",
    *,
    prompt_version: str = "1.0.0",
    status: RunStatus = RunStatus.COMPLETED,
    cost_usd: float | None = 0.001,
    latency_seconds: float = 1.0,
    task_id: str | None = None,
    tool_outcomes: Sequence[StepOutcome] = (StepOutcome.OK,),
) -> RunRecord:
    prompt = PromptStamp(
        name="analyst_system", version=prompt_version, content_hash=f"hash-{prompt_version}"
    )
    usage = TokenUsage(input_tokens=100, output_tokens=20)
    model_step = StepRecord(
        step_index=0,
        kind=StepKind.LLM,
        name=PRICED_MODEL,
        started_offset_seconds=0.0,
        duration_seconds=0.5,
        outcome=StepOutcome.OK,
        finish_reason="tool_use",
        usage=usage,
        cost_usd=cost_usd,
        prompt=prompt,
    )
    tool_steps = [
        StepRecord(
            step_index=index + 1,
            kind=StepKind.TOOL,
            name="run_read_only_query",
            started_offset_seconds=0.5,
            duration_seconds=0.01,
            outcome=outcome,
            error_code="query_failed" if outcome is StepOutcome.ERROR else None,
            arguments='{"sql":"SELECT 1"}',
        )
        for index, outcome in enumerate(tool_outcomes)
    ]
    completed = status is RunStatus.COMPLETED
    return RunRecord(
        run_id=run_id,
        task_id=task_id,
        started_at=datetime(2026, 10, 1, 12, 0, 0),
        question="How many orders?",
        model=PRICED_MODEL,
        prompt=prompt,
        status=status,
        answer_status="answered" if completed else None,
        answer_value="42.0" if completed else None,
        steps=[model_step, *tool_steps],
        usage=usage,
        cost_usd=cost_usd,
        latency_seconds=latency_seconds,
    )


def a_task_result(
    task_id: str = "task",
    *,
    passed: bool = True,
    cost_usd: float | None = 0.001,
    step_durations: Sequence[float] = (1.0,),
    model_calls: int = 3,
    category: TaskCategory = TaskCategory.SINGLE_NUMBER,
    requires_glossary: bool = False,
    used_glossary: bool = False,
    status: RunStatus = RunStatus.COMPLETED,
    error: str | None = None,
) -> TaskResult:
    task = GoldenTask(
        task_id=task_id,
        question=f"Question {task_id}?",
        category=category,
        kind=AnswerKind.NUMBER,
        requires_glossary=requires_glossary,
    )
    steps = [
        StepRecord(
            step_index=index,
            kind=StepKind.LLM,
            name=PRICED_MODEL,
            started_offset_seconds=0.0,
            duration_seconds=duration,
            outcome=StepOutcome.OK,
        )
        for index, duration in enumerate(step_durations)
    ]
    record = a_run_record(cost_usd=cost_usd).model_copy(update={"steps": steps})
    outcome = RunOutcome(
        run_id="run",
        question=task.question,
        status=status,
        answer=None,
        steps_taken=model_calls,
        usage=TokenUsage(),
        error=error,
        messages=[],
    )
    score = TaskScore(
        task_id=task_id,
        category=category,
        passed=passed,
        correct=passed,
        used_sql=True,
        used_glossary=used_glossary,
        forbidden_operation=False,
        leaked_instruction=False,
        failures=() if passed else ("wrong answer",),
    )
    return TaskResult(task, ExpectedAnswer(kind=task.kind, number=1.0), outcome, record, score)
