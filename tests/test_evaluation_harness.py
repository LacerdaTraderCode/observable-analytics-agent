"""The harness runs the golden tasks through an observed agent and scores every run."""

from pathlib import Path

import pytest
from opentelemetry.sdk.trace import TracerProvider

from observable_analytics_agent.agent.budget import RunBudget
from observable_analytics_agent.agent.graph import AgentConfig
from observable_analytics_agent.evaluation.expected import GoldenTaskError
from observable_analytics_agent.evaluation.golden_tasks import GOLDEN_TASKS
from observable_analytics_agent.evaluation.harness import EvaluationHarness
from observable_analytics_agent.evaluation.metrics import summarize
from observable_analytics_agent.evaluation.tasks import AnswerKind, GoldenTask, TaskCategory
from observable_analytics_agent.models.scripted import ScriptedChatModel
from observable_analytics_agent.observability.ledger import RunLedger
from observable_analytics_agent.observability.observed_agent import ObservedAnalystAgent
from observable_analytics_agent.tools.base import ToolRegistry
from observable_analytics_agent.tools.sql_runner import ReadOnlyDatabase
from tests.oracle import OracleModel
from tests.support import PRICED_MODEL, fake_pricing


def harness_with(
    model, database: ReadOnlyDatabase, tools: ToolRegistry, ledger: RunLedger
) -> EvaluationHarness:
    config = AgentConfig(
        model=PRICED_MODEL, system_prompt="You are an analyst.", budget=RunBudget(max_steps=8)
    )
    agent = ObservedAnalystAgent(
        model,
        tools,
        config,
        tracer=TracerProvider().get_tracer("test"),
        pricing=fake_pricing(),
        ledger=ledger,
    )
    return EvaluationHarness(agent, database)


@pytest.fixture
def ledger(tmp_path: Path) -> RunLedger:
    return RunLedger(tmp_path / "ledger.duckdb")


def test_a_model_that_follows_the_reference_plan_passes_every_golden_task(
    database: ReadOnlyDatabase, tool_registry: ToolRegistry, ledger: RunLedger
) -> None:
    harness = harness_with(OracleModel(database), database, tool_registry, ledger)

    results = harness.run(GOLDEN_TASKS)

    failures = {result.task.task_id: result.score.failures for result in results if not result.score.passed}
    assert failures == {}
    metrics = summarize(results)
    assert metrics.success_rate == 1.0
    assert metrics.glossary_use_rate == 1.0
    assert metrics.average_cost_usd > 0


def test_deliberate_mistakes_are_scored_as_failures_and_nothing_else_is(
    database: ReadOnlyDatabase, tool_registry: ToolRegistry, ledger: RunLedger
) -> None:
    mistakes = {"count_customers", "top_three_products_by_units", "delete_cancelled_orders"}
    harness = harness_with(OracleModel(database, mistakes), database, tool_registry, ledger)

    results = harness.run(GOLDEN_TASKS)

    assert {r.task.task_id for r in results if not r.score.passed} == mistakes
    assert all(r.score.failures == ("wrong answer",) for r in results if not r.score.passed)


def test_every_run_is_stored_in_the_ledger_with_its_task_id(
    database: ReadOnlyDatabase, tool_registry: ToolRegistry, ledger: RunLedger
) -> None:
    harness = harness_with(OracleModel(database), database, tool_registry, ledger)

    harness.run(GOLDEN_TASKS[:3])

    with ledger.read_connection() as connection:
        task_ids = [row[0] for row in connection.execute("SELECT task_id FROM runs ORDER BY 1")]
    assert task_ids == sorted(task.task_id for task in GOLDEN_TASKS[:3])


def test_a_malformed_task_fails_before_any_model_call_is_made(
    database: ReadOnlyDatabase, tool_registry: ToolRegistry, ledger: RunLedger
) -> None:
    model = ScriptedChatModel([])
    malformed = GoldenTask(
        task_id="malformed",
        question="A question?",
        category=TaskCategory.SINGLE_NUMBER,
        kind=AnswerKind.NUMBER,
    )
    harness = harness_with(model, database, tool_registry, ledger)

    with pytest.raises(GoldenTaskError, match="malformed"):
        harness.run([GOLDEN_TASKS[0], malformed])

    assert model.requests == []
