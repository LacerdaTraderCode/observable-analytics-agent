"""A run leaves spans, a record and a ledger row, whatever the model and the tools do."""

from pathlib import Path

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode, Tracer

from observable_analytics_agent.agent.budget import RunBudget
from observable_analytics_agent.agent.graph import AgentConfig
from observable_analytics_agent.agent.state import RunOutcome, RunStatus
from observable_analytics_agent.models.base import ModelUnavailable, TokenUsage
from observable_analytics_agent.models.scripted import (
    DEFAULT_SCRIPTED_USAGE,
    SCRIPTED_MODEL_NAME,
    ScriptedChatModel,
    submit_answer_call,
    tool_call,
    tool_use_reply,
)
from observable_analytics_agent.observability import semconv
from observable_analytics_agent.observability.ledger import RunLedger
from observable_analytics_agent.observability.observed_agent import (
    ObservedAnalystAgent,
    ObservedRun,
    utc_now,
)
from observable_analytics_agent.observability.recorder import AGENT_NAME, RunRecorder
from observable_analytics_agent.observability.records import PromptStamp, StepKind, StepOutcome
from observable_analytics_agent.tools.base import ToolRegistry
from tests.support import (
    FAKE_INPUT_PRICE,
    FAKE_OUTPUT_PRICE,
    PRICED_MODEL,
    ClockAdvancingModel,
    FakeClock,
    fake_pricing,
)

MODEL_CALL_SECONDS = 2.0
COST_PER_SCRIPTED_CALL = (
    DEFAULT_SCRIPTED_USAGE.input_tokens * FAKE_INPUT_PRICE
    + DEFAULT_SCRIPTED_USAGE.output_tokens * FAKE_OUTPUT_PRICE
) / 1_000_000
ROOT_SPAN = f"invoke_agent {AGENT_NAME}"
MODEL_SPAN = f"chat {PRICED_MODEL}"
QUERY_SPAN = "execute_tool run_read_only_query"


@pytest.fixture
def span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def tracer(span_exporter: InMemorySpanExporter) -> Tracer:
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    return provider.get_tracer("test")


@pytest.fixture
def ledger(tmp_path: Path) -> RunLedger:
    return RunLedger(tmp_path / "ledger.duckdb")


def spans_named(exporter: InMemorySpanExporter, name: str) -> list[ReadableSpan]:
    return [span for span in exporter.get_finished_spans() if span.name == name]


def query(sql: str):
    return tool_call("run_read_only_query", {"sql": sql})


def answering_model(clock: FakeClock) -> ClockAdvancingModel:
    scripted = ScriptedChatModel(
        [
            tool_use_reply(query("SELECT count(*) FROM customers")),
            tool_use_reply(submit_answer_call(7)),
        ]
    )
    return ClockAdvancingModel(scripted, clock, MODEL_CALL_SECONDS)


def observed_agent(
    model,
    tools: ToolRegistry,
    tracer: Tracer,
    ledger: RunLedger,
    *,
    clock: FakeClock | None = None,
    prompt: PromptStamp | None = None,
    model_name: str = PRICED_MODEL,
) -> ObservedAnalystAgent:
    config = AgentConfig(model=model_name, system_prompt="You are an analyst.", budget=RunBudget())
    return ObservedAnalystAgent(
        model,
        tools,
        config,
        tracer=tracer,
        pricing=fake_pricing(),
        ledger=ledger,
        prompt=prompt,
        clock=clock or FakeClock(),
    )


def run_answering_question(
    tools: ToolRegistry, tracer: Tracer, ledger: RunLedger, **options
) -> ObservedRun:
    clock = FakeClock()
    agent = observed_agent(answering_model(clock), tools, tracer, ledger, clock=clock, **options)
    return agent.run("How many customers?", run_id="run-1", task_id="task-1")


def test_a_run_leaves_a_record_with_one_step_per_call(
    tool_registry: ToolRegistry, tracer: Tracer, ledger: RunLedger
) -> None:
    run = run_answering_question(tool_registry, tracer, ledger)
    record = run.record

    assert run.outcome.status is RunStatus.COMPLETED
    assert [(step.step_index, step.kind, step.name) for step in record.steps] == [
        (0, StepKind.LLM, PRICED_MODEL),
        (1, StepKind.TOOL, "run_read_only_query"),
        (2, StepKind.LLM, PRICED_MODEL),
        (3, StepKind.TOOL, "submit_answer"),
    ]
    assert [step.started_offset_seconds for step in record.steps] == [0.0, 2.0, 2.0, 4.0]
    assert [step.duration_seconds for step in record.steps] == [2.0, 0.0, 2.0, 0.0]
    assert record.latency_seconds == 4.0
    assert record.usage == TokenUsage(input_tokens=200, output_tokens=40)
    assert record.cost_usd == pytest.approx(2 * COST_PER_SCRIPTED_CALL)
    assert (record.answer_status, record.answer_value) == ("answered", "7.0")
    assert (record.run_id, record.task_id, record.model) == ("run-1", "task-1", PRICED_MODEL)


def test_every_call_becomes_a_span_under_the_run_span(
    tool_registry: ToolRegistry, tracer: Tracer, ledger: RunLedger, span_exporter
) -> None:
    run_answering_question(tool_registry, tracer, ledger)

    spans = span_exporter.get_finished_spans()
    root = spans_named(span_exporter, ROOT_SPAN)[0]
    children = [span for span in spans if span is not root]
    assert sorted(span.name for span in children) == sorted(
        [MODEL_SPAN, MODEL_SPAN, QUERY_SPAN, "execute_tool submit_answer"]
    )
    assert root.parent is None
    assert {span.parent.span_id for span in children} == {root.context.span_id}
    assert len({span.context.trace_id for span in spans}) == 1


def test_model_spans_carry_the_genai_attributes_usage_cost_and_latency(
    tool_registry: ToolRegistry, tracer: Tracer, ledger: RunLedger, span_exporter
) -> None:
    run_answering_question(tool_registry, tracer, ledger)

    attributes = spans_named(span_exporter, MODEL_SPAN)[0].attributes
    assert attributes[semconv.GEN_AI_OPERATION_NAME] == "chat"
    assert attributes[semconv.GEN_AI_PROVIDER_NAME] == "anthropic"
    assert attributes[semconv.GEN_AI_REQUEST_MODEL] == PRICED_MODEL
    assert attributes[semconv.GEN_AI_RESPONSE_MODEL] == SCRIPTED_MODEL_NAME
    assert attributes[semconv.GEN_AI_USAGE_INPUT_TOKENS] == DEFAULT_SCRIPTED_USAGE.input_tokens
    assert attributes[semconv.GEN_AI_USAGE_OUTPUT_TOKENS] == DEFAULT_SCRIPTED_USAGE.output_tokens
    assert attributes[semconv.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] == 0
    assert tuple(attributes[semconv.GEN_AI_RESPONSE_FINISH_REASONS]) == ("tool_use",)
    assert attributes[semconv.AGENT_COST_USD] == pytest.approx(COST_PER_SCRIPTED_CALL)
    assert attributes[semconv.AGENT_LATENCY_SECONDS] == MODEL_CALL_SECONDS


def test_tool_spans_carry_the_name_the_arguments_and_the_outcome(
    tool_registry: ToolRegistry, tracer: Tracer, ledger: RunLedger, span_exporter
) -> None:
    run_answering_question(tool_registry, tracer, ledger)

    attributes = spans_named(span_exporter, QUERY_SPAN)[0].attributes
    assert attributes[semconv.GEN_AI_OPERATION_NAME] == "execute_tool"
    assert attributes[semconv.GEN_AI_TOOL_NAME] == "run_read_only_query"
    assert attributes[semconv.GEN_AI_TOOL_TYPE] == "function"
    assert attributes[semconv.GEN_AI_TOOL_CALL_ARGUMENTS] == (
        '{"sql":"SELECT count(*) FROM customers"}'
    )
    assert attributes[semconv.AGENT_TOOL_OUTCOME] == "ok"


def test_the_run_span_summarizes_the_run(
    tool_registry: ToolRegistry, tracer: Tracer, ledger: RunLedger, span_exporter
) -> None:
    run_answering_question(tool_registry, tracer, ledger)

    attributes = spans_named(span_exporter, ROOT_SPAN)[0].attributes
    assert attributes[semconv.GEN_AI_CONVERSATION_ID] == "run-1"
    assert attributes[semconv.AGENT_RUN_ID] == "run-1"
    assert attributes[semconv.AGENT_TASK_ID] == "task-1"
    assert attributes[semconv.AGENT_QUESTION] == "How many customers?"
    assert attributes[semconv.AGENT_RUN_STATUS] == "completed"
    assert attributes[semconv.AGENT_STEP_COUNT] == 4
    assert attributes[semconv.AGENT_LATENCY_SECONDS] == 4.0


def test_a_failing_tool_marks_its_span_and_its_step_as_errors(
    tool_registry: ToolRegistry, tracer: Tracer, ledger: RunLedger, span_exporter
) -> None:
    model = ScriptedChatModel(
        [tool_use_reply(query("SELECT missing FROM orders")), tool_use_reply(submit_answer_call(1))]
    )

    run = observed_agent(model, tool_registry, tracer, ledger).run("Question?")

    failed_step = run.record.steps[1]
    assert failed_step.outcome is StepOutcome.ERROR
    assert failed_step.error_code == "query_failed"
    span = spans_named(span_exporter, QUERY_SPAN)[0]
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes[semconv.ERROR_TYPE] == "query_failed"
    assert span.attributes[semconv.AGENT_TOOL_OUTCOME] == "error"


def test_a_model_failure_is_recorded_as_an_error_step_and_an_error_span(
    tool_registry: ToolRegistry, tracer: Tracer, ledger: RunLedger, span_exporter
) -> None:
    model = ScriptedChatModel([ModelUnavailable("connection lost")])

    run = observed_agent(model, tool_registry, tracer, ledger).run("Question?")

    step = run.record.steps[0]
    assert run.outcome.status is RunStatus.MODEL_FAILED
    assert (step.outcome, step.error_code) == (StepOutcome.ERROR, "ModelUnavailable")
    assert step.usage is None
    assert step.cost_usd == 0.0
    assert run.record.cost_usd == 0.0
    assert spans_named(span_exporter, MODEL_SPAN)[0].status.status_code is StatusCode.ERROR
    root = spans_named(span_exporter, ROOT_SPAN)[0]
    assert root.status.status_code is StatusCode.ERROR
    assert root.attributes[semconv.AGENT_RUN_STATUS] == "model_failed"


def test_the_run_is_stored_in_the_ledger_and_reads_back_identically(
    tool_registry: ToolRegistry, tracer: Tracer, ledger: RunLedger
) -> None:
    run = run_answering_question(tool_registry, tracer, ledger)

    assert ledger.fetch_run("run-1") == run.record


def test_the_prompt_stamp_reaches_the_spans_the_record_and_the_ledger(
    tool_registry: ToolRegistry, tracer: Tracer, ledger: RunLedger, span_exporter
) -> None:
    prompt = PromptStamp(name="analyst_system", version="1.0.0", content_hash="abc123")

    run = run_answering_question(tool_registry, tracer, ledger, prompt=prompt)

    for span_name in (ROOT_SPAN, MODEL_SPAN):
        attributes = spans_named(span_exporter, span_name)[0].attributes
        assert attributes[semconv.AGENT_PROMPT_NAME] == "analyst_system"
        assert attributes[semconv.AGENT_PROMPT_VERSION] == "1.0.0"
        assert attributes[semconv.AGENT_PROMPT_HASH] == "abc123"
    assert run.record.prompt == prompt
    assert run.record.steps[0].prompt == prompt
    assert ledger.fetch_run("run-1").steps[0].prompt == prompt


def test_a_model_without_a_listed_price_has_no_cost(
    tool_registry: ToolRegistry, tracer: Tracer, ledger: RunLedger, span_exporter
) -> None:
    run = run_answering_question(tool_registry, tracer, ledger, model_name="unpriced-model")

    assert run.record.cost_usd is None
    assert run.record.steps[0].cost_usd is None
    assert semconv.AGENT_COST_USD not in spans_named(span_exporter, ROOT_SPAN)[0].attributes
    assert ledger.fetch_run("run-1").cost_usd is None


def test_finishing_a_run_that_never_started_is_an_error(tracer: Tracer) -> None:
    recorder = RunRecorder(
        run_id="run",
        task_id=None,
        model_name=PRICED_MODEL,
        prompt=None,
        tracer=tracer,
        pricing=fake_pricing(),
        clock=FakeClock(),
        now=utc_now,
    )
    outcome = RunOutcome(
        run_id="run",
        question="Question?",
        status=RunStatus.COMPLETED,
        answer=None,
        steps_taken=0,
        usage=TokenUsage(),
        error=None,
        messages=[],
    )

    with pytest.raises(RuntimeError, match="run_span"):
        recorder.finish_run(outcome)


def test_the_span_attribute_names_follow_the_genai_conventions() -> None:
    assert semconv.GEN_AI_OPERATION_NAME == "gen_ai.operation.name"
    assert semconv.GEN_AI_USAGE_INPUT_TOKENS == "gen_ai.usage.input_tokens"
    assert semconv.GEN_AI_USAGE_OUTPUT_TOKENS == "gen_ai.usage.output_tokens"
    assert semconv.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS == "gen_ai.usage.cache_read.input_tokens"
    assert semconv.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS == (
        "gen_ai.usage.cache_creation.input_tokens"
    )
    assert semconv.GEN_AI_TOOL_NAME == "gen_ai.tool.name"
    assert semconv.GEN_AI_REQUEST_MODEL == "gen_ai.request.model"
