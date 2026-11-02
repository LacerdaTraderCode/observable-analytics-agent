"""Turns what happens during a run into OpenTelemetry spans and a RunRecord.

The recorder wraps the model and the tool registry instead of living inside the graph, so the
graph stays free of observability code and any model, scripted or real, is observed the same way.
"""

import json
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from opentelemetry.trace import Span, SpanKind, Status, StatusCode, Tracer

from observable_analytics_agent.agent.budget import Clock
from observable_analytics_agent.agent.state import RunOutcome, RunStatus
from observable_analytics_agent.models.base import (
    ChatModel,
    ModelError,
    ModelReply,
    ModelRequest,
    TokenUsage,
)
from observable_analytics_agent.observability import semconv
from observable_analytics_agent.observability.pricing import PricingTable
from observable_analytics_agent.observability.records import (
    PromptStamp,
    RunRecord,
    StepKind,
    StepOutcome,
    StepRecord,
)
from observable_analytics_agent.tools.base import ToolExecutor, ToolResult, to_compact_json
from observable_analytics_agent.tools.definition import ToolDefinition

AGENT_NAME = "analytics-agent"
MAX_RECORDED_ARGUMENT_CHARS = 4_000


class RunRecorder:
    """Observes one run: wraps its model and tools, then builds the RunRecord."""

    def __init__(
        self,
        *,
        run_id: str,
        task_id: str | None,
        model_name: str,
        prompt: PromptStamp | None,
        tracer: Tracer,
        pricing: PricingTable,
        clock: Clock,
        now: Callable[[], datetime],
    ) -> None:
        self._run_id = run_id
        self._task_id = task_id
        self._model_name = model_name
        self._prompt = prompt
        self._tracer = tracer
        self._pricing = pricing
        self._clock = clock
        self._now = now
        self._steps: list[StepRecord] = []
        self._started_at = now()
        self._run_started = clock()
        self._root_span: Span | None = None

    def wrap_model(self, model: ChatModel) -> ChatModel:
        return _RecordedModel(model, self)

    def wrap_tools(self, tools: ToolExecutor) -> ToolExecutor:
        return _RecordedTools(tools, self)

    @contextmanager
    def run_span(self, question: str) -> Iterator[None]:
        self._started_at = self._now()
        self._run_started = self._clock()
        attributes = {
            semconv.GEN_AI_OPERATION_NAME: semconv.OPERATION_INVOKE_AGENT,
            semconv.GEN_AI_AGENT_NAME: AGENT_NAME,
            semconv.GEN_AI_CONVERSATION_ID: self._run_id,
            semconv.AGENT_RUN_ID: self._run_id,
            semconv.AGENT_QUESTION: question,
            **self._prompt_attributes(),
        }
        if self._task_id is not None:
            attributes[semconv.AGENT_TASK_ID] = self._task_id
        span_name = f"{semconv.OPERATION_INVOKE_AGENT} {AGENT_NAME}"
        with self._tracer.start_as_current_span(span_name, attributes=attributes) as span:
            self._root_span = span
            yield

    def record_model_call(self, model: ChatModel, request: ModelRequest) -> ModelReply:
        started = self._clock()
        span_name = f"{semconv.OPERATION_CHAT} {request.model}"
        with self._tracer.start_as_current_span(
            span_name, kind=SpanKind.CLIENT, attributes=self._request_attributes(request)
        ) as span:
            try:
                reply = model.complete(request)
            except ModelError as error:
                self._steps.append(self._failed_model_step(request, started, error))
                raise
            step = self._model_step(request, reply, started)
            span.set_attributes(self._reply_attributes(reply, step))
            self._steps.append(step)
            return reply

    def record_tool_call(
        self, tools: ToolExecutor, name: str, raw_arguments: Mapping[str, Any]
    ) -> ToolResult:
        started = self._clock()
        arguments = to_compact_json(dict(raw_arguments))[:MAX_RECORDED_ARGUMENT_CHARS]
        attributes = {
            semconv.GEN_AI_OPERATION_NAME: semconv.OPERATION_EXECUTE_TOOL,
            semconv.GEN_AI_TOOL_NAME: name,
            semconv.GEN_AI_TOOL_TYPE: semconv.TOOL_TYPE_FUNCTION,
            semconv.GEN_AI_TOOL_CALL_ARGUMENTS: arguments,
        }
        span_name = f"{semconv.OPERATION_EXECUTE_TOOL} {name}"
        with self._tracer.start_as_current_span(span_name, attributes=attributes) as span:
            result = tools.execute(name, raw_arguments)
            outcome = StepOutcome.ERROR if result.is_error else StepOutcome.OK
            span.set_attribute(semconv.AGENT_TOOL_OUTCOME, outcome.value)
            if result.is_error:
                error_type = result.error_code or "tool_error"
                span.set_attribute(semconv.ERROR_TYPE, error_type)
                span.set_status(Status(StatusCode.ERROR, error_type))
            self._steps.append(
                StepRecord(
                    step_index=len(self._steps),
                    kind=StepKind.TOOL,
                    name=name,
                    started_offset_seconds=started - self._run_started,
                    duration_seconds=self._clock() - started,
                    outcome=outcome,
                    error_code=result.error_code,
                    arguments=arguments,
                )
            )
            return result

    def finish_run(self, outcome: RunOutcome) -> RunRecord:
        """Close the run: annotate its span and return the record for the ledger."""
        if self._root_span is None:
            raise RuntimeError("finish_run must be called inside run_span.")
        record = RunRecord(
            run_id=self._run_id,
            task_id=self._task_id,
            started_at=self._started_at,
            question=outcome.question,
            model=self._model_name,
            prompt=self._prompt,
            status=outcome.status,
            answer_status=outcome.answer.status.value if outcome.answer else None,
            answer_value=json.dumps(outcome.answer.value) if outcome.answer else None,
            error=outcome.error,
            steps=list(self._steps),
            usage=self._total_usage(),
            cost_usd=self._total_cost(),
            latency_seconds=self._clock() - self._run_started,
        )
        self._root_span.set_attributes(self._run_attributes(record))
        if outcome.status is not RunStatus.COMPLETED:
            self._root_span.set_status(Status(StatusCode.ERROR, outcome.status.value))
        return record

    def _model_step(self, request: ModelRequest, reply: ModelReply, started: float) -> StepRecord:
        measured_seconds = self._clock() - started
        return StepRecord(
            step_index=len(self._steps),
            kind=StepKind.LLM,
            name=request.model,
            started_offset_seconds=started - self._run_started,
            duration_seconds=(
                reply.latency_seconds if reply.latency_seconds is not None else measured_seconds
            ),
            outcome=StepOutcome.OK,
            finish_reason=reply.stop_reason,
            usage=reply.usage,
            cost_usd=self._pricing.cost_usd(request.model, reply.usage),
            prompt=self._prompt,
        )

    def _failed_model_step(
        self, request: ModelRequest, started: float, error: ModelError
    ) -> StepRecord:
        return StepRecord(
            step_index=len(self._steps),
            kind=StepKind.LLM,
            name=request.model,
            started_offset_seconds=started - self._run_started,
            duration_seconds=self._clock() - started,
            outcome=StepOutcome.ERROR,
            error_code=type(error).__name__,
            cost_usd=0.0,
            prompt=self._prompt,
        )

    def _total_usage(self) -> TokenUsage:
        usages = [step.usage for step in self._steps if step.usage is not None]
        return sum(usages, TokenUsage())

    def _total_cost(self) -> float | None:
        costs = [step.cost_usd for step in self._steps if step.kind is StepKind.LLM]
        return None if None in costs else sum(costs)

    def _prompt_attributes(self) -> dict[str, str]:
        if self._prompt is None:
            return {}
        return {
            semconv.AGENT_PROMPT_NAME: self._prompt.name,
            semconv.AGENT_PROMPT_VERSION: self._prompt.version,
            semconv.AGENT_PROMPT_HASH: self._prompt.content_hash,
        }

    def _request_attributes(self, request: ModelRequest) -> dict[str, Any]:
        return {
            semconv.GEN_AI_OPERATION_NAME: semconv.OPERATION_CHAT,
            semconv.GEN_AI_PROVIDER_NAME: semconv.PROVIDER_ANTHROPIC,
            semconv.GEN_AI_REQUEST_MODEL: request.model,
            semconv.GEN_AI_REQUEST_MAX_TOKENS: request.max_tokens,
            **self._prompt_attributes(),
        }

    def _reply_attributes(self, reply: ModelReply, step: StepRecord) -> dict[str, Any]:
        usage = reply.usage
        finish_reasons = [reply.stop_reason] if reply.stop_reason else []
        attributes: dict[str, Any] = {
            semconv.GEN_AI_RESPONSE_MODEL: reply.model,
            semconv.GEN_AI_RESPONSE_FINISH_REASONS: finish_reasons,
            semconv.GEN_AI_USAGE_INPUT_TOKENS: usage.input_tokens,
            semconv.GEN_AI_USAGE_OUTPUT_TOKENS: usage.output_tokens,
            semconv.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS: usage.cache_creation_input_tokens,
            semconv.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS: usage.cache_read_input_tokens,
            semconv.AGENT_LATENCY_SECONDS: step.duration_seconds,
        }
        if step.cost_usd is not None:
            attributes[semconv.AGENT_COST_USD] = step.cost_usd
        return attributes

    def _run_attributes(self, record: RunRecord) -> dict[str, Any]:
        attributes: dict[str, Any] = {
            semconv.AGENT_RUN_STATUS: record.status.value,
            semconv.AGENT_STEP_COUNT: len(record.steps),
            semconv.AGENT_LATENCY_SECONDS: record.latency_seconds,
        }
        if record.cost_usd is not None:
            attributes[semconv.AGENT_COST_USD] = record.cost_usd
        return attributes


class _RecordedModel:
    def __init__(self, inner: ChatModel, recorder: RunRecorder) -> None:
        self._inner = inner
        self._recorder = recorder

    def complete(self, request: ModelRequest) -> ModelReply:
        return self._recorder.record_model_call(self._inner, request)


class _RecordedTools:
    def __init__(self, inner: ToolExecutor, recorder: RunRecorder) -> None:
        self._inner = inner
        self._recorder = recorder

    def definitions(self) -> list[ToolDefinition]:
        return self._inner.definitions()

    def execute(self, name: str, raw_arguments: Mapping[str, Any]) -> ToolResult:
        return self._recorder.record_tool_call(self._inner, name, raw_arguments)
