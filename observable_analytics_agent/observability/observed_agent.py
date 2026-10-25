"""The analyst agent with tracing, cost, latency and a ledger around it."""

import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from opentelemetry.trace import Tracer

from observable_analytics_agent.agent.budget import Clock
from observable_analytics_agent.agent.graph import AgentConfig
from observable_analytics_agent.agent.runner import AnalystAgent
from observable_analytics_agent.agent.state import RunOutcome
from observable_analytics_agent.models.base import ChatModel
from observable_analytics_agent.observability.ledger import RunLedger
from observable_analytics_agent.observability.pricing import PricingTable
from observable_analytics_agent.observability.recorder import RunRecorder
from observable_analytics_agent.observability.records import PromptStamp, RunRecord
from observable_analytics_agent.tools.base import ToolExecutor


@dataclass(frozen=True)
class ObservedRun:
    outcome: RunOutcome
    record: RunRecord


def utc_now() -> datetime:
    """The current UTC time without a zone, which is how the ledger stores timestamps."""
    return datetime.now(UTC).replace(tzinfo=None)


class ObservedAnalystAgent:
    """Runs the agent while a recorder watches it, then stores the run in the ledger."""

    def __init__(
        self,
        model: ChatModel,
        tools: ToolExecutor,
        config: AgentConfig,
        *,
        tracer: Tracer,
        pricing: PricingTable,
        ledger: RunLedger,
        prompt: PromptStamp | None = None,
        clock: Clock = time.monotonic,
    ) -> None:
        self._model = model
        self._tools = tools
        self._config = config
        self._tracer = tracer
        self._pricing = pricing
        self._ledger = ledger
        self._prompt = prompt
        self._clock = clock

    def run(
        self, question: str, run_id: str | None = None, task_id: str | None = None
    ) -> ObservedRun:
        run_id = run_id or uuid.uuid4().hex
        recorder = RunRecorder(
            run_id=run_id,
            task_id=task_id,
            model_name=self._config.model,
            prompt=self._prompt,
            tracer=self._tracer,
            pricing=self._pricing,
            clock=self._clock,
            now=utc_now,
        )
        agent = AnalystAgent(
            recorder.wrap_model(self._model),
            recorder.wrap_tools(self._tools),
            self._config,
            self._clock,
        )
        with recorder.run_span(question):
            outcome = agent.run(question, run_id=run_id)
            record = recorder.finish_run(outcome)
        self._ledger.write(record)
        return ObservedRun(outcome=outcome, record=record)
