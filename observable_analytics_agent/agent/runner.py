"""The entry point that runs the agent graph for one question."""

import time
import uuid

from observable_analytics_agent.agent.budget import Clock, RunBudget
from observable_analytics_agent.agent.graph import AgentConfig, build_agent_graph
from observable_analytics_agent.agent.state import AgentState, RunOutcome
from observable_analytics_agent.models.base import ChatModel, text_message
from observable_analytics_agent.tools.base import ToolExecutor

SUPERSTEPS_PER_MODEL_CALL = 3
RECURSION_MARGIN = 5


class AnalystAgent:
    """Answers one question per call by running the graph with the given model and tools."""

    def __init__(
        self,
        model: ChatModel,
        tools: ToolRegistry,
        config: AgentConfig,
        clock: Clock = time.monotonic,
    ) -> None:
        self._model = model
        self._tools = tools
        self._config = config
        self._clock = clock

    def run(self, question: str, run_id: str | None = None) -> RunOutcome:
        graph = build_agent_graph(self._model, self._tools, self._config, self._clock)
        initial_state = AgentState(
            run_id=run_id or uuid.uuid4().hex,
            question=question,
            started_at=self._clock(),
            messages=[text_message("user", question)],
        )
        final_values = graph.invoke(
            initial_state, config={"recursion_limit": _recursion_limit(self._config.budget)}
        )
        return RunOutcome.from_state(AgentState.model_validate(final_values))


def _recursion_limit(budget: RunBudget) -> int:
    """Leave room for the budget to stop the run before LangGraph's own safety net does."""
    return budget.max_steps * SUPERSTEPS_PER_MODEL_CALL + RECURSION_MARGIN
