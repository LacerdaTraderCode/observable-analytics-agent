"""The agent as an explicit LangGraph StateGraph: nodes, conditional edges and budgets.

    call_model -> run_tools -> call_model ... -> finish
         |                                         ^
         +-> remind_to_submit -> call_model        |
         +-----------------------------------------+

A reply that holds tool calls goes to run_tools. A reply without any gets one reminder to call
submit_answer. Budgets are checked before another model call starts, never after one finishes,
so a valid answer is not thrown away because it used the last allowed step.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from observable_analytics_agent.agent.budget import Clock, RunBudget, exceeded_budget
from observable_analytics_agent.agent.state import AgentState, RunStatus
from observable_analytics_agent.models.base import (
    ChatModel,
    Message,
    ModelError,
    ModelRequest,
    text_message,
    tool_result_block,
)
from observable_analytics_agent.tools.base import ToolExecutor

DEFAULT_MAX_OUTPUT_TOKENS = 4096
MAX_REMINDERS = 1
SUBMIT_REMINDER = (
    "You have not submitted an answer yet. Call submit_answer with the final answer, "
    "or with status cannot_answer if the data cannot answer the question."
)


@dataclass(frozen=True)
class AgentConfig:
    model: str
    system_prompt: str
    budget: RunBudget = field(default_factory=RunBudget)
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS


class AgentNodes:
    """The node functions and routers of the graph, sharing the run's dependencies."""

    def __init__(
        self, model: ChatModel, tools: ToolExecutor, config: AgentConfig, clock: Clock
    ) -> None:
        self._model = model
        self._tools = tools
        self._config = config
        self._clock = clock

    def call_model(self, state: AgentState) -> dict[str, Any]:
        request = ModelRequest(
            model=self._config.model,
            system=self._config.system_prompt,
            messages=list(state.messages),
            tools=self._tools.definitions(),
            max_tokens=self._config.max_output_tokens,
        )
        try:
            reply = self._model.complete(request)
        except ModelError as error:
            return {"status": RunStatus.MODEL_FAILED, "error": str(error)}
        return {
            "messages": [Message(role="assistant", content=reply.content)],
            "steps_taken": state.steps_taken + 1,
            "usage": state.usage + reply.usage,
        }

    def run_tools(self, state: AgentState) -> dict[str, Any]:
        calls = state.messages[-1].tool_calls
        results = [self._tools.execute(call.name, call.arguments) for call in calls]
        blocks = [
            tool_result_block(call.id, result.content, result.is_error)
            for call, result in zip(calls, results, strict=True)
        ]
        update: dict[str, Any] = {"messages": [Message(role="user", content=blocks)]}
        answer = next((result.answer for result in results if result.answer is not None), None)
        if answer is not None:
            update["answer"] = answer
        return update

    def remind_to_submit(self, state: AgentState) -> dict[str, Any]:
        return {
            "messages": [text_message("user", SUBMIT_REMINDER)],
            "reminders_sent": state.reminders_sent + 1,
        }

    def finish(self, state: AgentState) -> dict[str, Any]:
        return {"status": self._final_status(state)}

    def route_after_model(
        self, state: AgentState
    ) -> Literal["run_tools", "remind_to_submit", "finish"]:
        if state.status is not None:
            return "finish"
        if state.messages[-1].tool_calls:
            return "run_tools"
        if state.reminders_sent < MAX_REMINDERS and self._exceeded(state) is None:
            return "remind_to_submit"
        return "finish"

    def route_after_tools(self, state: AgentState) -> Literal["call_model", "finish"]:
        if state.answer is not None or self._exceeded(state) is not None:
            return "finish"
        return "call_model"

    def _exceeded(self, state: AgentState) -> RunStatus | None:
        elapsed_seconds = self._clock() - state.started_at
        return exceeded_budget(state, self._config.budget, elapsed_seconds)

    def _final_status(self, state: AgentState) -> RunStatus:
        if state.status is not None:
            return state.status
        if state.answer is not None:
            return RunStatus.COMPLETED
        return self._exceeded(state) or RunStatus.NO_ANSWER


def build_agent_graph(model: ChatModel, tools: ToolExecutor, config: AgentConfig, clock: Clock):
    nodes = AgentNodes(model, tools, config, clock)
    graph = StateGraph(AgentState)
    graph.add_node("call_model", nodes.call_model)
    graph.add_node("run_tools", nodes.run_tools)
    graph.add_node("remind_to_submit", nodes.remind_to_submit)
    graph.add_node("finish", nodes.finish)
    graph.add_edge(START, "call_model")
    graph.add_conditional_edges(
        "call_model", nodes.route_after_model, ["run_tools", "remind_to_submit", "finish"]
    )
    graph.add_conditional_edges("run_tools", nodes.route_after_tools, ["call_model", "finish"])
    graph.add_edge("remind_to_submit", "call_model")
    graph.add_edge("finish", END)
    return graph.compile()
