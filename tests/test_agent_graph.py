"""The agent graph answers from the data, recovers from failures and respects its budgets."""

import json

from observable_analytics_agent.agent.budget import RunBudget
from observable_analytics_agent.agent.graph import (
    MAX_REMINDERS,
    SUBMIT_REMINDER,
    AgentConfig,
    AgentNodes,
    build_agent_graph,
)
from observable_analytics_agent.agent.runner import AnalystAgent
from observable_analytics_agent.agent.state import AgentState, RunOutcome, RunStatus
from observable_analytics_agent.dataset.generator import CUSTOMER_COUNT
from observable_analytics_agent.models.base import (
    ContentBlock,
    Message,
    ModelReply,
    ModelUnavailable,
    TokenUsage,
    ToolCall,
    text_message,
)
from observable_analytics_agent.models.scripted import (
    DEFAULT_SCRIPTED_USAGE,
    SCRIPTED_MODEL_NAME,
    ScriptedChatModel,
    submit_answer_call,
    text_reply,
    tool_call,
    tool_use_reply,
)
from observable_analytics_agent.tools.base import ToolRegistry
from observable_analytics_agent.tools.final_answer import AnswerStatus, FinalAnswer

SYSTEM_PROMPT = "You are a careful data analyst."
MODEL_NAME = "test-model"
EXPECTED_TOOL_NAMES = [
    "describe_schema",
    "run_read_only_query",
    "search_business_glossary",
    "submit_answer",
]
SLOW_CALL_SECONDS = 100.0
SHORT_TIMEOUT_SECONDS = 60.0


class FakeClock:
    def __init__(self) -> None:
        self.seconds = 0.0

    def __call__(self) -> float:
        return self.seconds

    def advance(self, seconds: float) -> None:
        self.seconds += seconds


class ClockAdvancingModel:
    """Makes every model call take `seconds` on the fake clock."""

    def __init__(self, inner: ScriptedChatModel, clock: FakeClock, seconds: float) -> None:
        self._inner = inner
        self._clock = clock
        self._seconds = seconds

    def complete(self, request):
        self._clock.advance(self._seconds)
        return self._inner.complete(request)


def config_with(**budget: float) -> AgentConfig:
    return AgentConfig(model=MODEL_NAME, system_prompt=SYSTEM_PROMPT, budget=RunBudget(**budget))


def agent_with(model, tools: ToolRegistry, clock: FakeClock | None = None, **budget: float):
    return AnalystAgent(model, tools, config_with(**budget), clock=clock or FakeClock())


def query(sql: str, call_id: str = "") -> ToolCall:
    return tool_call("run_read_only_query", {"sql": sql}, call_id)


def tool_result_contents(message: Message) -> list[dict]:
    return [json.loads(block.get("content")) for block in message.content]


def test_answers_a_question_by_querying_the_data(tool_registry: ToolRegistry) -> None:
    model = ScriptedChatModel(
        [
            tool_use_reply(tool_call("describe_schema")),
            tool_use_reply(query("SELECT count(*) AS customers FROM customers")),
            tool_use_reply(submit_answer_call(CUSTOMER_COUNT)),
        ]
    )

    outcome = agent_with(model, tool_registry).run("How many customers?", run_id="run-1")

    assert outcome.status is RunStatus.COMPLETED
    assert outcome.run_id == "run-1"
    assert outcome.answer.value == float(CUSTOMER_COUNT)
    assert outcome.steps_taken == 3
    assert outcome.usage.total_tokens == 3 * DEFAULT_SCRIPTED_USAGE.total_tokens
    assert [message.role for message in outcome.messages] == ["user", "assistant"] * 3 + ["user"]
    assert tool_result_contents(outcome.messages[4])[0]["rows"] == [[CUSTOMER_COUNT]]


def test_every_request_carries_the_system_prompt_the_tools_and_the_conversation(
    tool_registry: ToolRegistry,
) -> None:
    model = ScriptedChatModel(
        [
            tool_use_reply(tool_call("describe_schema")),
            tool_use_reply(query("SELECT 1")),
            tool_use_reply(submit_answer_call(1)),
        ]
    )

    agent_with(model, tool_registry).run("Question?")

    assert {request.system for request in model.requests} == {SYSTEM_PROMPT}
    assert {request.model for request in model.requests} == {MODEL_NAME}
    assert [tool.name for tool in model.requests[0].tools] == EXPECTED_TOOL_NAMES
    assert [len(request.messages) for request in model.requests] == [1, 3, 5]


def test_a_failing_query_is_returned_to_the_model_and_retried(tool_registry: ToolRegistry) -> None:
    model = ScriptedChatModel(
        [
            tool_use_reply(
                query("SELECT count(*) FROM orders WHERE order_date > DATE '2025-01-01'")
            ),
            tool_use_reply(
                query("SELECT count(*) FROM orders WHERE ordered_at > DATE '2025-01-01'")
            ),
            tool_use_reply(submit_answer_call(1)),
        ]
    )

    outcome = agent_with(model, tool_registry).run("Orders in 2025?")

    failed_call = model.requests[1].messages[-1].content[0]
    assert failed_call.get("is_error") is True
    assert "ordered_at" in failed_call.get("content")
    assert outcome.status is RunStatus.COMPLETED
    assert tool_result_contents(outcome.messages[4])[0]["row_count"] == 1


def test_several_tool_calls_in_one_reply_get_one_message_with_all_results(
    tool_registry: ToolRegistry,
) -> None:
    model = ScriptedChatModel(
        [
            tool_use_reply(
                query("SELECT count(*) FROM orders", "toolu_a"),
                query("SELECT count(*) FROM customers", "toolu_b"),
            ),
            tool_use_reply(submit_answer_call(1)),
        ]
    )

    agent_with(model, tool_registry).run("Counts?")

    results_message = model.requests[1].messages[-1]
    assert results_message.role == "user"
    assert [block.get("tool_use_id") for block in results_message.content] == ["toolu_a", "toolu_b"]


def test_cannot_answer_is_a_completed_run(tool_registry: ToolRegistry) -> None:
    refusal = tool_call(
        "submit_answer",
        {"status": "cannot_answer", "explanation": "The dataset has no cost data."},
    )
    model = ScriptedChatModel([tool_use_reply(refusal)])

    outcome = agent_with(model, tool_registry).run("What is our profit margin?")

    assert outcome.status is RunStatus.COMPLETED
    assert outcome.answer.status is AnswerStatus.CANNOT_ANSWER
    assert outcome.answer.value is None


def test_an_invalid_answer_is_reported_to_the_model_and_the_run_continues(
    tool_registry: ToolRegistry,
) -> None:
    incomplete = tool_call("submit_answer", {"status": "answered", "explanation": "No value."})
    model = ScriptedChatModel([tool_use_reply(incomplete), tool_use_reply(submit_answer_call(7))])

    outcome = agent_with(model, tool_registry).run("Question?")

    assert outcome.status is RunStatus.COMPLETED
    assert outcome.answer.value == 7.0
    assert outcome.steps_taken == 2
    assert "value is required" in model.requests[1].messages[-1].content[0].get("content")


def test_an_unknown_tool_is_reported_to_the_model(tool_registry: ToolRegistry) -> None:
    model = ScriptedChatModel(
        [tool_use_reply(tool_call("drop_everything")), tool_use_reply(submit_answer_call(1))]
    )

    outcome = agent_with(model, tool_registry).run("Question?")

    assert outcome.status is RunStatus.COMPLETED
    assert tool_result_contents(outcome.messages[2])[0]["error"] == "unknown_tool"


def test_thinking_blocks_go_back_to_the_model_unchanged(tool_registry: ToolRegistry) -> None:
    thinking = ContentBlock(type="thinking", thinking="Check the schema first.", signature="sig-1")
    tool_use = ContentBlock(type="tool_use", id="toolu_1", name="describe_schema", input={})
    first_reply = ModelReply(
        content=[thinking, tool_use],
        model=SCRIPTED_MODEL_NAME,
        stop_reason="tool_use",
        usage=DEFAULT_SCRIPTED_USAGE,
    )
    model = ScriptedChatModel([first_reply, tool_use_reply(submit_answer_call(1))])

    agent_with(model, tool_registry).run("Question?")

    returned_blocks = model.requests[1].messages[1].content
    assert returned_blocks[0].to_wire() == {
        "type": "thinking",
        "thinking": "Check the schema first.",
        "signature": "sig-1",
    }


def test_a_reply_without_tool_calls_gets_one_reminder_to_submit(tool_registry: ToolRegistry) -> None:
    model = ScriptedChatModel([text_reply("I think it is 5."), tool_use_reply(submit_answer_call(5))])

    outcome = agent_with(model, tool_registry).run("Question?")

    assert outcome.status is RunStatus.COMPLETED
    assert outcome.steps_taken == 2
    reminder = model.requests[1].messages[-1]
    assert reminder.role == "user"
    assert reminder.content[0].get("text") == SUBMIT_REMINDER


def test_a_second_reply_without_tool_calls_ends_the_run_without_an_answer(
    tool_registry: ToolRegistry,
) -> None:
    model = ScriptedChatModel([text_reply("Maybe 5."), text_reply("Probably 5.")])

    outcome = agent_with(model, tool_registry).run("Question?")

    assert outcome.status is RunStatus.NO_ANSWER
    assert outcome.answer is None
    assert outcome.steps_taken == 2


def test_the_step_budget_ends_a_run_that_never_answers(tool_registry: ToolRegistry) -> None:
    model = ScriptedChatModel([tool_use_reply(query("SELECT 1"))] * 10)

    outcome = agent_with(model, tool_registry, max_steps=3).run("Question?")

    assert outcome.status is RunStatus.STEP_BUDGET_EXCEEDED
    assert outcome.steps_taken == 3
    assert len(model.requests) == 3


def test_the_last_allowed_step_can_still_submit_the_answer(tool_registry: ToolRegistry) -> None:
    model = ScriptedChatModel([tool_use_reply(query("SELECT 1")), tool_use_reply(submit_answer_call(1))])

    outcome = agent_with(model, tool_registry, max_steps=2).run("Question?")

    assert outcome.status is RunStatus.COMPLETED
    assert outcome.steps_taken == 2


def test_the_token_budget_ends_the_run(tool_registry: ToolRegistry) -> None:
    heavy = TokenUsage(input_tokens=5_000, output_tokens=1_000)
    model = ScriptedChatModel([tool_use_reply(query("SELECT 1"), usage=heavy)] * 5)

    outcome = agent_with(model, tool_registry, max_total_tokens=5_000).run("Question?")

    assert outcome.status is RunStatus.TOKEN_BUDGET_EXCEEDED
    assert outcome.steps_taken == 1
    assert outcome.usage.total_tokens == 6_000


def test_the_timeout_ends_the_run(tool_registry: ToolRegistry) -> None:
    clock = FakeClock()
    scripted = ScriptedChatModel([tool_use_reply(query("SELECT 1"))] * 5)
    slow_model = ClockAdvancingModel(scripted, clock, SLOW_CALL_SECONDS)

    outcome = agent_with(
        slow_model, tool_registry, clock=clock, timeout_seconds=SHORT_TIMEOUT_SECONDS
    ).run("Question?")

    assert outcome.status is RunStatus.TIMED_OUT
    assert outcome.steps_taken == 1


def test_a_model_failure_ends_the_run_with_a_status_instead_of_an_exception(
    tool_registry: ToolRegistry,
) -> None:
    model = ScriptedChatModel([ModelUnavailable("connection lost")])

    outcome = agent_with(model, tool_registry).run("Question?")

    assert outcome.status is RunStatus.MODEL_FAILED
    assert outcome.error == "connection lost"
    assert outcome.steps_taken == 0


def test_a_failure_in_the_middle_keeps_what_was_already_collected(
    tool_registry: ToolRegistry,
) -> None:
    model = ScriptedChatModel([tool_use_reply(query("SELECT 1")), ModelUnavailable("rate limited")])

    outcome = agent_with(model, tool_registry).run("Question?")

    assert outcome.status is RunStatus.MODEL_FAILED
    assert outcome.steps_taken == 1
    assert len(outcome.messages) == 3


def test_an_exhausted_script_is_a_model_failure(tool_registry: ToolRegistry) -> None:
    model = ScriptedChatModel([tool_use_reply(query("SELECT 1"))])

    outcome = agent_with(model, tool_registry).run("Question?")

    assert outcome.status is RunStatus.MODEL_FAILED
    assert "no reply left" in outcome.error


def test_the_run_gets_an_id_when_none_is_given(tool_registry: ToolRegistry) -> None:
    model = ScriptedChatModel([tool_use_reply(submit_answer_call(1))] * 2)
    agent = agent_with(model, tool_registry)

    first, second = agent.run("Question?"), agent.run("Question?")

    assert first.run_id != second.run_id
    assert len(first.run_id) == 32


def test_the_outcome_serializes_to_json(tool_registry: ToolRegistry) -> None:
    model = ScriptedChatModel([tool_use_reply(submit_answer_call(1))])

    outcome = agent_with(model, tool_registry).run("Question?")

    assert RunOutcome.model_validate_json(outcome.model_dump_json()) == outcome


def nodes_with(clock: FakeClock | None = None, **budget: float) -> AgentNodes:
    return AgentNodes(
        ScriptedChatModel([]), ToolRegistry([]), config_with(**budget), clock or FakeClock()
    )


def state_after(reply: ModelReply, **fields) -> AgentState:
    messages = [text_message("user", "Question?"), Message(role="assistant", content=reply.content)]
    return AgentState(run_id="run", question="Question?", started_at=0.0, messages=messages, **fields)


def test_a_reply_with_tool_calls_goes_to_the_tools() -> None:
    state = state_after(tool_use_reply(tool_call("describe_schema")))

    assert nodes_with().route_after_model(state) == "run_tools"


def test_a_reply_without_tool_calls_is_reminded_once() -> None:
    nodes = nodes_with()

    assert nodes.route_after_model(state_after(text_reply("hmm"))) == "remind_to_submit"
    assert (
        nodes.route_after_model(state_after(text_reply("hmm"), reminders_sent=MAX_REMINDERS))
        == "finish"
    )


def test_no_reminder_is_sent_once_the_budget_is_spent() -> None:
    state = state_after(text_reply("hmm"), steps_taken=1)

    assert nodes_with(max_steps=1).route_after_model(state) == "finish"


def test_a_failed_model_call_goes_straight_to_finish() -> None:
    state = state_after(text_reply("hmm"), status=RunStatus.MODEL_FAILED)

    assert nodes_with().route_after_model(state) == "finish"


def test_after_the_tools_the_run_continues_until_there_is_an_answer() -> None:
    nodes = nodes_with()
    state = state_after(tool_use_reply(tool_call("describe_schema")), steps_taken=1)

    assert nodes.route_after_tools(state) == "call_model"


def test_after_the_tools_an_answer_ends_the_run() -> None:
    answer = FinalAnswer(status=AnswerStatus.ANSWERED, value=1.0, explanation="Counted.")
    state = state_after(tool_use_reply(tool_call("submit_answer")), steps_taken=1, answer=answer)

    assert nodes_with().route_after_tools(state) == "finish"


def test_after_the_tools_a_spent_budget_ends_the_run() -> None:
    state = state_after(tool_use_reply(tool_call("describe_schema")), steps_taken=3)

    assert nodes_with(max_steps=3).route_after_tools(state) == "finish"


def test_the_graph_has_the_documented_shape(tool_registry: ToolRegistry) -> None:
    graph = build_agent_graph(
        ScriptedChatModel([]), tool_registry, config_with(), FakeClock()
    ).get_graph()

    edges = {(edge.source, edge.target) for edge in graph.edges}
    assert set(graph.nodes) == {
        "__start__",
        "call_model",
        "run_tools",
        "remind_to_submit",
        "finish",
        "__end__",
    }
    assert edges == {
        ("__start__", "call_model"),
        ("call_model", "run_tools"),
        ("call_model", "remind_to_submit"),
        ("call_model", "finish"),
        ("run_tools", "call_model"),
        ("run_tools", "finish"),
        ("remind_to_submit", "call_model"),
        ("finish", "__end__"),
    }
