"""Recordings make an evaluation deterministic: the same request, the same reply, no network."""

import json
from pathlib import Path

import pytest

from observable_analytics_agent.agent.budget import RunBudget
from observable_analytics_agent.agent.graph import AgentConfig
from observable_analytics_agent.agent.runner import AnalystAgent
from observable_analytics_agent.agent.state import RunStatus
from observable_analytics_agent.models.base import ModelRequest, text_message
from observable_analytics_agent.models.recording import (
    MISSING_RECORDING_PREFIX,
    RecordingChatModel,
    RecordingCorrupt,
    RecordingMissing,
    ReplayChatModel,
    request_fingerprint,
)
from observable_analytics_agent.models.scripted import (
    ScriptedChatModel,
    submit_answer_call,
    text_reply,
    tool_call,
    tool_use_reply,
)
from observable_analytics_agent.tools.base import ToolRegistry
from observable_analytics_agent.tools.definition import ToolDefinition
from tests.support import ClockAdvancingModel, FakeClock

LIVE_CALL_SECONDS = 3.5
SHA256_LENGTH = 64


def a_request(**changes: object) -> ModelRequest:
    fields = {
        "model": "test-model",
        "system": "You are an analyst.",
        "messages": [text_message("user", "How many orders?")],
        "tools": [ToolDefinition(name="t", description="d", input_schema={"type": "object"})],
        "max_tokens": 100,
    }
    return ModelRequest(**{**fields, **changes})


def recording_model(directory: Path, replies: list, clock: FakeClock | None = None):
    clock = clock or FakeClock()
    inner = ClockAdvancingModel(ScriptedChatModel(replies), clock, LIVE_CALL_SECONDS)
    return RecordingChatModel(inner, directory, clock)


def test_the_fingerprint_is_stable_and_covers_the_whole_request() -> None:
    base = a_request()

    assert request_fingerprint(base) == request_fingerprint(a_request())
    assert len(request_fingerprint(base)) == SHA256_LENGTH
    for change in (
        {"model": "other-model"},
        {"system": "You are a different analyst."},
        {"messages": [text_message("user", "How many customers?")]},
        {"tools": []},
        {"max_tokens": 200},
    ):
        assert request_fingerprint(a_request(**change)) != request_fingerprint(base), change


def test_a_recorded_reply_is_replayed_with_its_live_latency(tmp_path: Path) -> None:
    reply = text_reply("Five orders.")
    recorded = recording_model(tmp_path, [reply]).complete(a_request())

    replayed = ReplayChatModel(tmp_path).complete(a_request())

    assert recorded.latency_seconds == LIVE_CALL_SECONDS
    assert replayed == recorded
    assert replayed.content[0].get("text") == "Five orders."


def test_a_recording_is_a_readable_file_named_after_the_request_hash(tmp_path: Path) -> None:
    request = a_request()
    recording_model(tmp_path, [text_reply("Five orders.")]).complete(request)

    payload = json.loads((tmp_path / f"{request_fingerprint(request)}.json").read_text())

    assert payload["fingerprint"] == request_fingerprint(request)
    assert payload["request_summary"]["model"] == "test-model"
    assert payload["request_summary"]["message_count"] == 1
    assert payload["request_summary"]["tool_names"] == ["t"]
    assert payload["reply"]["latency_seconds"] == LIVE_CALL_SECONDS
    assert list(tmp_path.glob("*.tmp")) == []


def test_a_changed_request_has_no_recording_and_the_message_says_how_to_record(
    tmp_path: Path,
) -> None:
    recording_model(tmp_path, [text_reply("Five orders.")]).complete(a_request())

    with pytest.raises(RecordingMissing) as error:
        ReplayChatModel(tmp_path).complete(a_request(system="A new prompt."))

    message = str(error.value)
    assert message.startswith(MISSING_RECORDING_PREFIX)
    assert "evaluate --mode record" in message
    assert "evaluation/recordings/" in message


@pytest.mark.parametrize(
    "content",
    ["{not json", json.dumps({"fingerprint": "x"}), json.dumps({"reply": {"content": "x"}})],
    ids=["not json", "no reply", "invalid reply"],
)
def test_a_damaged_recording_is_reported(tmp_path: Path, content: str) -> None:
    request = a_request()
    (tmp_path / f"{request_fingerprint(request)}.json").write_text(content, encoding="utf-8")

    with pytest.raises(RecordingCorrupt):
        ReplayChatModel(tmp_path).complete(request)


def test_a_whole_run_replays_identically_without_the_live_model(
    tmp_path: Path, tool_registry: ToolRegistry
) -> None:
    replies = [
        tool_use_reply(tool_call("run_read_only_query", {"sql": "SELECT count(*) FROM orders"})),
        tool_use_reply(submit_answer_call(3942)),
    ]
    config = AgentConfig(
        model="test-model", system_prompt="You are an analyst.", budget=RunBudget()
    )
    live_clock = FakeClock()
    live = AnalystAgent(recording_model(tmp_path, replies, live_clock), tool_registry, config)

    recorded_run = live.run("How many orders?", run_id="run-1")
    replayed_run = AnalystAgent(ReplayChatModel(tmp_path), tool_registry, config).run(
        "How many orders?", run_id="run-1"
    )

    assert recorded_run.status is RunStatus.COMPLETED
    assert replayed_run == recorded_run
    assert len(list(tmp_path.glob("*.json"))) == 2
