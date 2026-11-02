"""Record and replay: a model client that makes evaluation deterministic, free and offline.

A recording is a file named after a hash of the full request: model, system prompt, messages,
tool definitions and token limit. Changing the prompt, a tool definition, the dataset or any
message therefore produces a different hash and a missing recording, which is exactly what makes
the CI gate meaningful: the author has to record again against the real model.
"""

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from observable_analytics_agent.agent.budget import Clock
from observable_analytics_agent.models.base import ChatModel, ModelError, ModelReply, ModelRequest

DEFAULT_RECORDINGS_DIR = Path(__file__).resolve().parents[2] / "evaluation" / "recordings"
MISSING_RECORDING_PREFIX = "No recording matches this request"
HASH_PREFIX_CHARS = 12


class ModelMode(StrEnum):
    LIVE = "live"
    RECORD = "record"
    REPLAY = "replay"


class RecordingMissing(ModelError):
    """No recording exists for the request: something it depends on changed."""


class RecordingCorrupt(ModelError):
    """A recording exists but cannot be read."""


def request_fingerprint(request: ModelRequest) -> str:
    canonical = json.dumps(
        request.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class RecordingChatModel:
    """Calls the real model and stores each reply under the hash of its request."""

    def __init__(self, inner: ChatModel, directory: Path, clock: Clock) -> None:
        self._inner = inner
        self._directory = directory
        self._clock = clock

    def complete(self, request: ModelRequest) -> ModelReply:
        started = self._clock()
        reply = self._inner.complete(request)
        reply = reply.model_copy(update={"latency_seconds": self._clock() - started})
        self._store(request, reply)
        return reply

    def _store(self, request: ModelRequest, reply: ModelReply) -> None:
        fingerprint = request_fingerprint(request)
        payload = {
            "fingerprint": fingerprint,
            "request_summary": _summarize(request),
            "reply": reply.model_dump(mode="json"),
        }
        self._directory.mkdir(parents=True, exist_ok=True)
        path = self._directory / f"{fingerprint}.json"
        unfinished = path.with_suffix(".json.tmp")
        text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
        unfinished.write_text(text + "\n", encoding="utf-8")
        unfinished.replace(path)


class ReplayChatModel:
    """Answers from committed recordings and never touches the network."""

    def __init__(self, directory: Path = DEFAULT_RECORDINGS_DIR) -> None:
        self._directory = directory

    def complete(self, request: ModelRequest) -> ModelReply:
        fingerprint = request_fingerprint(request)
        path = self._directory / f"{fingerprint}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise RecordingMissing(_missing_message(request, fingerprint)) from error
        except (OSError, json.JSONDecodeError) as error:
            raise RecordingCorrupt(f"Cannot read the recording {path.name}: {error}") from error
        try:
            return ModelReply.model_validate(payload["reply"])
        except (KeyError, ValidationError) as error:
            raise RecordingCorrupt(f"The recording {path.name} has no valid reply.") from error


def _summarize(request: ModelRequest) -> dict[str, Any]:
    return {
        "model": request.model,
        "max_tokens": request.max_tokens,
        "system_sha256": hashlib.sha256(request.system.encode("utf-8")).hexdigest(),
        "message_count": len(request.messages),
        "tool_names": [tool.name for tool in request.tools],
    }


def _missing_message(request: ModelRequest, fingerprint: str) -> str:
    return (
        f"{MISSING_RECORDING_PREFIX} (hash {fingerprint[:HASH_PREFIX_CHARS]}, model "
        f"{request.model}, {len(request.messages)} messages). A prompt, a tool definition, the "
        "dataset or the conversation changed since the last recording. Record again against the "
        "real model with `python -m observable_analytics_agent evaluate --mode record` and "
        "commit the new files under evaluation/recordings/."
    )
