"""What a run leaves behind: one record per run and one per model or tool call."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from observable_analytics_agent.agent.state import RunStatus
from observable_analytics_agent.models.base import TokenUsage


class PromptStamp(BaseModel):
    """Which prompt a model call used: its name, semantic version and content hash."""

    model_config = ConfigDict(frozen=True)

    name: str
    version: str
    content_hash: str


class StepKind(StrEnum):
    LLM = "llm"
    TOOL = "tool"


class StepOutcome(StrEnum):
    OK = "ok"
    ERROR = "error"


class StepRecord(BaseModel):
    step_index: int
    kind: StepKind
    name: str
    started_offset_seconds: float
    duration_seconds: float
    outcome: StepOutcome
    error_code: str | None = None
    arguments: str | None = None
    finish_reason: str | None = None
    usage: TokenUsage | None = None
    cost_usd: float | None = None
    prompt: PromptStamp | None = None


class RunRecord(BaseModel):
    run_id: str
    task_id: str | None = None
    started_at: datetime
    question: str
    model: str
    prompt: PromptStamp | None = None
    status: RunStatus
    answer_status: str | None = None
    answer_value: str | None = None
    error: str | None = None
    steps: list[StepRecord]
    usage: TokenUsage
    cost_usd: float | None = None
    latency_seconds: float
