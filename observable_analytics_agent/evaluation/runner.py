"""Builds an evaluation: the agent in live, record or replay mode over the packaged dataset."""

import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from observable_analytics_agent.agent.budget import RunBudget
from observable_analytics_agent.agent.graph import AgentConfig
from observable_analytics_agent.dataset.generator import ensure_dataset
from observable_analytics_agent.evaluation.harness import EvaluationHarness
from observable_analytics_agent.models.anthropic_model import build_anthropic_model
from observable_analytics_agent.models.base import ChatModel
from observable_analytics_agent.models.recording import (
    DEFAULT_RECORDINGS_DIR,
    ModelMode,
    RecordingChatModel,
    ReplayChatModel,
)
from observable_analytics_agent.observability.ledger import DEFAULT_LEDGER_PATH, RunLedger
from observable_analytics_agent.observability.observed_agent import ObservedAnalystAgent
from observable_analytics_agent.observability.pricing import load_pricing_table
from observable_analytics_agent.observability.records import PromptStamp
from observable_analytics_agent.observability.tracing import SERVICE_NAME, build_tracer_provider
from observable_analytics_agent.prompts.registry import DEFAULT_PROMPTS_DIR, PromptRegistry
from observable_analytics_agent.tools.factory import build_tool_registry
from observable_analytics_agent.tools.glossary import LocalGlossary, load_glossary
from observable_analytics_agent.tools.sql_runner import ReadOnlyDatabase

EVALUATION_STEP_BUDGET = 8
DEFAULT_DATASET_PATH = Path("var/analytics.duckdb")


@dataclass(frozen=True)
class EvaluationSetup:
    mode: ModelMode
    model: str
    prompt_name: str
    prompt_version: str | None = None
    recordings_dir: Path = DEFAULT_RECORDINGS_DIR
    dataset_path: Path = DEFAULT_DATASET_PATH
    ledger_path: Path = DEFAULT_LEDGER_PATH
    prompts_dir: Path = DEFAULT_PROMPTS_DIR
    otlp_endpoint: str | None = None


@dataclass(frozen=True)
class PreparedEvaluation:
    harness: EvaluationHarness
    prompt: PromptStamp
    model: str
    mode: ModelMode


def prepare_evaluation(
    setup: EvaluationSetup, environment: Mapping[str, str]
) -> PreparedEvaluation:
    """Wire the agent for an evaluation.

    The glossary is always the local file, whatever KNOWLEDGE_BASE_URL says: an evaluation has to
    give the same definitions to every run.
    """
    prompt = PromptRegistry.load(setup.prompts_dir).get(setup.prompt_name, setup.prompt_version)
    database = ReadOnlyDatabase(ensure_dataset(setup.dataset_path))
    tools = build_tool_registry(database, LocalGlossary(load_glossary()))
    config = AgentConfig(
        model=setup.model,
        system_prompt=prompt.text,
        budget=RunBudget(max_steps=EVALUATION_STEP_BUDGET),
    )
    provider = build_tracer_provider(setup.otlp_endpoint, export_to_console=False)
    agent = ObservedAnalystAgent(
        _build_model(setup, environment),
        tools,
        config,
        tracer=provider.get_tracer(SERVICE_NAME),
        pricing=load_pricing_table(),
        ledger=RunLedger(setup.ledger_path),
        prompt=prompt.stamp,
    )
    return PreparedEvaluation(
        harness=EvaluationHarness(agent, database),
        prompt=prompt.stamp,
        model=setup.model,
        mode=setup.mode,
    )


def _build_model(setup: EvaluationSetup, environment: Mapping[str, str]) -> ChatModel:
    if setup.mode is ModelMode.REPLAY:
        return ReplayChatModel(setup.recordings_dir)
    live = build_anthropic_model(environment)
    if setup.mode is ModelMode.RECORD:
        return RecordingChatModel(live, setup.recordings_dir, time.monotonic)
    return live
