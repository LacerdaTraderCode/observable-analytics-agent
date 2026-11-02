"""The committed baseline: the metrics of the last evaluation a human accepted."""

from datetime import date
from pathlib import Path

from pydantic import BaseModel, ValidationError

from observable_analytics_agent.evaluation.metrics import EvaluationMetrics
from observable_analytics_agent.observability.records import PromptStamp

DEFAULT_BASELINE_PATH = Path(__file__).resolve().parents[2] / "evaluation" / "baseline.json"


class BaselineError(Exception):
    """The baseline is missing or unreadable."""


class Baseline(BaseModel):
    prompt_name: str
    prompt_version: str
    model: str
    task_count: int
    success_rate: float
    average_cost_usd: float
    latency_p95_seconds: float
    average_steps: float
    recorded_on: date


def baseline_from_metrics(
    metrics: EvaluationMetrics, prompt: PromptStamp, model: str, recorded_on: date
) -> Baseline:
    if metrics.average_cost_usd is None:
        raise BaselineError(f"Cannot write a baseline: the model {model} has no listed price.")
    return Baseline(
        prompt_name=prompt.name,
        prompt_version=prompt.version,
        model=model,
        task_count=metrics.task_count,
        success_rate=metrics.success_rate,
        average_cost_usd=metrics.average_cost_usd,
        latency_p95_seconds=metrics.latency_p95_seconds,
        average_steps=metrics.average_steps,
        recorded_on=recorded_on,
    )


def load_baseline(path: Path = DEFAULT_BASELINE_PATH) -> Baseline:
    try:
        return Baseline.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise BaselineError(f"No baseline at {path}.") from error
    except (OSError, ValidationError) as error:
        raise BaselineError(f"Cannot read the baseline at {path}: {error}") from error


def save_baseline(baseline: Baseline, path: Path = DEFAULT_BASELINE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(baseline.model_dump_json(indent=2) + "\n", encoding="utf-8")
