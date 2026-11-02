"""The baseline is a committed file; it must round-trip and fail loudly when it is unusable."""

import json
from datetime import date
from pathlib import Path

import pytest

from observable_analytics_agent.evaluation.baseline import (
    BaselineError,
    baseline_from_metrics,
    load_baseline,
    save_baseline,
)
from observable_analytics_agent.evaluation.metrics import summarize
from observable_analytics_agent.observability.records import PromptStamp
from tests.support import a_task_result

PROMPT = PromptStamp(name="analyst_system", version="2.0.0", content_hash="abc")
RECORDED_ON = date(2026, 10, 24)


def a_summary(**result_options):
    return summarize([a_task_result("a", **result_options), a_task_result("b", passed=False)])


def test_the_baseline_stores_the_metrics_and_where_they_came_from() -> None:
    baseline = baseline_from_metrics(a_summary(), PROMPT, "claude-haiku-5-5", RECORDED_ON)

    assert (baseline.prompt_name, baseline.prompt_version) == ("analyst_system", "2.0.0")
    assert baseline.model == "claude-haiku-5-5"
    assert baseline.task_count == 2
    assert baseline.success_rate == 0.5
    assert baseline.average_cost_usd == pytest.approx(0.001)
    assert baseline.recorded_on == RECORDED_ON


def test_a_baseline_survives_a_round_trip_through_its_file(tmp_path: Path) -> None:
    path = tmp_path / "evaluation" / "baseline.json"
    baseline = baseline_from_metrics(a_summary(), PROMPT, "claude-haiku-5-5", RECORDED_ON)

    save_baseline(baseline, path)

    assert load_baseline(path) == baseline
    assert json.loads(path.read_text())["recorded_on"] == "2026-10-24"
    assert path.read_text().endswith("\n")


def test_a_baseline_cannot_be_written_for_a_model_without_a_price() -> None:
    metrics = summarize([a_task_result("a", cost_usd=None)])

    with pytest.raises(BaselineError, match="no listed price"):
        baseline_from_metrics(metrics, PROMPT, "unpriced-model", RECORDED_ON)


def test_a_missing_baseline_is_reported(tmp_path: Path) -> None:
    with pytest.raises(BaselineError, match="No baseline at"):
        load_baseline(tmp_path / "baseline.json")


@pytest.mark.parametrize("content", ["{not json", json.dumps({"success_rate": 1.0})])
def test_an_unreadable_baseline_is_reported(tmp_path: Path, content: str) -> None:
    path = tmp_path / "baseline.json"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(BaselineError, match="Cannot read the baseline"):
        load_baseline(path)
