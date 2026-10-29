"""The aggregate metrics are what the baseline stores and the gate compares."""

import pytest

from observable_analytics_agent.agent.state import RunStatus
from observable_analytics_agent.evaluation.metrics import percentile, summarize
from observable_analytics_agent.evaluation.tasks import TaskCategory
from observable_analytics_agent.models.recording import MISSING_RECORDING_PREFIX
from tests.support import a_task_result


@pytest.mark.parametrize(
    ("share", "expected"),
    [(0.0, 1.0), (0.5, 2.0), (0.95, 2.9), (1.0, 3.0)],
)
def test_a_percentile_interpolates_between_ranks(share: float, expected: float) -> None:
    assert percentile([3.0, 1.0, 2.0], share) == pytest.approx(expected)


def test_a_single_value_is_every_percentile() -> None:
    assert percentile([4.2], 0.95) == 4.2


def test_summarizes_success_cost_steps_and_latency() -> None:
    results = [
        a_task_result("a", cost_usd=0.001, step_durations=(1.0, 1.0), model_calls=2),
        a_task_result("b", cost_usd=0.003, step_durations=(2.0, 2.0), model_calls=4),
        a_task_result("c", passed=False, cost_usd=0.002, step_durations=(3.0,), model_calls=6),
    ]

    metrics = summarize(results)

    assert (metrics.task_count, metrics.passed_count) == (3, 2)
    assert metrics.success_rate == pytest.approx(2 / 3)
    assert metrics.average_cost_usd == pytest.approx(0.002)
    assert metrics.average_steps == pytest.approx(4.0)
    assert metrics.latency_p95_seconds == pytest.approx(3.9)


def test_the_task_latency_is_the_time_spent_in_its_calls() -> None:
    assert a_task_result(step_durations=(1.5, 0.25, 0.25)).latency_seconds == 2.0


def test_an_unknown_cost_makes_the_average_cost_unknown() -> None:
    results = [a_task_result("a"), a_task_result("b", cost_usd=None)]

    assert summarize(results).average_cost_usd is None


def test_the_glossary_use_rate_counts_only_tasks_that_need_the_glossary() -> None:
    results = [
        a_task_result("a", requires_glossary=True, used_glossary=True),
        a_task_result("b", requires_glossary=True, used_glossary=False),
        a_task_result("c", requires_glossary=False, used_glossary=False),
    ]

    assert summarize(results).glossary_use_rate == 0.5


def test_without_tasks_that_need_the_glossary_its_use_rate_is_unknown() -> None:
    assert summarize([a_task_result("a")]).glossary_use_rate is None


def test_counts_the_runs_that_hit_a_missing_recording() -> None:
    missing = a_task_result(
        "a",
        passed=False,
        status=RunStatus.MODEL_FAILED,
        error=f"{MISSING_RECORDING_PREFIX} (hash abc)",
    )
    other_failure = a_task_result(
        "b", passed=False, status=RunStatus.MODEL_FAILED, error="connection lost"
    )

    assert summarize([missing, other_failure, a_task_result("c")]).missing_recordings == 1


def test_breaks_the_success_rate_down_by_category() -> None:
    results = [
        a_task_result("a", category=TaskCategory.RANKING),
        a_task_result("b", category=TaskCategory.RANKING, passed=False),
        a_task_result("c", category=TaskCategory.INJECTION),
    ]

    assert summarize(results).success_rate_by_category == {"injection": 1.0, "ranking": 0.5}


def test_an_evaluation_without_results_cannot_be_summarized() -> None:
    with pytest.raises(ValueError, match="without results"):
        summarize([])
