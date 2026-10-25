"""Aggregates over the run ledger: cost, latency, success rate and tool error rate.

A run counts as a success when it ended with the status `completed`, that is, when the agent
submitted an answer. Whether the answer is right is a question for the evaluation, which has
the reference queries; the ledger only knows how the runs went.
"""

from dataclasses import dataclass
from typing import Any

from observable_analytics_agent.observability.ledger import RunLedger

LATENCY_MEDIAN = 0.5
LATENCY_TAIL = 0.95
UNVERSIONED = "unversioned"
NOT_AVAILABLE = "n/a"

BY_PROMPT_SQL = f"""
    SELECT
        coalesce(prompt_name || '@' || prompt_version, '{UNVERSIONED}') AS prompt,
        count(*) AS runs,
        avg(CAST(status = 'completed' AS INTEGER)) AS success_rate,
        avg(cost_usd) AS average_cost_usd,
        sum(cost_usd) AS total_cost_usd,
        quantile_cont(latency_seconds, {LATENCY_MEDIAN}) AS latency_p50_seconds,
        quantile_cont(latency_seconds, {LATENCY_TAIL}) AS latency_p95_seconds
    FROM runs
    GROUP BY 1
    ORDER BY 1
"""

BY_TASK_SQL = f"""
    SELECT
        task_id,
        coalesce(prompt_name || '@' || prompt_version, '{UNVERSIONED}') AS prompt,
        count(*) AS runs,
        avg(cost_usd) AS average_cost_usd,
        avg(latency_seconds) AS average_latency_seconds
    FROM runs
    WHERE task_id IS NOT NULL
    GROUP BY 1, 2
    ORDER BY 1, 2
"""

TOOL_ERRORS_SQL = f"""
    SELECT
        coalesce(r.prompt_name || '@' || r.prompt_version, '{UNVERSIONED}') AS prompt,
        s.name AS tool,
        count(*) AS calls,
        sum(CAST(s.outcome = 'error' AS INTEGER)) AS errors,
        avg(CAST(s.outcome = 'error' AS DOUBLE)) AS error_rate
    FROM steps s
    JOIN runs r USING (run_id)
    WHERE s.kind = 'tool'
    GROUP BY 1, 2
    ORDER BY 1, 2
"""


@dataclass(frozen=True)
class PromptSummary:
    prompt: str
    runs: int
    success_rate: float
    average_cost_usd: float | None
    total_cost_usd: float | None
    latency_p50_seconds: float
    latency_p95_seconds: float


@dataclass(frozen=True)
class TaskSummary:
    task_id: str
    prompt: str
    runs: int
    average_cost_usd: float | None
    average_latency_seconds: float


@dataclass(frozen=True)
class ToolErrorRate:
    prompt: str
    tool: str
    calls: int
    errors: int
    error_rate: float


@dataclass(frozen=True)
class LedgerReport:
    by_prompt: list[PromptSummary]
    by_task: list[TaskSummary]
    tool_errors: list[ToolErrorRate]


def build_report(ledger: RunLedger) -> LedgerReport:
    with ledger.read_connection() as connection:
        return LedgerReport(
            by_prompt=[PromptSummary(*row) for row in connection.execute(BY_PROMPT_SQL).fetchall()],
            by_task=[TaskSummary(*row) for row in connection.execute(BY_TASK_SQL).fetchall()],
            tool_errors=[
                ToolErrorRate(*row) for row in connection.execute(TOOL_ERRORS_SQL).fetchall()
            ],
        )


def render_report(report: LedgerReport) -> str:
    sections = [
        _table(
            "Runs by prompt version",
            ["prompt", "runs", "success", "avg cost USD", "total cost USD", "p50 s", "p95 s"],
            [
                [
                    row.prompt,
                    row.runs,
                    _percent(row.success_rate),
                    _money(row.average_cost_usd),
                    _money(row.total_cost_usd),
                    f"{row.latency_p50_seconds:.2f}",
                    f"{row.latency_p95_seconds:.2f}",
                ]
                for row in report.by_prompt
            ],
        ),
        _table(
            "Cost and latency by task",
            ["task", "prompt", "runs", "avg cost USD", "avg latency s"],
            [
                [
                    row.task_id,
                    row.prompt,
                    row.runs,
                    _money(row.average_cost_usd),
                    f"{row.average_latency_seconds:.2f}",
                ]
                for row in report.by_task
            ],
        ),
        _table(
            "Tool error rate by prompt version",
            ["prompt", "tool", "calls", "errors", "error rate"],
            [
                [row.prompt, row.tool, row.calls, row.errors, _percent(row.error_rate)]
                for row in report.tool_errors
            ],
        ),
    ]
    return "\n\n".join(sections) + "\n"


def _percent(share: float) -> str:
    return f"{share * 100:.1f}%"


def _money(amount: float | None) -> str:
    return NOT_AVAILABLE if amount is None else f"{amount:.6f}"


def _table(title: str, header: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return f"{title}\n  (no data)"
    cells = [[str(cell) for cell in row] for row in rows]
    widths = [max(len(name), *(len(row[index]) for row in cells)) for index, name in enumerate(header)]
    lines = ["  ".join(name.ljust(width) for name, width in zip(header, widths, strict=True))]
    lines += ["  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)) for row in cells]
    return "\n".join([title, *lines])
