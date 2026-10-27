"""The shape of a golden task and of the answer computed for it from the data."""

from dataclasses import dataclass
from enum import StrEnum


class TaskCategory(StrEnum):
    SINGLE_NUMBER = "single_number"
    FILTERED_COUNT = "filtered_count"
    RETRY = "retry"
    RANKING = "ranking"
    GLOSSARY = "glossary"
    INJECTION = "injection"
    WRITE_REQUEST = "write_request"
    IMPOSSIBLE = "impossible"


class AnswerKind(StrEnum):
    NUMBER = "number"
    TEXT = "text"
    DATE = "date"
    RANKING = "ranking"
    CANNOT_ANSWER = "cannot_answer"


@dataclass(frozen=True)
class GoldenTask:
    """One question with the query that defines its right answer.

    The expected answer is never typed by hand: it is computed by running `reference_sql` on the
    dataset when the evaluation starts. A ranking query returns (label, metric) rows and includes
    one row beyond `top_n`, so a tie at the cutoff can be detected.
    """

    task_id: str
    question: str
    category: TaskCategory
    kind: AnswerKind
    reference_sql: str | None = None
    top_n: int = 1
    tolerance: float = 0.0
    requires_glossary: bool = False
    forbidden_substrings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExpectedAnswer:
    kind: AnswerKind
    number: float | None = None
    text: str | None = None
    labels: tuple[str, ...] = ()
