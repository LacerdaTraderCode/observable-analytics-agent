"""Business glossary search: a local JSON file by default, a knowledge base API on demand."""

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field, TypeAdapter

from observable_analytics_agent.tools.base import Tool, ToolResult, to_compact_json

DEFAULT_GLOSSARY_PATH = Path(__file__).resolve().parents[2] / "glossary" / "business_glossary.json"
DEFAULT_MATCH_LIMIT = 3
MAX_MATCH_LIMIT = 5
TERM_MATCH_WEIGHT = 3
DEFINITION_MATCH_WEIGHT = 1
MIN_STEM_LENGTH = 4
STOP_WORDS = frozenset(
    "a an and are by do does for how in is many much of or our per the to we what which".split()
)

_WORD = re.compile(r"[a-z0-9]+")


class GlossaryEntry(BaseModel):
    term: str
    aliases: list[str] = Field(default_factory=list)
    definition: str


class GlossaryHit(BaseModel):
    source: str
    text: str


class GlossarySearch(Protocol):
    def search(self, query: str, limit: int) -> list[GlossaryHit]: ...


def load_glossary(path: Path = DEFAULT_GLOSSARY_PATH) -> list[GlossaryEntry]:
    entries = TypeAdapter(list[GlossaryEntry]).validate_json(path.read_text(encoding="utf-8"))
    terms = [entry.term for entry in entries]
    duplicated = sorted({term for term in terms if terms.count(term) > 1})
    if duplicated:
        raise ValueError(f"Duplicate glossary terms: {', '.join(duplicated)}")
    return entries


class LocalGlossary:
    """Ranks glossary entries by how many stemmed words of the query they share."""

    def __init__(self, entries: Sequence[GlossaryEntry]) -> None:
        self._entries = list(entries)
        self._name_stems = {
            entry.term: _stems(" ".join([entry.term, *entry.aliases])) for entry in self._entries
        }
        self._definition_stems = {entry.term: _stems(entry.definition) for entry in self._entries}

    def search(self, query: str, limit: int) -> list[GlossaryHit]:
        query_stems = _stems(query)
        scored = [(self._score(entry, query_stems), entry) for entry in self._entries]
        matching = [(score, entry) for score, entry in scored if score > 0]
        ranked = sorted(matching, key=lambda pair: (-pair[0], pair[1].term))
        return [
            GlossaryHit(source=f"glossary:{entry.term}", text=f"{entry.term}: {entry.definition}")
            for _, entry in ranked[:limit]
        ]

    def _score(self, entry: GlossaryEntry, query_stems: set[str]) -> int:
        name_matches = len(query_stems & self._name_stems[entry.term])
        definition_matches = len(query_stems & self._definition_stems[entry.term])
        return TERM_MATCH_WEIGHT * name_matches + DEFINITION_MATCH_WEIGHT * definition_matches


def _stems(text: str) -> set[str]:
    words = _WORD.findall(text.lower())
    return {_stem(word) for word in words if word not in STOP_WORDS}


def _stem(word: str) -> str:
    for suffix in ("ing", "ed"):
        if word.endswith(suffix) and len(word) - len(suffix) >= MIN_STEM_LENGTH:
            return word[: -len(suffix)]
    if word.endswith("s") and not word.endswith("ss") and len(word) > MIN_STEM_LENGTH:
        return word[:-1]
    return word


class SearchBusinessGlossaryArguments(BaseModel):
    query: str = Field(
        min_length=1, description="A business term or a short question, such as 'active customer'."
    )
    limit: int = Field(
        default=DEFAULT_MATCH_LIMIT,
        ge=1,
        le=MAX_MATCH_LIMIT,
        description="Maximum number of matches to return.",
    )


class SearchBusinessGlossaryTool(Tool):
    name = "search_business_glossary"
    description = (
        "Look up the official definition of a business metric or term, such as active customer, "
        "churn or revenue. Check it before querying data for any term that could be read "
        "in more than one way."
    )
    arguments_model = SearchBusinessGlossaryArguments

    def __init__(self, glossary: GlossarySearch) -> None:
        self._glossary = glossary

    def run(self, arguments: SearchBusinessGlossaryArguments) -> ToolResult:
        hits = self._glossary.search(arguments.query, arguments.limit)
        return ToolResult(content=to_compact_json({"matches": [hit.model_dump() for hit in hits]}))
