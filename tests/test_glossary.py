"""The local glossary ranks definitions and rejects malformed glossary files."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from observable_analytics_agent.tools.errors import InvalidArguments
from observable_analytics_agent.tools.glossary import (
    GlossaryHit,
    LocalGlossary,
    SearchBusinessGlossaryTool,
    load_glossary,
)

RELIED_ON_TERMS = {
    "reference date",
    "completed order",
    "revenue",
    "average order value",
    "active customer",
    "churned customer",
    "repeat customer",
    "high-value customer",
    "refund rate",
    "cancellation rate",
    "open ticket backlog",
    "resolution time",
}


@pytest.fixture(scope="module")
def glossary() -> LocalGlossary:
    return LocalGlossary(load_glossary())


def terms_of(hits: list[GlossaryHit]) -> list[str]:
    return [hit.source.removeprefix("glossary:") for hit in hits]


def test_finds_a_term_by_its_name(glossary: LocalGlossary) -> None:
    hits = glossary.search("What is an active customer?", 3)

    assert terms_of(hits)[0] == "active customer"
    assert "90 days" in hits[0].text


def test_finds_a_term_by_an_alias(glossary: LocalGlossary) -> None:
    assert terms_of(glossary.search("gmv", 3))[0] == "gross merchandise value"


def test_finds_a_term_by_one_of_its_words(glossary: LocalGlossary) -> None:
    assert terms_of(glossary.search("churn", 3))[0] == "churned customer"


def test_the_exact_term_outranks_entries_that_only_mention_it(glossary: LocalGlossary) -> None:
    assert terms_of(glossary.search("revenue", 3))[0] == "revenue"


def test_matches_inflected_words(glossary: LocalGlossary) -> None:
    assert "churned customer" in terms_of(glossary.search("how many customers churned?", 3))


def test_respects_the_limit(glossary: LocalGlossary) -> None:
    assert len(glossary.search("customer", 2)) == 2


def test_an_unrelated_query_matches_nothing(glossary: LocalGlossary) -> None:
    assert glossary.search("zzz qqq", 3) == []


def test_the_packaged_glossary_defines_the_terms_the_tasks_rely_on() -> None:
    terms = [entry.term for entry in load_glossary()]

    assert RELIED_ON_TERMS <= set(terms)
    assert len(terms) == len(set(terms))


def test_rejects_duplicate_terms(tmp_path: Path) -> None:
    entry = {"term": "revenue", "definition": "Money."}
    path = tmp_path / "glossary.json"
    path.write_text(json.dumps([entry, entry]), encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate glossary terms: revenue"):
        load_glossary(path)


def test_rejects_an_entry_without_a_definition(tmp_path: Path) -> None:
    path = tmp_path / "glossary.json"
    path.write_text(json.dumps([{"term": "revenue"}]), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_glossary(path)


def test_the_tool_returns_matches_as_json(glossary: LocalGlossary) -> None:
    outcome = SearchBusinessGlossaryTool(glossary).invoke({"query": "revenue", "limit": 1})

    matches = json.loads(outcome.content)["matches"]
    assert len(matches) == 1
    assert matches[0]["text"].startswith("revenue:")


@pytest.mark.parametrize("arguments", [{"query": ""}, {"query": "revenue", "limit": 99}])
def test_the_tool_rejects_invalid_arguments(
    glossary: LocalGlossary, arguments: dict[str, object]
) -> None:
    with pytest.raises(InvalidArguments):
        SearchBusinessGlossaryTool(glossary).invoke(arguments)
