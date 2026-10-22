"""The knowledge base glossary talks to the RAG API contract; the HTTP layer is mocked."""

import json
from collections.abc import Callable

import httpx2
import pytest

from observable_analytics_agent.tools.errors import GlossaryUnavailable
from observable_analytics_agent.tools.glossary import (
    GlossaryEntry,
    GlossaryHit,
    LocalGlossary,
    SearchBusinessGlossaryTool,
)
from observable_analytics_agent.tools.knowledge_base_glossary import (
    KnowledgeBaseGlossary,
    publish_glossary,
)

BASE_URL = "http://knowledge-base.test"
SERVER_SOURCES = [
    {"document_id": "glossary-revenue", "chunk_index": 0, "text": "revenue: net sales", "score": 0.9}
]


def client_with(handler: Callable[[httpx2.Request], httpx2.Response]) -> httpx2.Client:
    return httpx2.Client(transport=httpx2.MockTransport(handler))


def answering_with_sources(request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(200, json={"answer": "generated text", "sources": SERVER_SOURCES})


def test_posts_the_question_and_maps_the_sources() -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return answering_with_sources(request)

    hits = KnowledgeBaseGlossary(f"{BASE_URL}/", client_with(handler)).search("revenue", 2)

    assert str(requests[0].url) == f"{BASE_URL}/query"
    assert json.loads(requests[0].content) == {"question": "revenue", "top_k": 2}
    assert hits == [GlossaryHit(source="glossary-revenue#0", text="revenue: net sales")]


def test_the_generated_answer_of_the_server_is_not_used() -> None:
    glossary = KnowledgeBaseGlossary(BASE_URL, client_with(answering_with_sources))

    assert "generated text" not in glossary.search("revenue", 1)[0].text


def test_an_error_status_makes_the_glossary_unavailable() -> None:
    glossary = KnowledgeBaseGlossary(BASE_URL, client_with(lambda _: httpx2.Response(503)))

    with pytest.raises(GlossaryUnavailable, match="request failed"):
        glossary.search("revenue", 1)


def test_a_network_failure_makes_the_glossary_unavailable() -> None:
    def unreachable(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("connection refused", request=request)

    glossary = KnowledgeBaseGlossary(BASE_URL, client_with(unreachable))

    with pytest.raises(GlossaryUnavailable, match="connection refused"):
        glossary.search("revenue", 1)


def test_an_unexpected_payload_makes_the_glossary_unavailable() -> None:
    glossary = KnowledgeBaseGlossary(
        BASE_URL, client_with(lambda _: httpx2.Response(200, json={"detail": "other contract"}))
    )

    with pytest.raises(GlossaryUnavailable, match="unexpected response"):
        glossary.search("revenue", 1)


def test_the_tool_has_the_same_interface_with_either_implementation() -> None:
    local = LocalGlossary([GlossaryEntry(term="revenue", definition="Net sales.")])
    remote = KnowledgeBaseGlossary(BASE_URL, client_with(answering_with_sources))

    outputs = [
        json.loads(SearchBusinessGlossaryTool(glossary).invoke({"query": "revenue"}).content)
        for glossary in (local, remote)
    ]

    for output in outputs:
        assert list(output) == ["matches"]
        assert set(output["matches"][0]) == {"source", "text"}


def test_publishes_one_document_per_glossary_entry() -> None:
    documents: list[dict[str, str]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/documents"
        documents.append(json.loads(request.content))
        return httpx2.Response(200, json={"document_id": "x", "chunks_created": 2})

    entries = [
        GlossaryEntry(term="active customer", definition="Ordered recently."),
        GlossaryEntry(term="revenue", definition="Net sales."),
    ]

    chunks_created = publish_glossary(entries, BASE_URL, client_with(handler))

    assert chunks_created == 4
    assert documents == [
        {"document_id": "glossary-active-customer", "text": "active customer: Ordered recently."},
        {"document_id": "glossary-revenue", "text": "revenue: Net sales."},
    ]


def test_a_publishing_failure_names_the_term() -> None:
    entries = [GlossaryEntry(term="revenue", definition="Net sales.")]

    with pytest.raises(GlossaryUnavailable, match="revenue"):
        publish_glossary(entries, BASE_URL, client_with(lambda _: httpx2.Response(500)))
