"""Glossary search backed by the RAG Knowledge Assistant HTTP API."""

from collections.abc import Sequence

import httpx2

from observable_analytics_agent.tools.errors import GlossaryUnavailable
from observable_analytics_agent.tools.glossary import GlossaryEntry, GlossaryHit

QUERY_PATH = "/query"
DOCUMENTS_PATH = "/documents"
DEFAULT_TIMEOUT_SECONDS = 10.0


class KnowledgeBaseGlossary:
    """Searches glossary documents ingested into a RAG Knowledge Assistant server.

    Only the retrieved sources are used. The server also generates an answer, which is ignored:
    this tool returns definitions as written, not a second model's paraphrase of them.
    """

    def __init__(self, base_url: str, client: httpx2.Client) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = client

    def search(self, query: str, limit: int) -> list[GlossaryHit]:
        try:
            response = self._client.post(
                f"{self._base_url}{QUERY_PATH}", json={"question": query, "top_k": limit}
            )
            response.raise_for_status()
            return [
                GlossaryHit(
                    source=f"{source['document_id']}#{source['chunk_index']}", text=source["text"]
                )
                for source in response.json()["sources"]
            ]
        except httpx2.HTTPError as error:
            raise GlossaryUnavailable(f"The knowledge base request failed: {error}") from error
        except (KeyError, TypeError, ValueError) as error:
            raise GlossaryUnavailable("The knowledge base returned an unexpected response.") from error


def publish_glossary(entries: Sequence[GlossaryEntry], base_url: str, client: httpx2.Client) -> int:
    """Ingest every entry as one document into the knowledge base; return the chunks created."""
    chunks_created = 0
    for entry in entries:
        document = {
            "document_id": "glossary-" + entry.term.lower().replace(" ", "-"),
            "text": f"{entry.term}: {entry.definition}",
        }
        try:
            response = client.post(f"{base_url.rstrip('/')}{DOCUMENTS_PATH}", json=document)
            response.raise_for_status()
            chunks_created += response.json()["chunks_created"]
        except httpx2.HTTPError as error:
            raise GlossaryUnavailable(f"Could not publish '{entry.term}': {error}") from error
    return chunks_created
