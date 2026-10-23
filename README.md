# Observable Analytics Agent

An LLM data analyst agent that answers business questions about a bundled synthetic dataset by calling tools. It is built to be **observable** (traces, cost and latency per call, versioned prompts) and **evaluated** (a replay-based quality gate that fails the pull request when results drop).

> **Status: work in progress.** The repository is delivered in seven milestones, one pull request each. Milestone 1 (dataset and tools) is what exists today. The agent graph, tracing, prompts, evaluation, API and Docker arrive in the next milestones, and this README grows with them.

## Milestones

| # | Scope | Status |
|---|-------|--------|
| 1 | Repository, CI skeleton, synthetic dataset, tools with SQL safety tests | In review |
| 2 | LangGraph agent and the model interface, with a scripted fake model | Planned |
| 3 | OpenTelemetry tracing, cost, latency, DuckDB run ledger and report command | Planned |
| 4 | Prompt registry with versions and content hashes | Planned |
| 5 | Evaluation harness, golden tasks, record and replay, baseline and CI gate | Planned |
| 6 | FastAPI, CLI polish, Docker and compose | Planned |
| 7 | Documentation and cross-references | Planned |

## What exists today

### The dataset

A seeded generator writes a DuckDB file with five tables covering two years (2024-01-01 to 2025-12-31): `customers`, `products`, `orders`, `order_items` and `support_tickets`. It holds about 1,500 customers and around 4,000 orders. The same seed always produces the same data, and the data is invented: no real people or companies.

The dataset also carries **adversarial rows**: ticket bodies, a product description and a customer note that contain instructions aimed at an AI reading them (override the answer, delete data, leak emails). They exist to prove that the agent treats everything a tool returns as data, never as instructions.

### The tools

| Tool | What it does |
|------|--------------|
| `describe_schema` | Lists tables, row counts, columns, types and the notes stored in the catalog. |
| `run_read_only_query` | Runs one SQL statement on DuckDB, strictly read-only. |
| `search_business_glossary` | Returns the official definition of a business term such as "active customer" or "churn". Reads a local file by default; when `KNOWLEDGE_BASE_URL` is set it queries the [RAG Knowledge Assistant](https://github.com/LacerdaTraderCode/rag-knowledge-assistant) API instead, through the same interface. |
| `submit_answer` | Ends a run with a structured answer (status, value, explanation, queries used) so answers can be scored deterministically. |

Each tool validates its arguments with Pydantic. Failures the model can fix (a wrong column, a rejected query) come back as readable results with a stable error code instead of exceptions.

### SQL safety

The SQL tool defends itself in two independent layers:

1. **Static guard.** The text must hold exactly one statement, start with `SELECT` or `WITH`, and be classified as a SELECT by the DuckDB parser. Multiple statements, `COPY`, `ATTACH`, `INSTALL`, `LOAD`, `SET`, `PRAGMA`, `EXPORT` and every write are refused before anything runs.
2. **Sandboxed connection.** The database is opened read-only with `enable_external_access` off, extension auto-install and auto-load off, a single thread and a memory limit, and the configuration is locked afterwards. File-reading table functions such as `read_csv`, `read_text` and `glob` fail even if the guard has a gap.

Results are capped at 50 rows, long text cells are truncated, and a query that runs longer than 5 seconds is interrupted.

The tests attack both layers: stacked statements, statements hidden behind comments, `COPY`, `ATTACH`, `INSTALL`/`LOAD`, file reads through table functions, URL reads, configuration changes and every write statement. They also check that the dataset fingerprint is unchanged afterwards.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt -c constraints.txt

python -m observable_analytics_agent generate-dataset   # writes var/analytics.duckdb
pytest
```

## Project layout

```
observable_analytics_agent/
  dataset/    schema, vocabulary and the seeded generator
  tools/      the four tools, the registry and the SQL safety layers
glossary/     business glossary used by search_business_glossary
tests/        offline, deterministic tests
```

## Dependencies

Every runtime and development dependency is pinned to an exact version in `requirements.txt` and `requirements-dev.txt`. `constraints.txt` locks the whole dependency tree and was resolved with a cutoff that excludes releases younger than 14 days. Dependabot follows the same rule through a 14-day cooldown.

## License

MIT. See [LICENSE](LICENSE).
