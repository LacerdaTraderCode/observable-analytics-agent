"""Shared fixtures: a generated dataset, a hardened database over it and the tool registry."""

from pathlib import Path

import pytest

from observable_analytics_agent.dataset.generator import DEFAULT_SEED, generate_dataset
from observable_analytics_agent.tools.base import ToolRegistry
from observable_analytics_agent.tools.factory import build_tool_registry
from observable_analytics_agent.tools.glossary import LocalGlossary, load_glossary
from observable_analytics_agent.tools.sql_runner import ReadOnlyDatabase


@pytest.fixture(scope="session")
def dataset_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("dataset") / "analytics.duckdb"
    generate_dataset(path, DEFAULT_SEED)
    return path


@pytest.fixture
def database(dataset_path: Path) -> ReadOnlyDatabase:
    return ReadOnlyDatabase(dataset_path)


@pytest.fixture
def tool_registry(database: ReadOnlyDatabase) -> ToolRegistry:
    return build_tool_registry(database, LocalGlossary(load_glossary()))
