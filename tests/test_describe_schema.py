"""describe_schema reports the catalog, including the notes stored in it."""

import json

import pytest

from observable_analytics_agent.dataset.generator import CUSTOMER_COUNT
from observable_analytics_agent.dataset.rows import Order
from observable_analytics_agent.dataset.schema import TABLE_NAMES
from observable_analytics_agent.tools.describe_schema import DescribeSchemaTool
from observable_analytics_agent.tools.errors import InvalidArguments
from observable_analytics_agent.tools.sql_runner import ReadOnlyDatabase


def described_tables(database: ReadOnlyDatabase, **arguments: str) -> dict[str, dict]:
    outcome = DescribeSchemaTool(database).invoke(arguments)
    return {table["name"]: table for table in json.loads(outcome.content)["tables"]}


def test_lists_every_table_with_row_counts_and_columns(database: ReadOnlyDatabase) -> None:
    tables = described_tables(database)

    assert set(tables) == set(TABLE_NAMES)
    assert tables["customers"]["row_count"] == CUSTOMER_COUNT
    assert [column["name"] for column in tables["orders"]["columns"]] == list(Order._fields)


def test_reports_column_types(database: ReadOnlyDatabase) -> None:
    columns = {column["name"]: column for column in described_tables(database)["orders"]["columns"]}

    assert columns["ordered_at"]["type"] == "TIMESTAMP"
    assert columns["order_id"]["type"] == "INTEGER"


def test_includes_the_notes_stored_in_the_catalog(database: ReadOnlyDatabase) -> None:
    tables = described_tables(database)
    country = next(c for c in tables["customers"]["columns"] if c["name"] == "country")

    assert country["description"] == "ISO 3166-1 alpha-2 country code."
    assert tables["orders"]["description"].startswith("One row per order")


def test_describes_a_single_table(database: ReadOnlyDatabase) -> None:
    assert list(described_tables(database, table="products")) == ["products"]


def test_an_unknown_table_lists_the_valid_ones(database: ReadOnlyDatabase) -> None:
    with pytest.raises(InvalidArguments, match="customers"):
        DescribeSchemaTool(database).invoke({"table": "invoices"})
