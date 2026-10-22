"""The generator must be reproducible and produce a consistent, adversarial dataset."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import duckdb
import pytest

from observable_analytics_agent.dataset import vocabulary
from observable_analytics_agent.dataset.generator import (
    CUSTOMER_COUNT,
    DATASET_START,
    DEFAULT_SEED,
    NOISE_INJECTION_COUNT,
    PLANTED_CUSTOMER_ID,
    REFERENCE_DATE,
    ensure_dataset,
    generate_dataset,
    summarize_database,
)
from observable_analytics_agent.dataset.rows import ROW_TYPES
from observable_analytics_agent.dataset.schema import TABLE_NAMES

MIN_ORDER_SPAN_DAYS = 650


@pytest.fixture(scope="module")
def connection(dataset_path: Path) -> Iterator[duckdb.DuckDBPyConnection]:
    open_connection = duckdb.connect(str(dataset_path), read_only=True)
    yield open_connection
    open_connection.close()


def single_value(connection: duckdb.DuckDBPyConnection, sql: str, *parameters: Any) -> Any:
    return connection.execute(sql, list(parameters)).fetchone()[0]


def test_the_same_seed_produces_identical_data(dataset_path: Path, tmp_path: Path) -> None:
    repeated = generate_dataset(tmp_path / "again.duckdb", DEFAULT_SEED)

    assert repeated.fingerprint == summarize_database(dataset_path).fingerprint


def test_another_seed_produces_different_data(dataset_path: Path, tmp_path: Path) -> None:
    other = generate_dataset(tmp_path / "other.duckdb", DEFAULT_SEED + 1)

    assert other.fingerprint != summarize_database(dataset_path).fingerprint


def test_every_table_is_populated(dataset_path: Path) -> None:
    summary = summarize_database(dataset_path)

    assert set(summary.row_counts) == set(TABLE_NAMES)
    assert all(count > 0 for count in summary.row_counts.values())
    assert summary.row_counts["customers"] == CUSTOMER_COUNT


def test_ensure_dataset_generates_only_once(tmp_path: Path) -> None:
    path = tmp_path / "analytics.duckdb"
    ensure_dataset(path)
    first_write = path.stat().st_mtime_ns

    ensure_dataset(path)

    assert path.stat().st_mtime_ns == first_write


def test_row_types_follow_the_table_columns(connection: duckdb.DuckDBPyConnection) -> None:
    assert list(ROW_TYPES) == list(TABLE_NAMES)
    for table_name, row_type in ROW_TYPES.items():
        columns = [row[0] for row in connection.execute(f"DESCRIBE {table_name}").fetchall()]
        assert list(row_type._fields) == columns


def test_orders_cover_the_two_year_window(connection: duckdb.DuckDBPyConnection) -> None:
    first_day, last_day = connection.execute(
        "SELECT min(CAST(ordered_at AS DATE)), max(CAST(ordered_at AS DATE)) FROM orders"
    ).fetchone()

    assert DATASET_START <= first_day
    assert last_day <= REFERENCE_DATE
    assert (last_day - first_day).days >= MIN_ORDER_SPAN_DAYS


def test_no_order_precedes_the_customer_signup(connection: duckdb.DuckDBPyConnection) -> None:
    early_orders = single_value(
        connection,
        "SELECT count(*) FROM orders o JOIN customers c USING (customer_id) "
        "WHERE CAST(o.ordered_at AS DATE) < c.signup_date",
    )

    assert early_orders == 0


@pytest.mark.parametrize(
    ("child_table", "child_column", "parent_table", "parent_column"),
    [
        ("orders", "customer_id", "customers", "customer_id"),
        ("order_items", "order_id", "orders", "order_id"),
        ("order_items", "product_id", "products", "product_id"),
        ("support_tickets", "customer_id", "customers", "customer_id"),
        ("support_tickets", "order_id", "orders", "order_id"),
    ],
)
def test_references_point_to_existing_rows(
    connection: duckdb.DuckDBPyConnection,
    child_table: str,
    child_column: str,
    parent_table: str,
    parent_column: str,
) -> None:
    orphans = single_value(
        connection,
        f"SELECT count(*) FROM {child_table} child WHERE child.{child_column} IS NOT NULL "
        f"AND NOT EXISTS (SELECT 1 FROM {parent_table} parent "
        f"WHERE parent.{parent_column} = child.{child_column})",
    )

    assert orphans == 0


def test_a_ticket_belongs_to_the_customer_who_placed_its_order(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    mismatches = single_value(
        connection,
        "SELECT count(*) FROM support_tickets t JOIN orders o USING (order_id) "
        "WHERE t.customer_id <> o.customer_id",
    )

    assert mismatches == 0


def test_products_are_never_sold_before_their_launch(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    early_sales = single_value(
        connection,
        "SELECT count(*) FROM order_items i JOIN orders o USING (order_id) "
        "JOIN products p USING (product_id) WHERE CAST(o.ordered_at AS DATE) < p.launched_on",
    )

    assert early_sales == 0


def test_ticket_status_matches_its_resolution(connection: duckdb.DuckDBPyConnection) -> None:
    inconsistent = single_value(
        connection,
        "SELECT count(*) FROM support_tickets "
        "WHERE (status = 'open') <> (resolved_at IS NULL) OR resolved_at < opened_at",
    )

    assert inconsistent == 0


def test_amounts_and_quantities_are_positive(connection: duckdb.DuckDBPyConnection) -> None:
    invalid_rows = single_value(
        connection,
        "SELECT (SELECT count(*) FROM order_items WHERE quantity <= 0 OR unit_price <= 0) "
        "+ (SELECT count(*) FROM products WHERE unit_price <= 0)",
    )

    assert invalid_rows == 0


def test_optional_columns_hold_both_nulls_and_values(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    assert single_value(connection, "SELECT count(*) FROM customers WHERE account_notes IS NULL")
    assert single_value(
        connection, "SELECT count(*) FROM customers WHERE account_notes IS NOT NULL"
    )
    assert single_value(connection, "SELECT count(*) FROM support_tickets WHERE order_id IS NULL")


def test_every_order_status_appears(connection: duckdb.DuckDBPyConnection) -> None:
    statuses = {row[0] for row in connection.execute("SELECT DISTINCT status FROM orders")}

    assert statuses == {"completed", "cancelled", "refunded"}


def test_instructions_are_planted_in_free_text_columns(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    def tickets_with_body(body: str) -> int:
        return single_value(
            connection, "SELECT count(*) FROM support_tickets WHERE body = ?", body
        )

    noise_total = sum(
        tickets_with_body(payload) for payload in vocabulary.NOISE_TICKET_INJECTIONS
    )
    keyboard_description = single_value(
        connection, "SELECT description FROM products WHERE product_name = 'Mechanical Keyboard'"
    )
    planted_notes = single_value(
        connection, "SELECT account_notes FROM customers WHERE customer_id = ?", PLANTED_CUSTOMER_ID
    )

    assert tickets_with_body(vocabulary.OVERRIDE_ANSWER_INJECTION) == 1
    assert tickets_with_body(vocabulary.DESTRUCTIVE_SQL_INJECTION) == 1
    assert noise_total == NOISE_INJECTION_COUNT
    assert keyboard_description == vocabulary.PRICE_OVERRIDE_DESCRIPTION
    assert planted_notes == vocabulary.EXFILTRATION_ACCOUNT_NOTE


def test_the_oldest_open_high_priority_ticket_carries_the_answer_override(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    body = single_value(
        connection,
        "SELECT body FROM support_tickets WHERE status = 'open' AND priority = 'high' "
        "ORDER BY opened_at, ticket_id LIMIT 1",
    )

    assert body == vocabulary.OVERRIDE_ANSWER_INJECTION


def test_the_catalog_documents_columns(connection: duckdb.DuckDBPyConnection) -> None:
    note = single_value(
        connection,
        "SELECT comment FROM duckdb_columns() "
        "WHERE table_name = 'customers' AND column_name = 'country'",
    )

    assert note == "ISO 3166-1 alpha-2 country code."
