"""DDL and catalog comments for the five business tables."""

TABLE_NAMES = ("customers", "products", "orders", "order_items", "support_tickets")

SCHEMA_STATEMENTS = (
    """
    CREATE TABLE customers (
        customer_id INTEGER PRIMARY KEY,
        full_name VARCHAR NOT NULL,
        email VARCHAR NOT NULL,
        country VARCHAR NOT NULL,
        segment VARCHAR NOT NULL,
        signup_date DATE NOT NULL,
        account_notes VARCHAR
    )
    """,
    """
    CREATE TABLE products (
        product_id INTEGER PRIMARY KEY,
        product_name VARCHAR NOT NULL,
        category VARCHAR NOT NULL,
        unit_price DECIMAL(10, 2) NOT NULL,
        launched_on DATE NOT NULL,
        description VARCHAR NOT NULL
    )
    """,
    """
    CREATE TABLE orders (
        order_id INTEGER PRIMARY KEY,
        customer_id INTEGER NOT NULL,
        ordered_at TIMESTAMP NOT NULL,
        status VARCHAR NOT NULL,
        channel VARCHAR NOT NULL
    )
    """,
    """
    CREATE TABLE order_items (
        order_item_id INTEGER PRIMARY KEY,
        order_id INTEGER NOT NULL,
        product_id INTEGER NOT NULL,
        quantity INTEGER NOT NULL,
        unit_price DECIMAL(10, 2) NOT NULL
    )
    """,
    """
    CREATE TABLE support_tickets (
        ticket_id INTEGER PRIMARY KEY,
        customer_id INTEGER NOT NULL,
        order_id INTEGER,
        opened_at TIMESTAMP NOT NULL,
        resolved_at TIMESTAMP,
        category VARCHAR NOT NULL,
        priority VARCHAR NOT NULL,
        status VARCHAR NOT NULL,
        subject VARCHAR NOT NULL,
        body VARCHAR NOT NULL
    )
    """,
)

TABLE_COMMENTS = {
    "customers": "One row per registered customer.",
    "products": "Product catalog with the current list price.",
    "orders": "One row per order placed between 2024-01-01 and 2025-12-31.",
    "order_items": "One row per product line of an order.",
    "support_tickets": "Customer support tickets, optionally linked to an order.",
}

COLUMN_COMMENTS = {
    ("customers", "country"): "ISO 3166-1 alpha-2 country code.",
    ("customers", "segment"): "Commercial segment of the customer.",
    ("customers", "account_notes"): "Free text written by account managers. Can be NULL.",
    ("products", "unit_price"): "Current list price in USD.",
    ("products", "launched_on"): "Date the product became available.",
    ("orders", "ordered_at"): "Timestamp when the order was placed.",
    ("orders", "status"): "Order lifecycle state.",
    ("orders", "channel"): "Sales channel the order came through.",
    ("order_items", "unit_price"): "Price in USD paid per unit at purchase time.",
    ("support_tickets", "order_id"): "Related order. NULL when the ticket is not about an order.",
    ("support_tickets", "resolved_at"): "Resolution timestamp. NULL while the ticket is open.",
    ("support_tickets", "status"): "Ticket state: open or resolved.",
    ("support_tickets", "body"): "Free text written by the customer.",
}


def comment_statements() -> list[str]:
    """Return the COMMENT ON statements that document tables and columns in the catalog."""
    table_comments = [f"COMMENT ON TABLE {table} IS '{text}'" for table, text in TABLE_COMMENTS.items()]
    column_comments = [
        f"COMMENT ON COLUMN {table}.{column} IS '{text}'"
        for (table, column), text in COLUMN_COMMENTS.items()
    ]
    return [*table_comments, *column_comments]
