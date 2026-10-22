"""The describe_schema tool: tables, columns, types and the notes stored in the catalog."""

from typing import Any

import duckdb
from pydantic import BaseModel, Field

from observable_analytics_agent.tools.base import Tool, ToolResult, to_compact_json
from observable_analytics_agent.tools.errors import InvalidArguments
from observable_analytics_agent.tools.sql_runner import ReadOnlyDatabase

TABLES_SQL = "SELECT table_name, comment FROM duckdb_tables() ORDER BY table_name"
COLUMNS_SQL = (
    "SELECT table_name, column_name, data_type, comment FROM duckdb_columns() "
    "WHERE NOT internal ORDER BY table_name, column_index"
)


class DescribeSchemaArguments(BaseModel):
    table: str | None = Field(
        default=None, description="One table name. Omit it to describe every table."
    )


class DescribeSchemaTool(Tool):
    name = "describe_schema"
    description = (
        "List the tables of the analytics database with row counts, columns, types and notes."
    )
    arguments_model = DescribeSchemaArguments

    def __init__(self, database: ReadOnlyDatabase) -> None:
        self._database = database

    def run(self, arguments: DescribeSchemaArguments) -> ToolResult:
        tables = self._read_catalog()
        if arguments.table is not None:
            known_names = ", ".join(table["name"] for table in tables)
            tables = [table for table in tables if table["name"] == arguments.table]
            if not tables:
                raise InvalidArguments(f"Unknown table '{arguments.table}'. Tables: {known_names}.")
        return ToolResult(content=to_compact_json({"tables": tables}))

    def _read_catalog(self) -> list[dict[str, Any]]:
        connection = self._database.connect()
        try:
            table_rows = connection.execute(TABLES_SQL).fetchall()
            column_rows = connection.execute(COLUMNS_SQL).fetchall()
            row_counts = {
                name: _count_rows(connection, name) for name, _comment in table_rows
            }
        finally:
            connection.close()
        columns_by_table = _group_columns(column_rows)
        return [
            _describe_table(name, comment, row_counts[name], columns_by_table.get(name, []))
            for name, comment in table_rows
        ]


def _count_rows(connection: duckdb.DuckDBPyConnection, table_name: str) -> int:
    return connection.execute(f'SELECT count(*) FROM "{table_name}"').fetchone()[0]


def _group_columns(column_rows: list[tuple[Any, ...]]) -> dict[str, list[dict[str, str]]]:
    columns_by_table: dict[str, list[dict[str, str]]] = {}
    for table_name, column_name, data_type, comment in column_rows:
        column = {"name": column_name, "type": data_type}
        if comment:
            column["description"] = comment
        columns_by_table.setdefault(table_name, []).append(column)
    return columns_by_table


def _describe_table(
    name: str, comment: str | None, row_count: int, columns: list[dict[str, str]]
) -> dict[str, Any]:
    table: dict[str, Any] = {"name": name, "row_count": row_count, "columns": columns}
    if comment:
        table["description"] = comment
    return table
