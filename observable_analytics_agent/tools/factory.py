"""Assembles the four tools from a database and a glossary search."""

from observable_analytics_agent.tools.base import ToolRegistry
from observable_analytics_agent.tools.describe_schema import DescribeSchemaTool
from observable_analytics_agent.tools.glossary import GlossarySearch, SearchBusinessGlossaryTool
from observable_analytics_agent.tools.run_read_only_query import RunReadOnlyQueryTool
from observable_analytics_agent.tools.sql_runner import ReadOnlyDatabase, SqlRunner
from observable_analytics_agent.tools.submit_answer import SubmitAnswerTool


def build_tool_registry(database: ReadOnlyDatabase, glossary: GlossarySearch) -> ToolRegistry:
    return ToolRegistry(
        [
            DescribeSchemaTool(database),
            RunReadOnlyQueryTool(SqlRunner(database)),
            SearchBusinessGlossaryTool(glossary),
            SubmitAnswerTool(),
        ]
    )
