"""Errors a tool raises to tell the model what went wrong."""

MAX_ERROR_LINES = 3
MAX_ERROR_CHARS = 500


class ToolError(Exception):
    """A failure the model can read and react to; its message goes back as the tool result."""

    code = "tool_error"


class UnknownTool(ToolError):
    code = "unknown_tool"


class InvalidArguments(ToolError):
    code = "invalid_arguments"


class QueryRejected(ToolError):
    """The query is not allowed to run, whatever its syntax."""

    code = "query_rejected"


class QueryFailed(ToolError):
    """The query was allowed but the database could not execute it."""

    code = "query_failed"


class QueryTimedOut(ToolError):
    code = "query_timed_out"


class GlossaryUnavailable(ToolError):
    code = "glossary_unavailable"


def summarize_error(error: Exception) -> str:
    """Keep the informative lines of a database error and drop the echo of the query."""
    lines = [line.strip() for line in str(error).splitlines() if line.strip()]
    informative = [line for line in lines if not line.startswith("LINE ") and set(line) != {"^"}]
    summary = " ".join(informative[:MAX_ERROR_LINES])
    return (summary or str(error))[:MAX_ERROR_CHARS]
