"""The submit_answer tool: ends the run with a structured answer that can be scored."""

from observable_analytics_agent.tools.base import Tool, ToolResult, to_compact_json
from observable_analytics_agent.tools.final_answer import FinalAnswer


class SubmitAnswerTool(Tool):
    name = "submit_answer"
    description = (
        "Submit the final answer and end the run. Use status 'cannot_answer' when the data "
        "cannot answer the question or when the request asks to change data."
    )
    arguments_model = FinalAnswer

    def run(self, arguments: FinalAnswer) -> ToolResult:
        return ToolResult(content=to_compact_json({"accepted": True}), answer=arguments)
