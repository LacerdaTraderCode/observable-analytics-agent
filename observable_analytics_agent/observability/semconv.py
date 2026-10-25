"""Span attribute names.

The `gen_ai.*` names follow the OpenTelemetry GenAI semantic conventions. Those conventions
are still in Development status, so every name lives here, in one place, and a test pins them.
The `agent.*` names are this project's own: the conventions have no attribute for cost or for a
prompt version.
"""

GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS = "gen_ai.usage.cache_creation.input_tokens"
GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS = "gen_ai.usage.cache_read.input_tokens"
GEN_AI_TOOL_NAME = "gen_ai.tool.name"
GEN_AI_TOOL_TYPE = "gen_ai.tool.type"
GEN_AI_TOOL_CALL_ARGUMENTS = "gen_ai.tool.call.arguments"
GEN_AI_AGENT_NAME = "gen_ai.agent.name"
GEN_AI_CONVERSATION_ID = "gen_ai.conversation.id"
ERROR_TYPE = "error.type"

OPERATION_CHAT = "chat"
OPERATION_EXECUTE_TOOL = "execute_tool"
OPERATION_INVOKE_AGENT = "invoke_agent"
PROVIDER_ANTHROPIC = "anthropic"
TOOL_TYPE_FUNCTION = "function"

AGENT_RUN_ID = "agent.run.id"
AGENT_TASK_ID = "agent.task.id"
AGENT_QUESTION = "agent.question"
AGENT_RUN_STATUS = "agent.run.status"
AGENT_STEP_COUNT = "agent.run.steps"
AGENT_PROMPT_NAME = "agent.prompt.name"
AGENT_PROMPT_VERSION = "agent.prompt.version"
AGENT_PROMPT_HASH = "agent.prompt.hash"
AGENT_COST_USD = "agent.cost_usd"
AGENT_LATENCY_SECONDS = "agent.latency_seconds"
AGENT_TOOL_OUTCOME = "agent.tool.outcome"
