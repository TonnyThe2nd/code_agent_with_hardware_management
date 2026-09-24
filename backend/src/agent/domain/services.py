from .entities import AgentSession, ConversationTurn
from .policies import SafetyPolicy
from .ports import LLMProvider, ToolExecutor
from .value_objects import Message, MessageRole, ToolCall, ToolResult


class AgentLoop:
    def __init__(self, llm: LLMProvider, tools: ToolExecutor, policy: SafetyPolicy) -> None:
        self._llm = llm
        self._tools = tools
        self._policy = policy

    def run(
        self, session: AgentSession, max_iterations: int = 10,
    ) -> tuple[AgentSession, int]:
        """Count LLM calls in this run, including the final response."""
        if not isinstance(session, AgentSession):
            raise ValueError("Expected an agent session")
        if type(max_iterations) is not int or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        if session.is_done():
            raise ValueError("Session is already complete")
        for iteration in range(1, max_iterations + 1):
            response = self._llm.complete(tuple(session.messages))
            if not isinstance(response, Message) or response.role is not MessageRole.ASSISTANT:
                raise ValueError("Provider must return an assistant message")
            turn = ConversationTurn(response)
            session.messages.append(response)
            session.turns.append(turn)
            if not response.tool_calls:
                session.completed = True
                return session, iteration
            for call in response.tool_calls:
                result = self._execute_tool(call)
                turn.results.append(result)
                session.messages.append(Message(
                    MessageRole.TOOL, ("ERROR: " if result.is_error else "") + result.content,
                    tool_call_id=result.tool_call_id, name=result.name,
                ))
        return session, max_iterations

    def _execute_tool(self, call: ToolCall) -> ToolResult:
        allowed, reason = self._policy.is_allowed(call)
        if not allowed:
            return ToolResult(call.id, call.name, reason or "Tool call blocked", is_error=True)
        try:
            result = self._tools.execute(call)
            if not isinstance(result, ToolResult):
                raise ValueError("Executor must return a tool result")
            if result.tool_call_id != call.id or result.name != call.name:
                raise ValueError("Tool result does not match its call")
            return result
        except (ValueError, OSError, RuntimeError) as exc:
            return ToolResult(call.id, call.name, str(exc), is_error=True)
