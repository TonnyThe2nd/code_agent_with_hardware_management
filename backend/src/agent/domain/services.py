from .entities import AgentSession, ConversationTurn
from .policies import SafetyPolicy
from .ports import LLMProvider, ToolExecutor
from .value_objects import Message, MessageRole, ToolResult


class AgentLoop:
    def __init__(self, llm: LLMProvider, tools: ToolExecutor, policy: SafetyPolicy) -> None:
        self._llm = llm
        self._tools = tools
        self._policy = policy

    def run(self, session: AgentSession, max_iterations: int = 10) -> AgentSession:
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if session.completed:
            raise ValueError("Session is already complete")
        for _ in range(max_iterations):
            response = self._llm.complete(tuple(session.messages))
            if response.role is not MessageRole.ASSISTANT:
                raise ValueError("Provider must return an assistant message")
            session.messages.append(response)
            turn = ConversationTurn(response)
            session.turns.append(turn)
            if not response.tool_calls:
                session.completed = True
                break
            for call in response.tool_calls:
                try:
                    self._policy.validate(call)
                    result = self._tools.execute(call)
                    if result.tool_call_id != call.id or result.name != call.name:
                        raise ValueError("Tool result does not match its call")
                except (ValueError, OSError, RuntimeError) as exc:
                    result = ToolResult(call.id, call.name, str(exc), is_error=True)
                turn.results.append(result)
                session.messages.append(Message(
                    MessageRole.TOOL, result.content,
                    tool_call_id=result.tool_call_id, name=result.name,
                ))
        return session
