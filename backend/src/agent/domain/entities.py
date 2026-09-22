from dataclasses import dataclass, field
from uuid import uuid4

from .value_objects import Message, MessageRole, ToolResult


@dataclass
class ConversationTurn:
    response: Message
    results: list[ToolResult] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.response, Message):
            raise ValueError("Turn requires a message")
        if self.response.role is not MessageRole.ASSISTANT:
            raise ValueError("Turn response must come from the assistant")
        if not isinstance(self.results, list) or any(
            not isinstance(result, ToolResult) for result in self.results
        ):
            raise ValueError("Turn results must be a list of tool results")
        calls = {call.id: call.name for call in self.response.tool_calls}
        seen: set[str] = set()
        for result in self.results:
            if calls.get(result.tool_call_id) != result.name:
                raise ValueError("Tool result must match a call from this turn")
            if result.tool_call_id in seen:
                raise ValueError("Duplicate tool result")
            seen.add(result.tool_call_id)
        self.results = list(self.results)


@dataclass
class AgentSession:
    id: str = field(default_factory=lambda: str(uuid4()))
    messages: list[Message] = field(default_factory=list)
    turns: list[ConversationTurn] = field(default_factory=list)
    completed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("Session requires an id")
        if not isinstance(self.messages, list) or any(
            not isinstance(message, Message) for message in self.messages
        ):
            raise ValueError("Session messages must be a list of messages")
        if not isinstance(self.turns, list) or any(
            not isinstance(turn, ConversationTurn) for turn in self.turns
        ):
            raise ValueError("Session turns must be a list of conversation turns")
        if not isinstance(self.completed, bool):
            raise ValueError("Session completion must be a boolean")
        if self.completed:
            if not self.turns or self.turns[-1].response.tool_calls:
                raise ValueError("Completed session requires a final assistant response")
            if not self.messages or self.messages[-1] != self.turns[-1].response:
                raise ValueError("Completed session history must end with its final response")
        self.messages = list(self.messages)
        self.turns = list(self.turns)

    def is_done(self) -> bool:
        return self.completed
