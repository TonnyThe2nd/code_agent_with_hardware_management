from dataclasses import dataclass, field
from uuid import uuid4

from .value_objects import Message, ToolResult


@dataclass
class ConversationTurn:
    response: Message
    results: list[ToolResult] = field(default_factory=list)


@dataclass
class AgentSession:
    id: str = field(default_factory=lambda: str(uuid4()))
    messages: list[Message] = field(default_factory=list)
    turns: list[ConversationTurn] = field(default_factory=list)
    completed: bool = False
