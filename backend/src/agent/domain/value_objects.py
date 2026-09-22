from dataclasses import dataclass
from enum import Enum


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True)
class FilePath:
    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or not self.value.strip():
            raise ValueError("Path must not be empty")
        normalized = self.value.replace("\\", "/")
        if normalized.startswith("/") or ":" in normalized or ".." in normalized.split("/"):
            raise ValueError("Path must be relative and remain inside the workspace")
        if any(ord(char) < 32 for char in normalized):
            raise ValueError("Control characters are forbidden in paths")


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("Tool call requires an id")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Tool call requires a name")
        if not isinstance(self.arguments, tuple) or any(
            not isinstance(pair, tuple) or len(pair) != 2
            or not all(isinstance(item, str) for item in pair)
            for pair in self.arguments
        ):
            raise ValueError("Arguments must be immutable string pairs")
        keys = [key for key, _ in self.arguments]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate argument names")


@dataclass(frozen=True)
class ToolResult:
    tool_call_id: str
    name: str
    content: str
    is_error: bool = False

    def __post_init__(self) -> None:
        if not self.tool_call_id.strip() or not self.name.strip():
            raise ValueError("Tool result requires call id and name")
        if not isinstance(self.content, str) or not isinstance(self.is_error, bool):
            raise ValueError("Invalid tool result")


@dataclass(frozen=True)
class Message:
    role: MessageRole
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.role, MessageRole) or not isinstance(self.content, str):
            raise ValueError("Invalid message role or content")
        if not isinstance(self.tool_calls, tuple) or any(
            not isinstance(call, ToolCall) for call in self.tool_calls
        ):
            raise ValueError("Tool calls must be an immutable tuple")
        if self.tool_calls and self.role is not MessageRole.ASSISTANT:
            raise ValueError("Only assistant messages may call tools")
        if len({call.id for call in self.tool_calls}) != len(self.tool_calls):
            raise ValueError("Duplicate tool call ids")
        if self.role is MessageRole.TOOL:
            if not self.tool_call_id or not self.name:
                raise ValueError("Tool messages require call id and name")
        elif self.tool_call_id is not None or self.name is not None:
            raise ValueError("Only tool messages may identify a tool result")
