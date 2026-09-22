from typing import Protocol

from .value_objects import Message, ToolCall, ToolResult


class LLMProvider(Protocol):
    def complete(self, messages: tuple[Message, ...]) -> Message: ...


class ToolExecutor(Protocol):
    def execute(self, call: ToolCall) -> ToolResult: ...
