from typing import Protocol

from .value_objects import Message, ToolCall, ToolResult


class LLMProvider(Protocol):
    """Keep the agent loop independent of the model provider."""

    def complete(self, messages: tuple[Message, ...]) -> Message:
        ...


class ToolExecutor(Protocol):
    """Keep tool dispatch and execution outside the domain."""

    def execute(self, call: ToolCall) -> ToolResult:
        ...
