from dataclasses import replace
from pathlib import Path

from ..domain.ports import LLMProvider
from ..domain.value_objects import Message, ToolCall


class WorkspaceProvider:
    """Normalize local absolute paths without weakening the domain's relative-path policy."""

    def __init__(self, provider: LLMProvider, workspace: Path) -> None:
        self._provider = provider
        self._workspace = workspace.resolve(strict=True)

    def complete(self, messages: tuple[Message, ...]) -> Message:
        response = self._provider.complete(messages)
        calls: list[ToolCall] = []
        for call in response.tool_calls:
            arguments = dict(call.arguments)
            for key in ("path", "cwd"):
                value = arguments.get(key)
                if not isinstance(value, str) or ".." in value.replace("\\", "/").split("/"):
                    continue
                path = Path(value)
                if not path.is_absolute():
                    continue
                try:
                    resolved = path.resolve()
                    relative = resolved.relative_to(self._workspace)
                except (ValueError, OSError, RuntimeError):
                    continue
                arguments[key] = relative.as_posix()
            calls.append(replace(call, arguments=arguments))
        return replace(response, tool_calls=tuple(calls))
