from pathlib import Path
from typing import Protocol

from ...domain.policies import SafetyPolicy
from ...domain.value_objects import FilePath, ToolCall, ToolResult


class WorkspaceTool(Protocol):
    def execute(self, arguments: dict[str, str]) -> str: ...


def resolve_workspace_path(workspace: Path, value: str) -> Path:
    FilePath(value)
    relative = Path(value.replace("\\", "/"))
    if any(Path(part).is_reserved() or part.endswith((" ", "."))
           for part in relative.parts if part != "."):
        raise ValueError("Reserved or ambiguous path")
    root = workspace.resolve(strict=True)
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise ValueError("Path escapes workspace")
    return target


class ToolRegistry:
    def __init__(self, tools: dict[str, WorkspaceTool], policy: SafetyPolicy) -> None:
        self._tools = dict(tools)
        self._policy = policy

    def execute(self, call: ToolCall) -> ToolResult:
        try:
            self._policy.validate(call)
            if call.name not in self._tools:
                raise ValueError("Tool is not registered: " + call.name)
            content = self._tools[call.name].execute(dict(call.arguments))
            return ToolResult(call.id, call.name, content)
        except (ValueError, OSError, RuntimeError) as exc:
            return ToolResult(call.id, call.name, str(exc), is_error=True)


def build_default_registry(
    read_file: WorkspaceTool, write_file: WorkspaceTool,
    list_dir: WorkspaceTool, run_command: WorkspaceTool, policy: SafetyPolicy,
) -> ToolRegistry:
    return ToolRegistry({"read_file": read_file, "write_file": write_file,
                         "list_dir": list_dir, "run_command": run_command}, policy)
