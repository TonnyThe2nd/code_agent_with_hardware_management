import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ...domain.value_objects import ToolCall, ToolResult


@dataclass(frozen=True)
class ToolActivity:
    name: str
    is_error: bool
    summary: str


def resolve_workspace_path(workspace: Path, value: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Path must not be empty")
    relative = Path(value.replace("\\", "/"))
    if any(ord(char) < 32 for char in value):
        raise PermissionError("Control characters are forbidden in paths")
    if any(Path(part).is_reserved() or ":" in part or part.endswith((" ", "."))
           for part in relative.parts if part not in (".", "..", relative.anchor)):
        raise PermissionError("Reserved or ambiguous path")
    root = workspace.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(str(root))
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise PermissionError("Path escapes workspace")
    return target


class ToolRegistry:
    def __init__(self) -> None:
        self.changed_files: set[str] = set()
        self.activities: list[ToolActivity] = []
        self._read_cache: dict[tuple[str, tuple[tuple[str, str | int], ...]], str] = {}
        self._tools: dict[str, Callable[..., str]] = {}
        self._schemas: dict[str, dict[str, Any]] = {}

    def register(self, name: str, fn: Callable[..., str], schema: dict[str, Any]) -> None:
        if not name.strip() or not callable(fn):
            raise ValueError("A tool requires a name and callable")
        if name in self._tools:
            raise ValueError("Tool already registered: " + name)
        if schema.get("type") != "function" or schema.get("function", {}).get("name") != name:
            raise ValueError("Schema must describe the registered function")
        copied = json.loads(json.dumps(schema))
        self._tools[name] = fn
        self._schemas[name] = copied

    def schemas(self) -> list[dict[str, Any]]:
        return json.loads(json.dumps(list(self._schemas.values())))

    def execute(self, call: ToolCall) -> ToolResult:
        try:
            if call.name not in self._tools:
                raise ValueError("Unknown tool: " + call.name)
            cache_key = (call.name, call.arguments)
            if call.name in {"read_file", "read_file_range"} and cache_key in self._read_cache:
                content = self._read_cache[cache_key]
            else:
                content = self._tools[call.name](**dict(call.arguments))
                if call.name in {"read_file", "read_file_range"}:
                    self._read_cache[cache_key] = content
            if call.name == "write_file" and isinstance(content, str) and content.startswith("Written: "):
                self.changed_files.add(content[len("Written: "):])
                self._read_cache.clear()
            if not isinstance(content, str):
                raise TypeError("Tool must return text")
            result = ToolResult(call.id, call.name, content)
        except Exception as exc:
            result = ToolResult(call.id, call.name, str(exc)[:10_000], is_error=True)
        self.activities.append(ToolActivity(result.name, result.is_error, result.content[:500]))
        return result
