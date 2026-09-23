from pathlib import Path
from typing import Any, Protocol

from .registry import resolve_workspace_path


SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "write_file", "description": (
            "Write a UTF-8 workspace file. For an existing file, first read it and "
            "preserve unrelated content; write only the minimal requested change."
        ),
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"], "additionalProperties": False,
        },
    },
}


class WriteFileFunction(Protocol):
    def __call__(self, path: str, content: str) -> str: ...


def write_file(workspace: Path) -> WriteFileFunction:
    root = resolve_workspace_path(workspace, ".")

    def execute(path: str, content: str) -> str:
        target = resolve_workspace_path(root, path)
        if not isinstance(content, str):
            raise ValueError("Content must be text")
        target.parent.mkdir(parents=True, exist_ok=True)
        target = resolve_workspace_path(root, path)
        if target.is_file() and target.read_bytes() == content.encode("utf-8"):
            return "Unchanged: " + path
        target.write_text(content, encoding="utf-8", newline="")
        return "Written: " + path

    return execute
