from pathlib import Path
from typing import Any, Protocol

from .registry import resolve_workspace_path


SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "read_file", "description": "Read a UTF-8 workspace file with a byte limit.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "max_bytes": {"type": "integer", "minimum": 1, "default": 50_000},
            },
            "required": ["path"], "additionalProperties": False,
        },
    },
}


class ReadFileFunction(Protocol):
    def __call__(self, path: str, max_bytes: int = 50_000) -> str: ...


def read_file(workspace: Path) -> ReadFileFunction:
    root = resolve_workspace_path(workspace, ".")

    def execute(path: str, max_bytes: int = 50_000) -> str:
        target = resolve_workspace_path(root, path)
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")
        with target.open("rb") as stream:
            return stream.read(max_bytes).decode("utf-8", errors="replace")

    return execute
