import json
from pathlib import Path
from typing import Any, Protocol

from .registry import resolve_workspace_path


SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "list_dir", "description": "List entries inside a workspace directory.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "default": "."}},
            "required": [], "additionalProperties": False,
        },
    },
}


class ListDirFunction(Protocol):
    def __call__(self, path: str = ".") -> str: ...


def list_dir(workspace: Path) -> ListDirFunction:
    root = resolve_workspace_path(workspace, ".")

    def execute(path: str = ".") -> str:
        target = resolve_workspace_path(root, path)
        entries: list[str] = []
        for child in sorted(target.iterdir()):
            try:
                resolved = resolve_workspace_path(root, str(child))
            except PermissionError:
                continue
            entries.append(child.name + ("/" if resolved.is_dir() else ""))
        return json.dumps(entries, ensure_ascii=False)

    return execute
