import json
from pathlib import Path

from .registry import resolve_workspace_path


class ListDir:
    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace.resolve(strict=True)

    def execute(self, arguments: dict[str, str]) -> str:
        path = resolve_workspace_path(self._workspace, arguments["path"])
        entries: list[str] = []
        for child in sorted(path.iterdir()):
            try:
                resolve_workspace_path(self._workspace, str(child.relative_to(self._workspace)))
            except ValueError:
                continue
            entries.append(child.name + ("/" if child.is_dir() else ""))
        return json.dumps(entries, ensure_ascii=False)
