from pathlib import Path

from .registry import resolve_workspace_path


class ReadFile:
    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace.resolve(strict=True)

    def execute(self, arguments: dict[str, str]) -> str:
        return resolve_workspace_path(self._workspace, arguments["path"]).read_text(encoding="utf-8")
