from pathlib import Path

from .registry import resolve_workspace_path


class WriteFile:
    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace.resolve(strict=True)

    def execute(self, arguments: dict[str, str]) -> str:
        path = resolve_workspace_path(self._workspace, arguments["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path = resolve_workspace_path(self._workspace, arguments["path"])
        path.write_text(arguments["content"], encoding="utf-8")
        return "Written: " + arguments["path"]
