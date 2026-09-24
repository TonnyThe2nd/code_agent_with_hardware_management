"""Small, deterministic repository briefing for an agent session.

It deliberately contains metadata only: source content remains available through
tools, which avoids sending an entire repository as untrusted prompt context.
"""
from pathlib import Path


class RepositoryContextBuilder:
    def __init__(self, workspace: Path, max_entries: int = 80) -> None:
        self._workspace = workspace.resolve(strict=True)
        self._max_entries = max_entries

    def build(self) -> str:
        ignored = {".git", ".venv", "__pycache__", ".pytest_cache", "node_modules"}
        files = [item for item in self._workspace.rglob("*") if item.is_file()
                 and not any(part in ignored for part in item.relative_to(self._workspace).parts)]
        relative = sorted(item.relative_to(self._workspace).as_posix() for item in files)
        roots = sorted({item.split("/", 1)[0] for item in relative})
        preview = ", ".join(relative[:self._max_entries])
        suffix = " (truncated)" if len(relative) > self._max_entries else ""
        return (f"Repository briefing: {len(relative)} files; top-level areas: {', '.join(roots) or '(empty)'}.\n"
                f"Paths: {preview}{suffix}\n"
                "Start by using git_status and search_code/find_symbol, then read only relevant ranges. "
                "Before an edit, state a short plan (goal, files, validation). After each write, use git_diff and "
                "inspect_diagnostics; when tests are enabled, run the narrowest affected pytest target. "
                "If validation fails, inspect the failure and attempt a focused correction before reporting it.")
