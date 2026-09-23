import ast
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Callable

from .registry import resolve_workspace_path


def schema(name: str, description: str, properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties,
                           "required": required or [], "additionalProperties": False}}}


RANGE_SCHEMA = schema("read_file_range", "Read an inclusive line range from a UTF-8 workspace file.",
    {"path": {"type": "string"}, "start_line": {"type": "integer", "minimum": 1}, "end_line": {"type": "integer", "minimum": 1}}, ["path", "start_line", "end_line"])
SEARCH_SCHEMA = schema("search_code", "Search literal text in workspace source files; returns paths and line numbers.",
    {"query": {"type": "string"}, "path": {"type": "string", "default": "."}, "max_results": {"type": "integer", "minimum": 1, "default": 50}}, ["query"])
SYMBOL_SCHEMA = schema("find_symbol", "Find likely class, function, method, or declaration symbols.",
    {"symbol": {"type": "string"}, "path": {"type": "string", "default": "."}, "max_results": {"type": "integer", "minimum": 1, "default": 50}}, ["symbol"])
STATUS_SCHEMA = schema("git_status", "Return concise git workspace status.", {})
DIFF_SCHEMA = schema("git_diff", "Return the current git diff, optionally limited to one workspace path.", {"path": {"type": "string"}})
DIAGNOSTICS_SCHEMA = schema("inspect_diagnostics", "Parse Python files and report syntax diagnostics without executing code.", {"path": {"type": "string", "default": "."}})


def _files(root: Path, relative: str) -> list[Path]:
    target = resolve_workspace_path(root, relative)
    if target.is_file(): return [target]
    ignored = {".git", ".venv", "__pycache__", ".pytest_cache"}
    return [p for p in target.rglob("*") if p.is_file() and not any(part in ignored for part in p.parts)]


def read_file_range(workspace: Path) -> Callable[..., str]:
    root = resolve_workspace_path(workspace, ".")
    def execute(path: str, start_line: int, end_line: int) -> str:
        if end_line < start_line: raise ValueError("end_line must be greater than or equal to start_line")
        lines = resolve_workspace_path(root, path).read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(f"{number}: {line}" for number, line in enumerate(lines[start_line - 1:end_line], start_line))
    return execute


def search_code(workspace: Path, symbols: bool = False) -> Callable[..., str]:
    root = resolve_workspace_path(workspace, ".")
    def execute(query: str | None = None, symbol: str | None = None, path: str = ".", max_results: int = 50) -> str:
        needle = symbol if symbols else query
        if not isinstance(needle, str) or not needle: raise ValueError("Search text must not be empty")
        if type(max_results) is not int or max_results < 1: raise ValueError("max_results must be positive")
        matcher = re.compile(rf"\b(?:class|def|async\s+def)\s+{re.escape(needle)}\b|\b{re.escape(needle)}\b") if symbols else None
        found = []
        for file in _files(root, path):
            if file.stat().st_size > 1_000_000: continue
            for number, line in enumerate(file.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if (matcher.search(line) if matcher else needle.lower() in line.lower()):
                    found.append({"path": file.relative_to(root).as_posix(), "line": number, "text": line[:300]})
                    if len(found) >= max_results: return json.dumps(found, ensure_ascii=False)
        return json.dumps(found, ensure_ascii=False)
    return execute


def git_inspection(workspace: Path, diff: bool = False) -> Callable[..., str]:
    root = resolve_workspace_path(workspace, ".")
    def execute(path: str | None = None) -> str:
        argv = ["git", "diff", "--"] if diff else ["git", "status", "--short"]
        if diff and path is not None:
            argv.append(resolve_workspace_path(root, path).relative_to(root).as_posix())
        result = subprocess.run(argv, cwd=root, shell=False, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10)
        if result.returncode == 128:
            return "(not a git workspace)"
        if result.returncode: raise RuntimeError((result.stderr or "git inspection failed")[:10_000])
        return (result.stdout or "(clean workspace)")[:20_000]
    return execute


def inspect_diagnostics(workspace: Path) -> Callable[..., str]:
    root = resolve_workspace_path(workspace, ".")
    def execute(path: str = ".") -> str:
        diagnostics = []
        for file in _files(root, path):
            if file.suffix != ".py": continue
            try: ast.parse(file.read_text(encoding="utf-8", errors="replace"), filename=str(file))
            except SyntaxError as exc: diagnostics.append({"path": file.relative_to(root).as_posix(), "line": exc.lineno, "message": exc.msg})
        return json.dumps({"checked": len([f for f in _files(root, path) if f.suffix == ".py"]), "diagnostics": diagnostics}, ensure_ascii=False)
    return execute
