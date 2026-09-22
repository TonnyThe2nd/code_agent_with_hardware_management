from pathlib import Path
import subprocess
from typing import Any, Protocol

from .registry import resolve_workspace_path


SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "run_command",
        "description": "Run a shell command with a workspace working directory and timeout.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "cwd": {"type": "string", "default": "."},
            },
            "required": ["command"], "additionalProperties": False,
        },
    },
}


class RunCommandFunction(Protocol):
    def __call__(self, command: str, cwd: str = ".") -> str: ...


def run_command(workspace: Path, timeout: float = 10) -> RunCommandFunction:
    root = resolve_workspace_path(workspace, ".")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout < float("inf"):
        raise ValueError("Timeout must be positive and finite")

    def execute(command: str, cwd: str = ".") -> str:
        workdir = resolve_workspace_path(root, cwd)
        if not workdir.is_dir():
            raise NotADirectoryError(str(workdir))
        if not isinstance(command, str) or not command.strip():
            raise ValueError("Command must not be empty")
        try:
            result = subprocess.run(
                command, shell=True, cwd=workdir, timeout=timeout,
                capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"Command timed out after {timeout} seconds") from exc
        output = result.stdout + result.stderr
        if result.returncode:
            raise RuntimeError(f"Exit code {result.returncode}: {output}"[:10_000])
        return output[:10_000]

    return execute
