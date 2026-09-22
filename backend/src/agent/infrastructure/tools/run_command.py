from pathlib import Path
import subprocess

from ...domain.policies import SafetyPolicy
from ...domain.value_objects import ToolCall


class RunCommand:
    """Only trusted absolute executables are eligible for version queries."""

    def __init__(self, workspace: Path, executables: dict[str, Path], policy: SafetyPolicy) -> None:
        self._workspace = workspace.resolve(strict=True)
        self._executables: dict[str, Path] = {}
        self._policy = policy
        for name, path in executables.items():
            if not path.is_absolute():
                raise ValueError("Executable path must be absolute")
            resolved = path.resolve(strict=True)
            if not resolved.is_file() or resolved.is_relative_to(self._workspace):
                raise ValueError("Executable must be a trusted file outside workspace")
            if resolved.suffix.lower() in (".bat", ".cmd", ".ps1"):
                raise ValueError("Shell scripts are not allowed")
            self._executables[name] = resolved

    def execute(self, arguments: dict[str, str]) -> str:
        self._policy.validate(ToolCall("command", "run_command", tuple(arguments.items())))
        argv = arguments["command"].split()
        if argv[0] not in self._executables:
            raise ValueError("Executable not configured; use --python-executable or --git-executable")
        try:
            result = subprocess.run(
                [str(self._executables[argv[0]]), *argv[1:]],
                cwd=self._workspace, shell=False, timeout=10,
                capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Command timed out after 10 seconds") from exc
        output = result.stdout + result.stderr
        if result.returncode:
            raise RuntimeError(f"Exit code {result.returncode}: {output}")
        return output
