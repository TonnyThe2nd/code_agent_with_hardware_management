from pathlib import Path
import subprocess
import shutil
import sys
import shlex
from typing import Any, Protocol
from uuid import uuid4

from .registry import resolve_workspace_path


SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "run_command",
        "description": "Run python --version or git --version; python -m pytest requires enabled Docker isolation.",
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


def run_command(
    workspace: Path, timeout: float = 10, allow_tests: bool = False,
    test_image: str = "code-agent-tests:local",
) -> RunCommandFunction:
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
            words = tuple(shlex.split(command))
        except ValueError as exc:
            raise ValueError("Invalid command syntax") from exc
        container_name: str | None = None
        docker: str | None = None
        if words == ("python", "--version"):
            argv = [sys.executable, "-I", "--version"]
        elif words == ("git", "--version"):
            executable = shutil.which("git")
            if executable is None or Path(executable).resolve().is_relative_to(root):
                raise PermissionError("Git must be installed outside the workspace")
            argv = [executable, "--version"]
        elif words[:3] == ("python", "-m", "pytest") and allow_tests and all(
            not item.startswith("-") or item == "-q" for item in words[3:]
        ):
            docker = shutil.which("docker")
            if docker is None or Path(docker).resolve().is_relative_to(root):
                raise RuntimeError("Docker nao encontrado. Instale Docker Desktop e construa a imagem de testes.")
            if not test_image or test_image.startswith("-"):
                raise ValueError("Invalid test image")
            container_name = "code-agent-" + uuid4().hex
            container_cwd = "/workspace"
            relative = workdir.relative_to(root).as_posix()
            if relative != ".":
                container_cwd += "/" + relative
            argv = [docker, "run", "--rm", "--pull=never", "--name", container_name,
                    "--network=none", "--read-only", "--cap-drop=ALL",
                    "--security-opt=no-new-privileges", "--pids-limit=128",
                    "--memory=2g", "--cpus=2", "--user=65534:65534",
                    "--tmpfs", "/tmp:rw,nosuid,size=256m",
                    "--mount", f"type=bind,source={root},target=/workspace,readonly",
                    "--workdir", container_cwd, "--env", "PYTHONDONTWRITEBYTECODE=1",
                    "--entrypoint", "python", test_image, "-B", "-m", "pytest",
                    "-p", "no:cacheprovider", *words[3:]]
        else:
            raise PermissionError("Command blocked. Use version queries or enable isolated pytest with --allow-tests.")
        try:
            result = subprocess.run(
                argv, shell=False, cwd=workdir, timeout=timeout,
                capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"Command timed out after {timeout} seconds") from exc
        finally:
            if container_name is not None and docker is not None:
                subprocess.run([docker, "rm", "-f", container_name], capture_output=True, timeout=10)
        output = result.stdout + result.stderr
        if result.returncode:
            raise RuntimeError(f"Exit code {result.returncode}: {output}"[:10_000])
        return output[:10_000]

    return execute
