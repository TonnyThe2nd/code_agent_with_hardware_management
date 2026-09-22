from pathlib import Path
from typing import Any
import subprocess

from src.agent.infrastructure.llm.ollama_provider import OllamaCatalog
from src.agent.infrastructure.tools.run_command import run_command
from src.agent.domain.policies import SafetyPolicy
from src.agent.domain.value_objects import ToolCall


def test_model_selection_checks_capabilities_and_ambiguity(monkeypatch: Any) -> None:
    catalog = OllamaCatalog()
    installed = [
        {"name": "embed:latest", "capabilities": ["embedding"]},
        {"name": "first:latest", "capabilities": ["completion", "tools"]},
    ]
    monkeypatch.setattr(catalog, "models", lambda: installed)
    assert catalog.select(None) == "first:latest"
    assert catalog.select("first") == "first:latest"
    for requested in ("missing", "embed"):
        try:
            catalog.select(requested)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid model accepted")
    installed.append({"name": "second:7b", "capabilities": ["tools"]})
    assert catalog.select("second:7b") == "second:7b"
    try:
        catalog.select(None)
    except ValueError as exc:
        assert "--model" in str(exc)
    else:
        raise AssertionError("Ambiguous model selected silently")


def test_project_tests_require_explicit_opt_in(tmp_path: Path) -> None:
    call = ToolCall("1", "run_command", {"command": "python -m pytest"})
    assert not SafetyPolicy().is_allowed(call)[0]
    assert SafetyPolicy(allow_tests=True).is_allowed(call)[0]
    try:
        run_command(tmp_path)("python -m pytest")
    except PermissionError:
        pass
    else:
        raise AssertionError("Host test execution accepted")


def test_docker_isolation_and_cleanup(tmp_path: Path, monkeypatch: Any) -> None:
    import shutil

    commands: list[list[str]] = []
    monkeypatch.setattr(shutil, "which", lambda name: str(tmp_path.parent / "docker.exe"))

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        commands.append(argv)
        if "run" in argv:
            assert kwargs["shell"] is False
            assert "--network=none" in argv and "--read-only" in argv
            assert "--user=65534:65534" in argv and "--cap-drop=ALL" in argv
            assert "--pull=never" in argv
            assert any("target=/workspace,readonly" in argument for argument in argv)
            raise subprocess.TimeoutExpired(argv, 10)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    try:
        run_command(tmp_path, allow_tests=True)("python -m pytest")
    except RuntimeError as exc:
        assert "timed out" in str(exc)
    else:
        raise AssertionError("Timeout was swallowed")
    assert commands[-1][1:3] == ["rm", "-f"]
    assert commands[-1][-1] == commands[0][commands[0].index("--name") + 1]


def test_no_shell_injection_even_when_calling_tool_directly(tmp_path: Path) -> None:
    for command in ("python --version & echo unsafe", "python -c print(1)", "rm -rf /"):
        try:
            run_command(tmp_path, allow_tests=True)(command)
        except PermissionError:
            pass
        else:
            raise AssertionError("Unapproved command executed")
