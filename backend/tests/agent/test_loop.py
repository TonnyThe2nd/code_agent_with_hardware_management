import json
from src.agent.infrastructure.llm import ollama_provider
import subprocess
from typing import Any

from src.agent.infrastructure.tools import build_default_registry
from src.agent.infrastructure.tools.read_file import read_file
from src.agent.infrastructure.tools.write_file import write_file
from src.agent.infrastructure.tools.list_dir import list_dir
from src.agent.infrastructure.tools.run_command import run_command
from src.agent.domain.entities import AgentSession
from src.agent.domain.policies import SafetyPolicy
from src.agent.domain.services import AgentLoop
from src.agent.domain.value_objects import Message, MessageRole, ToolCall, ToolResult


def test_cli_defaults_and_rich_output(monkeypatch: Any) -> None:
    from typer.testing import CliRunner
    from src.main import app
    from src.agent.presentation import cli

    printed: list[tuple[str, dict[str, Any]]] = []
    configured: dict[str, Any] = {}

    class FakeConsole:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def print(self, text: str, **kwargs: Any) -> None:
            printed.append((text, kwargs))

    class Registry:
        def schemas(self) -> list[dict[str, Any]]:
            return [{"type": "function", "function": {"name": "read_file"}}]

        def execute(self, call: ToolCall) -> ToolResult:
            return ToolResult(call.id, call.name, "x" * 250)

    def registry_factory(workspace: Path, **kwargs: Any) -> Registry:
        configured["workspace"] = workspace
        return Registry()

    def provider_factory(model: str, tools_schema: list[dict[str, Any]], **kwargs: Any) -> FakeLLM:
        configured["model"] = model
        configured["schemas"] = tools_schema
        return FakeLLM([
            Message(MessageRole.ASSISTANT, "[bold]Reading[/bold]", (
                ToolCall("1", "read_file", (("path", "file.txt"),)),
            )),
            Message(MessageRole.ASSISTANT, "Done"),
        ])

    monkeypatch.setattr(cli, "Console", FakeConsole)
    monkeypatch.setattr(cli, "build_default_registry", registry_factory)
    monkeypatch.setattr(cli, "OllamaProvider", provider_factory)
    class Catalog:
        def __init__(self, base_url: str) -> None:
            pass

        def select(self, requested: str | None) -> str:
            return requested or "installed:7b"

    monkeypatch.setattr(cli, "OllamaCatalog", Catalog)
    result = CliRunner().invoke(app, ["agent", "run", "Read file", "-m", "installed:7b"])
    assert result.exit_code == 0, result.output
    assert configured["model"] == "installed:7b"
    assert configured["workspace"] == Path.cwd().resolve()
    assert configured["schemas"][0]["function"]["name"] == "read_file"
    assert [options["style"] for _, options in printed] == ["cyan", "yellow", "dim", "cyan", "green"]
    assert printed[1][0] == '→ tool: read_file({"path": "file.txt"})'
    assert printed[2][0] == "x" * 200
    assert all(options.get("markup") is False for _, options in printed)
    assert printed[-1][0] == "Done"

    printed.clear()
    result = CliRunner().invoke(app, ["agent", "run", "Read file", "-w", ".", "-m", "custom", "--max-iter", "1"])
    assert result.exit_code == 2
    assert configured["model"] == "custom"
    assert "Limite" in printed[-1][0]


class FakeLLM:
    def __init__(self, responses: list[Message]) -> None:
        self.responses = responses
        self.histories: list[tuple[Message, ...]] = []

    def complete(self, messages: tuple[Message, ...]) -> Message:
        self.histories.append(messages)
        return self.responses[len(self.histories) - 1]


class FakeTools:
    def __init__(self) -> None:
        self.calls: list[ToolCall] = []

    def execute(self, call: ToolCall) -> ToolResult:
        self.calls.append(call)
        return ToolResult(call.id, call.name, "file contents")


def test_loop_executes_tool_and_finishes() -> None:
    call = ToolCall("1", "read_file", (("path", "hello.txt"),))
    llm = FakeLLM([Message(MessageRole.ASSISTANT, tool_calls=(call,)),
                   Message(MessageRole.ASSISTANT, "Done")])
    tools = FakeTools()
    session = AgentSession(messages=[Message(MessageRole.USER, "Read hello.txt")])
    AgentLoop(llm, tools, SafetyPolicy()).run(session)
    assert tools.calls == [call]
    assert session.completed
    assert len(session.turns) == 2
    assert llm.histories[1][-1] == Message(
        MessageRole.TOOL, "file contents", tool_call_id="1", name="read_file",
    )
    assert session.messages[-1].content == "Done"


def test_safety_policy_blocks_path_traversal() -> None:
    for path in ("../secret", "sub/../../secret", "..\\secret", "/etc/passwd", "C:\\secret"):
        call = ToolCall("1", "read_file", (("path", path),))
        try:
            SafetyPolicy().validate(call)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Unsafe path accepted: {path}")
        llm = FakeLLM([Message(MessageRole.ASSISTANT, tool_calls=(call,)),
                       Message(MessageRole.ASSISTANT, "Blocked")])
        tools = FakeTools()
        session = AgentSession()
        AgentLoop(llm, tools, SafetyPolicy()).run(session)
        assert tools.calls == []
        assert session.turns[0].results[0].is_error


def test_loop_stops_at_iteration_limit() -> None:
    call = ToolCall("1", "list_dir", (("path", "."),))
    llm = FakeLLM([Message(MessageRole.ASSISTANT, tool_calls=(call,))])
    session = AgentSession()
    AgentLoop(llm, FakeTools(), SafetyPolicy()).run(session, max_iterations=1)
    assert not session.completed
    assert len(session.turns) == 1


def test_policy_blocks_dangerous_commands_and_unknown_tools() -> None:
    calls = [ToolCall("1", "unknown")]
    calls.extend(ToolCall("1", "run_command", (("command", command),)) for command in (
        "rm -rf /", "format C:", "mkfs /dev/sda", ":(){ :|:& };:",
        "shutdown", "reboot", "git --version; shutdown", "python -c 'print(1)'",
    ))
    for call in calls:
        try:
            SafetyPolicy().validate(call)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Unsafe call accepted: {call}")


def test_use_case_returns_dto_and_notifies_between_iterations() -> None:
    call = ToolCall("1", "read_file", (("path", "hello.txt"),))
    llm = FakeLLM([
        Message(MessageRole.ASSISTANT, tool_calls=(call,)),
        Message(MessageRole.ASSISTANT, "Done"),
    ])
    notifications: list[tuple[int, Message]] = []

    def on_message(message: Message) -> None:
        notifications.append((len(llm.histories), message))

    workspace = Path.cwd()
    result = RunAgentSessionUseCase(llm, FakeTools(), workspace, 3).execute(
        "Read hello.txt", on_message,
    )
    assert result.to_dict() == {
        "session_id": result.session_id,
        "final_message": "Done",
        "iterations": 2,
        "messages_count": 5,
    }
    assert result.session_id
    assert [count for count, _ in notifications] == [1, 1, 2]
    assert [message.role for _, message in notifications] == [
        MessageRole.ASSISTANT, MessageRole.TOOL, MessageRole.ASSISTANT,
    ]
    assert llm.histories[0][0].content.startswith(DEFAULT_SYSTEM_PROMPT)
    assert str(workspace.resolve()) in llm.histories[0][0].content
    assert llm.histories[0][1] == Message(MessageRole.USER, "Read hello.txt")


def test_use_case_reports_iteration_exhaustion() -> None:
    call = ToolCall("1", "list_dir", (("path", "."),))
    llm = FakeLLM([Message(MessageRole.ASSISTANT, "Checking", tool_calls=(call,))])
    result = RunAgentSessionUseCase(llm, FakeTools(), Path.cwd(), 1).execute("List files")
    assert result.final_message is None
    assert result.iterations == 1
    assert result.messages_count == 4


def test_use_case_creates_a_fresh_session_for_each_run() -> None:
    llm = FakeLLM([Message(MessageRole.ASSISTANT, ""), Message(MessageRole.ASSISTANT, "Done")])
    use_case = RunAgentSessionUseCase(llm, FakeTools(), Path.cwd())
    first = use_case.execute("First")
    second = use_case.execute("Second")
    assert first.session_id != second.session_id
    assert first.final_message == ""
    assert first.iterations == second.iterations == 1
    assert first.messages_count == second.messages_count == 3
    assert llm.histories[1][1].content == "Second"


def test_use_case_rejects_blank_prompt_before_calling_llm() -> None:
    llm = FakeLLM([])
    use_case = RunAgentSessionUseCase(llm, FakeTools(), Path.cwd())
    try:
        use_case.execute("  ")
    except ValueError:
        pass
    else:
        raise AssertionError("Blank prompt accepted")
    assert llm.histories == []
from pathlib import Path

from src.agent.application.use_cases import DEFAULT_SYSTEM_PROMPT, RunAgentSessionUseCase


def test_tool_factories_and_registry(tmp_path: Path) -> None:
    registry = build_default_registry(tmp_path)
    result = registry.execute(ToolCall("1", "write_file", (("path", "nested/file.txt"), ("content", "abcdef"))))
    assert not result.is_error
    result = registry.execute(ToolCall("2", "read_file", (("path", "nested/file.txt"), ("max_bytes", 3))))
    assert result.content == "abc"
    assert json.loads(registry.execute(ToolCall("3", "list_dir")).content) == ["nested/"]
    assert registry.execute(ToolCall("4", "unknown")).is_error
    assert registry.execute(ToolCall("5", "read_file")).is_error
    schemas = registry.schemas()
    assert len(schemas) == 10
    schemas[0]["function"]["name"] = "changed"
    assert registry.schemas()[0]["function"]["name"] != "changed"
    assert SafetyPolicy().is_allowed(ToolCall("6", "read_file", (("path", "file"), ("max_bytes", 3)))) == (True, None)


def test_factories_reject_external_paths(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for path in ("../outside", "..\\outside", str(tmp_path / "outside")):
        operations = (
            lambda: read_file(workspace)(path),
            lambda: write_file(workspace)(path, "blocked"),
            lambda: list_dir(workspace)(path),
            lambda: run_command(workspace)("echo blocked", cwd=path),
        )
        for operation in operations:
            try:
                operation()
            except PermissionError:
                pass
            else:
                raise AssertionError("External path accepted")
    assert not (tmp_path / "outside").exists()


def test_developer_inspection_tools_offer_context_and_diagnostics(tmp_path: Path) -> None:
    (tmp_path / "package").mkdir()
    (tmp_path / "package" / "service.py").write_text("class Service:\n    pass\n", encoding="utf-8")
    registry = build_default_registry(tmp_path)
    code = registry.execute(ToolCall("1", "search_code", (("query", "Service"),)))
    assert not code.is_error and '"line": 1' in code.content
    symbol = registry.execute(ToolCall("2", "find_symbol", (("symbol", "Service"),)))
    assert not symbol.is_error and "service.py" in symbol.content
    excerpt = registry.execute(ToolCall("3", "read_file_range", (("path", "package/service.py"), ("start_line", 1), ("end_line", 1))))
    assert excerpt.content == "1: class Service:"
    diagnostics = registry.execute(ToolCall("4", "inspect_diagnostics"))
    assert not diagnostics.is_error and '"diagnostics": []' in diagnostics.content
    assert not registry.execute(ToolCall("5", "git_status")).is_error


def test_command_contract_and_truncation(tmp_path: Path, monkeypatch: Any) -> None:
    subdirectory = tmp_path / "sub"
    subdirectory.mkdir()

    def fake_run(command: str, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert command[-1] == "--version"
        assert kwargs["shell"] is False
        assert kwargs["cwd"] == subdirectory.resolve()
        assert kwargs["timeout"] == 10
        return subprocess.CompletedProcess(command, 0, "x" * 12_000, "stderr")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert run_command(tmp_path)("python --version", cwd="sub") == "x" * 10_000


def test_registry_reports_timeout(tmp_path: Path, monkeypatch: Any) -> None:
    def fake_run(command: str, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = build_default_registry(tmp_path).execute(ToolCall("1", "run_command", (("command", "python --version"),)))
    assert result.is_error and "timed out" in result.content


def test_ollama_chat_payload_and_parsing(monkeypatch: Any) -> None:
    schemas = [{"type": "function", "function": {"name": "read_file"}}]
    messages = [Message(MessageRole.USER, "Read file")]

    def post(url: str, **kwargs: Any) -> Any:
        assert url == "http://localhost:11434/api/chat"
        assert kwargs["timeout"] == 3.0
        assert kwargs["json"] == {
            "model": "fake", "messages": [message.to_dict() for message in messages],
            "stream": False, "tools": schemas,
        }
        return ollama_provider.httpx.Response(200, request=ollama_provider.httpx.Request("POST", url), json={
            "message": {"content": None, "tool_calls": [
                {"id": "known", "function": {"name": "read_file", "arguments": {"path": "file", "max_bytes": 3}}},
                {"function": {"name": "list_dir", "arguments": "{}"}},
            ]},
        })

    monkeypatch.setattr(ollama_provider.httpx, "post", post)
    result = ollama_provider.OllamaProvider("fake", "http://localhost:11434/", schemas, 3.0).chat(messages)
    assert result.role is MessageRole.ASSISTANT and result.content == ""
    assert result.tool_calls[0].id == "known"
    assert dict(result.tool_calls[0].arguments) == {"path": "file", "max_bytes": 3}
    assert result.tool_calls[1].id and dict(result.tool_calls[1].arguments) == {}
    assert result.to_dict()["tool_calls"][0]["function"]["arguments"]["max_bytes"] == 3


def test_ollama_omits_empty_tools_and_supports_complete(monkeypatch: Any) -> None:
    def post(url: str, **kwargs: Any) -> Any:
        assert "tools" not in kwargs["json"]
        return ollama_provider.httpx.Response(200, request=ollama_provider.httpx.Request("POST", url),
                                             json={"message": {"content": "Done"}})

    monkeypatch.setattr(ollama_provider.httpx, "post", post)
    assert ollama_provider.OllamaProvider("fake").complete(()).content == "Done"


def test_ollama_propagates_http_status_errors(monkeypatch: Any) -> None:
    for status in (302, 400, 500):
        def post(url: str, **kwargs: Any) -> Any:
            return ollama_provider.httpx.Response(status, request=ollama_provider.httpx.Request("POST", url))

        monkeypatch.setattr(ollama_provider.httpx, "post", post)
        try:
            ollama_provider.OllamaProvider("fake").chat([])
        except ollama_provider.httpx.HTTPStatusError as exc:
            assert exc.response.status_code == status
        else:
            raise AssertionError("HTTP failure was swallowed")


def test_registry_reports_exit_code(tmp_path: Path, monkeypatch: Any) -> None:
    def fake_run(command: str, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 2, "", "error" * 3_000)

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = build_default_registry(tmp_path).execute(ToolCall("1", "run_command", (("command", "python --version"),)))
    assert result.is_error and result.content.startswith("Exit code 2:")
    assert len(result.content) == 10_000
