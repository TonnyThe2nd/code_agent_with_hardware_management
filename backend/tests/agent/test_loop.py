from src.agent.domain.entities import AgentSession
from src.agent.domain.policies import SafetyPolicy
from src.agent.domain.services import AgentLoop
from src.agent.domain.value_objects import Message, MessageRole, ToolCall, ToolResult


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
