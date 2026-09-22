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
