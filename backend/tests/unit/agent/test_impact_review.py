from pathlib import Path

from src.agent.application.impact_review import ReviewImpactUseCase
from src.agent.application.use_cases import RunAgentSessionUseCase
from src.agent.domain.value_objects import Message, MessageRole, ToolCall, ToolResult
from src.agent.infrastructure.tools.registry import ToolActivity


class Source:
    changed_files = {"src/service.py"}
    activities = [
        ToolActivity("git_diff", False, "diff"),
        ToolActivity("inspect_diagnostics", False, '{"diagnostics": []}'),
        ToolActivity("run_command", False, "1 passed"),
    ]


def test_impact_review_requires_real_review_and_collects_validation() -> None:
    review = ReviewImpactUseCase().execute(Source())
    assert review.ready
    assert review.changed_files == ("src/service.py",)
    assert review.validation == ('{"diagnostics": []}', "1 passed")


def test_session_runs_diagnostics_after_a_successful_write() -> None:
    class LLM:
        responses = [
            Message(MessageRole.ASSISTANT, tool_calls=(ToolCall("write", "write_file", {"path": "x.py", "content": "x"}),)),
            Message(MessageRole.ASSISTANT, "implementation complete"),
            Message(MessageRole.ASSISTANT, "final report with validation"),
        ]
        def __init__(self) -> None: self.index = 0
        def complete(self, messages: tuple[Message, ...]) -> Message:
            response = self.responses[self.index]
            self.index += 1
            return response

    class Tools:
        def __init__(self) -> None: self.calls: list[str] = []
        def execute(self, call: ToolCall) -> ToolResult:
            self.calls.append(call.name)
            return ToolResult(call.id, call.name, '{"diagnostics": []}')

    tools = Tools()
    result = RunAgentSessionUseCase(LLM(), tools, Path.cwd(), max_iterations=4).execute("Change x")
    assert result.final_message == "final report with validation"
    assert tools.calls == ["write_file", "inspect_diagnostics"]
