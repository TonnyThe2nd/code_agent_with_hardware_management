from pathlib import Path
from typing import Callable

import pytest

from src.agent.application.dto import AgentRunResult, RoutingPlanDTO, SubtaskDTO
from src.agent.application.execute_plan import ExecuteRoutingPlanUseCase
from src.agent.application.use_cases import RunAgentSessionUseCase
from src.agent.domain.value_objects import Message, MessageRole, ToolCall
from src.agent.infrastructure.tools import build_default_registry


def plan() -> RoutingPlanDTO:
    return RoutingPlanDTO("plan", "Implement", [
        SubtaskDTO("1", "Write file", "simple", "file", "small"),
        SubtaskDTO("2", "Review file", "complex", "review", "large", ("1",)),
    ], 10, ["small", "large"], [])


def test_execution_writes_file_and_hands_off_dependencies(tmp_path: Path) -> None:
    calls: list[str] = []
    prompts: list[str] = []

    class LLM:
        def __init__(self, model: str) -> None:
            self.model = model
            self.count = 0

        def complete(self, messages: tuple[Message, ...]) -> Message:
            self.count += 1
            if self.count == 1:
                prompts.append(messages[1].content)
            if self.model == "small" and self.count == 1:
                return Message(MessageRole.ASSISTANT, tool_calls=(ToolCall(
                    "write", "write_file", {"path": "result.txt", "content": "hello"},
                ),))
            return Message(MessageRole.ASSISTANT, "Wrote result.txt" if self.model == "small" else "Reviewed")

    registry = build_default_registry(tmp_path)

    def factory(model: str) -> RunAgentSessionUseCase:
        calls.append(model)
        return RunAgentSessionUseCase(LLM(model), registry, tmp_path, fail_on_tool_error=True)

    result = ExecuteRoutingPlanUseCase(factory).execute(plan())
    assert result.completed
    assert calls == ["small", "large"]
    assert (tmp_path / "result.txt").read_text() == "hello"
    assert "Wrote result.txt" in prompts[1]


def test_iteration_limit_stops_dependents() -> None:
    calls: list[str] = []

    class Runner:
        def execute(self, user_prompt: str, on_message: Callable[[Message], None] | None = None) -> AgentRunResult:
            return AgentRunResult("session", None, 1, 4)

    def factory(model: str) -> Runner:
        calls.append(model)
        return Runner()

    result = ExecuteRoutingPlanUseCase(factory).execute(plan())
    assert not result.completed and result.failed_subtask == "1"
    assert calls == ["small"]


def test_invalid_plan_rejected_before_factory() -> None:
    invalid = plan()
    invalid.subtasks.reverse()

    def factory(model: str) -> RunAgentSessionUseCase:
        raise AssertionError("Factory must not be invoked")

    with pytest.raises(ValueError):
        ExecuteRoutingPlanUseCase(factory).execute(invalid)


def test_tool_failure_stops_execution(tmp_path: Path) -> None:
    class LLM:
        def complete(self, messages: tuple[Message, ...]) -> Message:
            return Message(MessageRole.ASSISTANT, tool_calls=(
                ToolCall("bad", "read_file", {"path": "missing.txt"}),
            ))

    def factory(model: str) -> RunAgentSessionUseCase:
        return RunAgentSessionUseCase(LLM(), build_default_registry(tmp_path), tmp_path,
                                     fail_on_tool_error=True)

    result = ExecuteRoutingPlanUseCase(factory).execute(plan())
    assert not result.completed and result.failed_subtask == "1"
    assert "tool failed" in (result.error or "")
