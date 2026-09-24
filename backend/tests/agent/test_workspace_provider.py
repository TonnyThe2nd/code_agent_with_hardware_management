from pathlib import Path

import pytest

from src.agent.application.workspace_provider import WorkspaceProvider
from src.agent.application.use_cases import RunAgentSessionUseCase
from src.agent.domain.value_objects import Message, MessageRole, ToolCall
from src.agent.domain.policies import SafetyPolicy
from src.agent.infrastructure.tools import build_default_registry


class FakeProvider:
    def __init__(self, call: ToolCall) -> None:
        self.call = call
        self.count = 0

    def complete(self, messages: tuple[Message, ...]) -> Message:
        self.count += 1
        return (Message(MessageRole.ASSISTANT, tool_calls=(self.call,)) if self.count == 1
                else Message(MessageRole.ASSISTANT, "Done"))


def test_absolute_workspace_root_is_listed(tmp_path: Path) -> None:
    call = ToolCall("1", "list_dir", {"path": str(tmp_path)})
    result = RunAgentSessionUseCase(FakeProvider(call), build_default_registry(tmp_path), tmp_path,
                                   fail_on_tool_error=True, require_tool_activity=True).execute("List files")
    assert result.final_message == "Done"


def test_normalizes_write_path_without_modifying_content(tmp_path: Path) -> None:
    call = ToolCall("1", "write_file", {"path": str(tmp_path / "file.txt"), "content": str(tmp_path)})
    normalized = WorkspaceProvider(FakeProvider(call), tmp_path).complete(())
    assert dict(normalized.tool_calls[0].arguments) == {"path": "file.txt", "content": str(tmp_path)}


def test_external_and_traversal_paths_stay_blocked(tmp_path: Path) -> None:
    for path in (str(tmp_path.parent / "outside.txt"), str(tmp_path / ".." / "outside.txt"), "../outside.txt"):
        call = ToolCall("1", "read_file", {"path": path})
        normalized = WorkspaceProvider(FakeProvider(call), tmp_path).complete(())
        assert not SafetyPolicy().is_allowed(normalized.tool_calls[0])[0]


def test_error_reports_rejected_path(tmp_path: Path) -> None:
    call = ToolCall("1", "read_file", {"path": "../outside.txt"})
    with pytest.raises(RuntimeError, match="outside.txt"):
        RunAgentSessionUseCase(FakeProvider(call), build_default_registry(tmp_path), tmp_path,
                              fail_on_tool_error=True).execute("Read")
