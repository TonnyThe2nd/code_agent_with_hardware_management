from pathlib import Path

import pytest

from src.agent.application.use_cases import RunAgentSessionUseCase
from src.agent.domain.value_objects import Message, MessageRole, ToolCall
from src.agent.infrastructure.tools import build_default_registry


class FakeLLM:
    def __init__(self, responses: list[Message]) -> None:
        self.responses = iter(responses)
        self.calls = 0

    def complete(self, messages: tuple[Message, ...]) -> Message:
        self.calls += 1
        return next(self.responses)


def test_textual_tool_json_is_not_execution(tmp_path: Path) -> None:
    fake = Message(MessageRole.ASSISTANT, '{"name":"write_file","arguments":{"path":"fake.py","content":"hi"}}')
    llm = FakeLLM([fake, fake])
    use_case = RunAgentSessionUseCase(llm, build_default_registry(tmp_path), tmp_path, require_tool_activity=True)
    with pytest.raises(RuntimeError, match="sem executar ferramentas"):
        use_case.execute("Write fake.py")
    assert llm.calls == 2
    assert not (tmp_path / "fake.py").exists()


def test_correction_can_lead_to_real_write(tmp_path: Path) -> None:
    llm = FakeLLM([
        Message(MessageRole.ASSISTANT, "Here is a patch"),
        Message(MessageRole.ASSISTANT, tool_calls=(ToolCall("1", "write_file", {"path": "real.py", "content": "print('ok')"}),)),
        Message(MessageRole.ASSISTANT, "Saved"),
    ])
    registry = build_default_registry(tmp_path)
    result = RunAgentSessionUseCase(llm, registry, tmp_path, require_tool_activity=True).execute("Write real.py")
    assert result.final_message == "Saved"
    assert result.iterations == 3
    assert registry.changed_files == {"real.py"}
    assert (tmp_path / "real.py").read_text() == "print('ok')"


def test_same_content_does_not_claim_change(tmp_path: Path) -> None:
    (tmp_path / "existing.txt").write_text("same", encoding="utf-8")
    registry = build_default_registry(tmp_path)
    result = registry.execute(ToolCall("1", "write_file", {"path": "existing.txt", "content": "same"}))
    assert not result.is_error
    assert not registry.changed_files


def test_recoverable_tool_error_is_returned_to_model(tmp_path: Path) -> None:
    llm = FakeLLM([
        Message(MessageRole.ASSISTANT, tool_calls=(ToolCall("1", "read_file", {"path": "missing.py"}),)),
        Message(MessageRole.ASSISTANT, tool_calls=(ToolCall("2", "list_dir", {"path": "."}),)),
        Message(MessageRole.ASSISTANT, "Recovered"),
    ])
    result = RunAgentSessionUseCase(
        llm, build_default_registry(tmp_path), tmp_path,
        fail_on_tool_error=False, require_tool_activity=True,
    ).execute("Inspect workspace")
    assert result.final_message == "Recovered"
    assert result.iterations == 3
