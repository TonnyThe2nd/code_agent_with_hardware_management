import json
from typing import Any

import pytest

from src.agent.infrastructure.llm.structured_actions import StructuredActions
from src.agent.infrastructure.llm import ollama_provider
from src.agent.infrastructure.tools.read_file import SCHEMA
from src.agent.domain.value_objects import Message, MessageRole, ToolCall
from src.agent.domain.policies import SafetyPolicy


def test_valid_action_and_tool_history() -> None:
    protocol = StructuredActions([SCHEMA])
    action = protocol.parse('{"action":"read_file","arguments":{"path":"file.py","max_bytes":20}}')
    call = action.tool_calls[0]
    assert call.name == "read_file"
    assert dict(call.arguments)["max_bytes"] == 20
    messages = protocol.messages([action, Message(MessageRole.TOOL, "hello", tool_call_id=call.id, name=call.name)])
    assert json.loads(messages[1]["content"])["action"] == "read_file"
    assert json.loads(messages[2]["content"])["tool_result"] == "read_file"
    assert protocol.parse('{"action":"final","content":"done"}').content == "done"


@pytest.mark.parametrize("content", [
    'Here is JSON: {"action":"read_file","arguments":{"path":"file"}}',
    '{"action":"apply_patch","arguments":{}}',
    '{"action":"read_file","arguments":{"path":{"type":"string","value":"."}}}',
    '{"action":"read_file","arguments":{"path":"file","max_bytes":true}}',
    '{"action":"read_file","arguments":{"path":"file","max_bytes":0}}',
    '{"action":"read_file","arguments":{}}',
    '{"action":"final","content":"done","arguments":{}}',
])
def test_rejects_invalid_envelopes(content: str) -> None:
    with pytest.raises(ValueError):
        StructuredActions([SCHEMA]).parse(content)


def test_structured_actions_still_require_safety_policy() -> None:
    action = StructuredActions([SCHEMA]).parse('{"action":"read_file","arguments":{"path":"../secret"}}')
    assert not SafetyPolicy().is_allowed(action.tool_calls[0])[0]


def test_policy_allows_safe_targeted_pytest_only_when_enabled() -> None:
    call = ToolCall("1", "run_command", (("command", "python -m pytest tests/agent -q"),))
    assert not SafetyPolicy().is_allowed(call)[0]
    assert SafetyPolicy(allow_tests=True).is_allowed(call)[0]


def test_provider_sends_schema_without_native_tools(monkeypatch: Any) -> None:
    def post(url: str, **kwargs: Any) -> Any:
        payload = kwargs["json"]
        assert "tools" not in payload
        assert "oneOf" in payload["format"]
        assert all(item["properties"]["action"]["const"] != "final" for item in payload["format"]["oneOf"])
        assert payload["options"] == {"num_ctx": 4096, "temperature": 0}
        return ollama_provider.httpx.Response(200, request=ollama_provider.httpx.Request("POST", url),
            json={"message": {"content": '{"action":"read_file","arguments":{"path":"file"}}'}})

    monkeypatch.setattr(ollama_provider.httpx, "post", post)
    provider = ollama_provider.OllamaProvider("fake", tools_schema=[SCHEMA], num_ctx=4096, action_mode="structured")
    assert provider.chat([Message(MessageRole.USER, "read")]).tool_calls[0].name == "read_file"
