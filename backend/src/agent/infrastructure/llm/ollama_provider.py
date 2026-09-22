import json
from typing import Any
from uuid import uuid4

import httpx

from ...domain.value_objects import Message, MessageRole, ToolCall


class OllamaProvider:
    def __init__(self, client: httpx.Client, model: str) -> None:
        if not model.strip():
            raise ValueError("Model must not be empty")
        self._client = client
        self._model = model

    def complete(self, messages: tuple[Message, ...]) -> Message:
        response = self._client.post("/api/chat", json={
            "model": self._model, "stream": False,
            "messages": [self._serialize(message) for message in messages],
            "tools": self._tool_schemas(),
        })
        response.raise_for_status()
        raw = response.json()["message"]
        if raw.get("role") != "assistant":
            raise ValueError("Unexpected Ollama message role")
        calls: list[ToolCall] = []
        for item in raw.get("tool_calls") or []:
            function = item["function"]
            arguments = function["arguments"]
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must be an object")
            calls.append(ToolCall(item.get("id") or str(uuid4()), function["name"],
                                  tuple(arguments.items())))
        return Message(MessageRole.ASSISTANT, raw.get("content") or "", tuple(calls))

    @staticmethod
    def _serialize(message: Message) -> dict[str, Any]:
        data: dict[str, Any] = {"role": message.role.value, "content": message.content}
        if message.tool_calls:
            data["tool_calls"] = [
                {"id": call.id, "type": "function", "function": {
                    "name": call.name, "arguments": dict(call.arguments),
                }} for call in message.tool_calls
            ]
        if message.role is MessageRole.TOOL:
            data["tool_name"] = message.name
            data["tool_call_id"] = message.tool_call_id
        return data

    @staticmethod
    def _tool_schemas() -> list[dict[str, Any]]:
        definitions = (
            ("read_file", "Read a UTF-8 workspace file", ("path",)),
            ("write_file", "Write a UTF-8 workspace file", ("path", "content")),
            ("list_dir", "List a workspace directory; use . for root", ("path",)),
            ("run_command", "Run python --version or git --version if configured", ("command",)),
        )
        return [{"type": "function", "function": {
            "name": name, "description": description,
            "parameters": {"type": "object", "properties": {
                key: {"type": "string"} for key in keys
            }, "required": list(keys), "additionalProperties": False},
        }} for name, description, keys in definitions]
