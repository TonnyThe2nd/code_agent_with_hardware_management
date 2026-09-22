import json
from typing import Any
from uuid import uuid4

import httpx

from ...domain.value_objects import Message, MessageRole, ToolCall


class OllamaProvider:
    def __init__(
        self, client: httpx.Client, model: str,
        schemas: list[dict[str, Any]] | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("Model must not be empty")
        self._client = client
        self._model = model
        self._schemas: list[dict[str, Any]] = json.loads(json.dumps(schemas or []))

    def complete(self, messages: tuple[Message, ...]) -> Message:
        response = self._client.post("/api/chat", json={
            "model": self._model, "stream": False,
            "messages": [self._serialize(message) for message in messages],
            "tools": self._schemas,
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
