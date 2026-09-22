import json
from typing import Any
from uuid import uuid4

import httpx

from ...domain.value_objects import Message, MessageRole, ToolCall


class OllamaProvider:
    def __init__(
        self, model: str, base_url: str = "http://localhost:11434",
        tools_schema: list[dict[str, Any]] | None = None, timeout: float = 120.0,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Model must not be empty")
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("Base URL must not be empty")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout < float("inf"):
            raise ValueError("Timeout must be positive and finite")
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._tools_schema: list[dict[str, Any]] = json.loads(json.dumps(tools_schema or []))
        self._timeout = timeout

    def chat(self, messages: list[Message]) -> Message:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [message.to_dict() for message in messages],
            "stream": False,
        }
        if self._tools_schema:
            payload["tools"] = self._tools_schema
        response = httpx.post(
            f"{self._base_url}/api/chat", json=payload, timeout=self._timeout,
        )
        response.raise_for_status()
        raw = response.json()["message"]
        if not isinstance(raw, dict):
            raise ValueError("Ollama message must be an object")
        calls: list[ToolCall] = []
        for item in raw.get("tool_calls") or []:
            function = item["function"]
            arguments = function["arguments"]
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must be an object")
            calls.append(ToolCall(
                id=item.get("id") or str(uuid4()),
                name=function["name"], arguments=arguments,
            ))
        content = raw.get("content")
        return Message(MessageRole.ASSISTANT, "" if content is None else content, tuple(calls))

    def complete(self, messages: tuple[Message, ...]) -> Message:
        """Adapt chat to the domain's existing LLMProvider contract."""
        return self.chat(list(messages))
