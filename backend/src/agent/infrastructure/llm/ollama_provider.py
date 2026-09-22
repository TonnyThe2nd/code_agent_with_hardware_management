import json
from typing import Any
from uuid import uuid4

import httpx

from ...domain.value_objects import Message, MessageRole, ToolCall


class OllamaCatalog:
    def __init__(self, base_url: str = "http://localhost:11434", timeout: float = 10.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def models(self) -> list[dict[str, Any]]:
        response = httpx.get(f"{self._base_url}/api/tags", timeout=self._timeout)
        response.raise_for_status()
        models: list[dict[str, Any]] = []
        for item in response.json()["models"]:
            info = httpx.post(f"{self._base_url}/api/show", json={"model": item["name"]}, timeout=self._timeout)
            info.raise_for_status()
            models.append({"name": item["name"], "size": item.get("size", 0),
                           "capabilities": info.json().get("capabilities", [])})
        return models

    def select(self, requested: str | None) -> str:
        models = self.models()
        if requested:
            match = next((item for item in models if item["name"] == requested
                          or item["name"] == requested + ":latest"), None)
            if match is None:
                raise ValueError(f"Modelo nao instalado: {requested}. Instale com ollama pull {requested}.")
            if "tools" not in match["capabilities"]:
                raise ValueError(f"Modelo {requested} nao declara suporte a ferramentas. Consulte agent models.")
            return str(match["name"])
        candidates = [str(item["name"]) for item in models if "tools" in item["capabilities"]]
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise ValueError("Nenhum modelo com suporte a ferramentas instalado. Consulte agent models.")
        raise ValueError("Escolha --model ou CODE_AGENT_MODEL entre: " + ", ".join(candidates))


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
