import json
from typing import Any
from uuid import uuid4

import httpx

from ...domain.value_objects import Message, MessageRole, ToolCall
from .structured_actions import StructuredActions


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
            metadata = info.json().get("model_info", {})
            architecture = metadata.get("general.architecture", "")
            models.append({"name": item["name"], "size": item.get("size", 0),
                           "parameter_count": metadata.get("general.parameter_count", 0),
                           "context_length": metadata.get(f"{architecture}.context_length", 0),
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
        num_ctx: int | None = None,
        keep_alive: int | None = None,
        action_mode: str = "native",
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
        if num_ctx is not None and (type(num_ctx) is not int or num_ctx < 1):
            raise ValueError("num_ctx must be a positive integer")
        self._num_ctx = num_ctx
        self._keep_alive = keep_alive
        if action_mode not in ("native", "structured"):
            raise ValueError("Unknown action mode")
        self._actions = StructuredActions(self._tools_schema) if action_mode == "structured" else None

    def chat(self, messages: list[Message]) -> Message:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [message.to_dict() for message in messages],
            "stream": False,
        }
        if self._tools_schema:
            payload["tools"] = self._tools_schema
        if self._num_ctx is not None:
            payload["options"] = {"num_ctx": self._num_ctx}
        if self._keep_alive is not None:
            payload["keep_alive"] = self._keep_alive
        if self._actions is not None:
            payload.pop("tools", None)
            payload["format"] = self._actions.schema(allow_final=any(
                message.role is MessageRole.TOOL for message in messages
            ))
            payload["messages"] = self._actions.messages(messages)
            payload.setdefault("options", {})["temperature"] = 0
        response = httpx.post(
            f"{self._base_url}/api/chat", json=payload, timeout=self._timeout,
        )
        response.raise_for_status()
        raw = response.json()["message"]
        if not isinstance(raw, dict):
            raise ValueError("Ollama message must be an object")
        if self._actions is not None:
            return self._actions.parse(raw.get("content", ""))
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
