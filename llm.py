"""Proveedores de modelo de lenguaje."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.error import URLError
from urllib.request import Request, urlopen


class LLMProvider(Protocol):
    def respond(self, text: str) -> str:
        """Genera una respuesta de asistente para el texto del usuario."""


@dataclass(frozen=True)
class LLMToolResponse:
    content: str
    tool_calls: list[dict[str, Any]]


class ToolCallingLLM(Protocol):
    def respond_with_tools(self, text: str, tools: list[dict[str, Any]]) -> LLMToolResponse: ...

    def respond_after_tools(self, text: str, response: LLMToolResponse, results: list[dict[str, Any]]) -> str: ...


class OllamaLLM:
    """Cliente mínimo para la API local de Ollama, sin dependencia HTTP extra."""

    def __init__(self, base_url: str, model: str, timeout_seconds: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def respond(self, text: str) -> str:
        payload = json.dumps({"model": self.model, "prompt": text, "stream": False}).encode()
        request = Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.load(response)
        except URLError as error:
            raise RuntimeError(f"No se pudo conectar con Ollama en {self.base_url}.") from error
        answer = body.get("response", "").strip()
        if not answer:
            raise RuntimeError("Ollama devolvió una respuesta vacía.")
        return answer

    def _chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.model, "messages": messages, "stream": False}
        if tools:
            payload["tools"] = tools
        request = Request(f"{self.base_url}/api/chat", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return json.load(response)
        except URLError as error:
            raise RuntimeError(f"No se pudo conectar con Ollama en {self.base_url}.") from error

    def respond_with_tools(self, text: str, tools: list[dict[str, Any]]) -> LLMToolResponse:
        message = self._chat([{"role": "user", "content": text}], tools).get("message", {})
        return LLMToolResponse(message.get("content", "").strip(), list(message.get("tool_calls", [])))

    def respond_after_tools(self, text: str, response: LLMToolResponse, results: list[dict[str, Any]]) -> str:
        messages = [{"role": "user", "content": text}, {"role": "assistant", "content": response.content, "tool_calls": response.tool_calls}]
        messages.extend({"role": "tool", "content": json.dumps(result, ensure_ascii=False)} for result in results)
        answer = self._chat(messages).get("message", {}).get("content", "").strip()
        if not answer:
            raise RuntimeError("Ollama devolvió una respuesta vacía después de usar herramientas.")
        return answer
