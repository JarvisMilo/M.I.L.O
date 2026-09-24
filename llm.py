"""Proveedores de modelo de lenguaje."""

from __future__ import annotations

import json
from typing import Protocol
from urllib.error import URLError
from urllib.request import Request, urlopen


class LLMProvider(Protocol):
    def respond(self, text: str) -> str:
        """Genera una respuesta de asistente para el texto del usuario."""


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
