"""Proveedores de speech-to-text."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class STTProvider(Protocol):
    def transcribe(self, audio_path: Path) -> str:
        """Devuelve la transcripción del WAV indicado."""


class FasterWhisperSTT:
    """Adaptador lazy para que importar el programa no cargue el modelo."""

    def __init__(self, model_name: str, device: str = "auto", compute_type: str = "int8") -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def transcribe(self, audio_path: Path) -> str:
        try:
            from faster_whisper import WhisperModel
        except ImportError as error:
            raise RuntimeError("Instala faster-whisper para usar el proveedor STT local.") from error

        if self._model is None:
            self._model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type)
        segments, _info = self._model.transcribe(str(audio_path), vad_filter=True)
        return " ".join(segment.text.strip() for segment in segments).strip()
