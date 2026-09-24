"""Detección de actividad de voz para evitar llamadas STT en silencio."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Protocol


class VADProvider(Protocol):
    def has_speech(self, audio_path: Path) -> bool:
        """Indica si el WAV contiene al menos un segmento de habla."""


class SileroVAD:
    """Adaptador lazy del paquete silero-vad."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self._model = None

    def has_speech(self, audio_path: Path) -> bool:
        silero_vad = importlib.import_module("silero_vad")
        get_speech_timestamps = silero_vad.get_speech_timestamps
        load_silero_vad = silero_vad.load_silero_vad
        read_audio = silero_vad.read_audio

        if self._model is None:
            self._model = load_silero_vad()
        waveform = read_audio(str(audio_path), sampling_rate=16_000)
        timestamps = get_speech_timestamps(waveform, self._model, threshold=self.threshold, sampling_rate=16_000)
        return bool(timestamps)
