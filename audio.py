"""Entrada y salida de audio aisladas de la lógica del asistente."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)


class AudioRecorder(Protocol):
    def record(self, destination: Path, seconds: float, sample_rate: int) -> Path:
        """Graba audio mono PCM WAV y devuelve su ruta."""


class AudioSpeaker(Protocol):
    def play(self, source: Path) -> None:
        """Reproduce un fichero WAV de forma bloqueante."""


class SoundDeviceRecorder:
    def record(self, destination: Path, seconds: float, sample_rate: int) -> Path:
        try:
            import sounddevice as sd
            import soundfile as sf
        except ImportError as error:
            raise RuntimeError("Instala sounddevice y soundfile para usar el micrófono.") from error

        destination.parent.mkdir(parents=True, exist_ok=True)
        frames = int(seconds * sample_rate)
        logger.info("Grabando %.1f s a %d Hz", seconds, sample_rate)
        recording = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="float32")
        sd.wait()
        sf.write(destination, recording, sample_rate)
        return destination


class SoundDeviceSpeaker:
    def play(self, source: Path) -> None:
        try:
            import sounddevice as sd
            import soundfile as sf
        except ImportError as error:
            raise RuntimeError("Instala sounddevice y soundfile para usar el altavoz.") from error

        data, sample_rate = sf.read(source, dtype="float32")
        logger.info("Reproduciendo %s", source)
        sd.play(data, sample_rate)
        sd.wait()
