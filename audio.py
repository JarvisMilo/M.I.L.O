"""Entrada y salida de audio aisladas de la lógica del asistente."""

from __future__ import annotations

import importlib
import logging
import threading
import time
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)


class AudioRecorder(Protocol):
    def record(
        self,
        destination: Path,
        seconds: float,
        sample_rate: int,
        stop_event: threading.Event | None = None,
    ) -> Path:
        """Graba WAV hasta el límite o, si se indica, hasta que termine el PTT."""


class AudioSpeaker(Protocol):
    def play(self, source: Path) -> None:
        """Reproduce un fichero WAV de forma bloqueante."""


class SoundDeviceRecorder:
    def record(
        self,
        destination: Path,
        seconds: float,
        sample_rate: int,
        stop_event: threading.Event | None = None,
    ) -> Path:
        sd = importlib.import_module("sounddevice")
        sf = importlib.import_module("soundfile")

        destination.parent.mkdir(parents=True, exist_ok=True)
        if seconds <= 0:
            raise ValueError("La duración máxima de grabación debe ser positiva.")

        chunks = []

        def collect(indata, frames, time_info, status) -> None:
            if status:
                logger.warning("Estado de entrada de audio: %s", status)
            chunks.append(indata.copy())

        logger.info("Grabando hasta %.1f s a %d Hz", seconds, sample_rate)
        started = time.monotonic()
        with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32", callback=collect):
            while time.monotonic() - started < seconds:
                if stop_event is not None and stop_event.wait(timeout=0.05):
                    break
        if not chunks:
            raise RuntimeError("No se capturó audio del micrófono.")
        np = importlib.import_module("numpy")

        sf.write(destination, np.concatenate(chunks), sample_rate)
        return destination


class SoundDeviceSpeaker:
    def play(self, source: Path) -> None:
        sd = importlib.import_module("sounddevice")
        sf = importlib.import_module("soundfile")

        data, sample_rate = sf.read(source, dtype="float32")
        logger.info("Reproduciendo %s", source)
        sd.play(data, sample_rate)
        sd.wait()
