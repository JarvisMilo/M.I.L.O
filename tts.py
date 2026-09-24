"""Proveedores de text-to-speech."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Protocol


class TTSProvider(Protocol):
    def synthesize(self, text: str, destination: Path) -> Path:
        """Sintetiza texto a un WAV y devuelve su ruta."""


class PiperTTS:
    def __init__(self, model_path: Path, executable: str = "piper") -> None:
        self.model_path = model_path
        self.executable = executable

    def synthesize(self, text: str, destination: Path) -> Path:
        if not self.model_path.is_file():
            raise RuntimeError(f"No existe el modelo Piper: {self.model_path}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                [self.executable, "--model", str(self.model_path), "--output_file", str(destination)],
                input=text,
                text=True,
                check=True,
                capture_output=True,
            )
        except FileNotFoundError as error:
            raise RuntimeError("No se encontró 'piper' en PATH.") from error
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"Piper falló: {error.stderr.strip()}") from error
        return destination
