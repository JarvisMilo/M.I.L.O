"""Configuración tipada para M.I.L.O., cargada exclusivamente desde el entorno."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    enabled: bool = True
    record_seconds: float = 5.0
    sample_rate: int = 16_000
    vad_threshold: float = 0.5
    stt_model: str = "base"
    stt_device: str = "auto"
    stt_compute_type: str = "int8"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    piper_model: Path | None = None
    data_dir: Path = Path(".milo/audio")
    memory_db: Path = Path(".milo/memory.sqlite3")
    memory_max_results: int = 3

    @classmethod
    def from_env(cls) -> "Settings":
        piper_model = os.getenv("MILO_PIPER_MODEL")
        return cls(
            enabled=os.getenv("MILO_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"},
            record_seconds=float(os.getenv("MILO_RECORD_SECONDS", "5")),
            sample_rate=int(os.getenv("MILO_SAMPLE_RATE", "16000")),
            vad_threshold=float(os.getenv("MILO_VAD_THRESHOLD", "0.5")),
            stt_model=os.getenv("MILO_STT_MODEL", "base"),
            stt_device=os.getenv("MILO_STT_DEVICE", "auto"),
            stt_compute_type=os.getenv("MILO_STT_COMPUTE_TYPE", "int8"),
            ollama_url=os.getenv("MILO_OLLAMA_URL", "http://localhost:11434"),
            ollama_model=os.getenv("MILO_OLLAMA_MODEL", "llama3.2"),
            piper_model=Path(piper_model) if piper_model else None,
            data_dir=Path(os.getenv("MILO_DATA_DIR", ".milo/audio")),
            memory_db=Path(os.getenv("MILO_MEMORY_DB", ".milo/memory.sqlite3")),
            memory_max_results=int(os.getenv("MILO_MEMORY_MAX_RESULTS", "3")),
        )
