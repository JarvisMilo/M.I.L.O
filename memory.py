"""Memoria persistente local con SQLite y recuperación determinista por reglas."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from uuid import uuid4


class SecretMemoryError(ValueError):
    """Se intenta persistir un secreto sin una decisión explícita del usuario."""


@dataclass(frozen=True)
class MemoryRecord:
    id: str
    type: str
    content: str
    timestamp: str
    origin: str
    confidence: float


@dataclass(frozen=True)
class MemoryMatch:
    record: MemoryRecord
    reason: str


class MemoryRetriever(Protocol):
    def search(self, query: str, limit: int) -> list[MemoryMatch]: ...


class EmbeddingMemoryRetriever(Protocol):
    """Punto de extensión para un índice vectorial (p. ej., Qdrant) futuro."""

    def search(self, query: str, limit: int) -> list[MemoryMatch]: ...


_SECRET_PATTERNS = (
    re.compile(r"\b(?:password|contrase(?:ñ|n)a|secret|token|api[_ -]?key)\s*[:=]", re.IGNORECASE),
    re.compile(r"\b(?:ghp|github_pat|sk|AKIA)[-_A-Za-z0-9]{12,}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._-]+", re.IGNORECASE),
)
_TOKENS = re.compile(r"\w+", re.UNICODE)
_VALID_TYPES = {"fact", "preference", "event"}


class SQLiteMemoryStore:
    """Fuente de verdad SQLite; las conexiones cortas permiten usarlo desde PTT."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY, type TEXT NOT NULL, content TEXT NOT NULL,
                    timestamp TEXT NOT NULL, origin TEXT NOT NULL, confidence REAL NOT NULL
                )"""
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_memories_type_timestamp ON memories(type, timestamp DESC)")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database_path)

    def save(self, type: str, content: str, origin: str, confidence: float, *, allow_secrets: bool = False) -> MemoryRecord:
        self._validate(type, content, confidence, allow_secrets)
        record = MemoryRecord(str(uuid4()), type, content.strip(), datetime.now(UTC).isoformat(), origin, confidence)
        with self._connect() as connection:
            connection.execute("INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?)", tuple(record.__dict__.values()))
        return record

    def get(self, memory_id: str) -> MemoryRecord | None:
        with self._connect() as connection:
            row = connection.execute("SELECT id, type, content, timestamp, origin, confidence FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return MemoryRecord(*row) if row else None

    def update(self, memory_id: str, *, content: str | None = None, type: str | None = None, origin: str | None = None, confidence: float | None = None, allow_secrets: bool = False) -> MemoryRecord:
        current = self.get(memory_id)
        if current is None:
            raise KeyError(f"No existe el recuerdo: {memory_id}")
        updated = MemoryRecord(memory_id, type or current.type, content if content is not None else current.content, current.timestamp, origin or current.origin, confidence if confidence is not None else current.confidence)
        self._validate(updated.type, updated.content, updated.confidence, allow_secrets)
        with self._connect() as connection:
            connection.execute("UPDATE memories SET type=?, content=?, origin=?, confidence=? WHERE id=?", (updated.type, updated.content.strip(), updated.origin, updated.confidence, memory_id))
        return self.get(memory_id)  # type: ignore[return-value]

    def delete(self, memory_id: str) -> bool:
        with self._connect() as connection:
            return connection.execute("DELETE FROM memories WHERE id = ?", (memory_id,)).rowcount == 1

    def search(self, query: str, limit: int) -> list[MemoryMatch]:
        if limit < 1:
            return []
        query_tokens = set(_TOKENS.findall(query.lower()))
        if not query_tokens:
            return []
        with self._connect() as connection:
            rows = connection.execute("SELECT id, type, content, timestamp, origin, confidence FROM memories").fetchall()
        matches: list[tuple[int, MemoryMatch]] = []
        for row in rows:
            record = MemoryRecord(*row)
            shared = sorted(query_tokens & set(_TOKENS.findall(record.content.lower())))
            if shared:
                matches.append((len(shared), MemoryMatch(record, f"coinciden términos: {', '.join(shared)}")))
        return [match for _score, match in sorted(matches, key=lambda item: (item[0], item[1].record.confidence, item[1].record.timestamp), reverse=True)[:limit]]

    @staticmethod
    def _validate(type: str, content: str, confidence: float, allow_secrets: bool) -> None:
        if type not in _VALID_TYPES:
            raise ValueError(f"Tipo no permitido: {type}")
        if not content.strip():
            raise ValueError("El contenido del recuerdo no puede estar vacío.")
        if not 0 <= confidence <= 1:
            raise ValueError("La confianza debe estar entre 0 y 1.")
        if not allow_secrets and any(pattern.search(content) for pattern in _SECRET_PATTERNS):
            raise SecretMemoryError("El contenido parece contener un secreto y no se guardó.")
