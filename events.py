"""Bus de eventos local, tipado y con reintentos acotados."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import RLock
from typing import Any, Callable, Mapping
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Event:
    type: str
    payload: Mapping[str, Any]
    source: str
    correlation_id: str = ""
    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass(frozen=True)
class RetryPolicy:
    attempts: int = 3
    delay_seconds: float = 0.05


EventHandler = Callable[[Event], None]


class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = {}
        self._lock = RLock()

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        with self._lock:
            self._handlers.setdefault(event_type, []).append(handler)

    def publish(self, event: Event, retry: RetryPolicy = RetryPolicy()) -> None:
        with self._lock:
            handlers = tuple(self._handlers.get(event.type, ())) + tuple(self._handlers.get("*", ()))
        for handler in handlers:
            self._deliver(event, handler, retry)

    @staticmethod
    def _deliver(event: Event, handler: EventHandler, retry: RetryPolicy) -> None:
        for attempt in range(1, retry.attempts + 1):
            try:
                handler(event)
                return
            except Exception:
                if attempt == retry.attempts:
                    logger.exception("Entrega de evento agotada type=%s event_id=%s", event.type, event.id)
                    return
                time.sleep(retry.delay_seconds * attempt)
