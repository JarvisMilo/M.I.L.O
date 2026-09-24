"""Gateway local-first: única entrada estable para las capacidades de M.I.L.O."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol
from uuid import uuid4

from events import Event, EventBus, RetryPolicy
from memory import MemoryMatch, SQLiteMemoryStore
from multiagent import MultiAgentOrchestrator, WorkflowState
from tools import Permission, ToolCall, ToolRegistry, ToolResult


class Capability(Enum):
    VOICE = "voice"
    MEMORY_READ = "memory.read"
    MEMORY_WRITE = "memory.write"
    TOOLS = "tools.invoke"
    AGENTS = "agents.route"
    DEVICES = "devices.control"
    HIGH_IMPACT = "actions.high_impact"
    SYSTEM_CONTROL = "system.control"


@dataclass(frozen=True)
class Principal:
    id: str
    capabilities: frozenset[Capability]


class VoiceAPI(Protocol):
    def run_turn(self) -> str | None: ...


class DeviceAPI(Protocol):
    def execute(self, command: str, arguments: dict[str, Any]) -> Any: ...


HealthCheck = Callable[[], dict[str, Any]]
Approval = Callable[[str, dict[str, Any]], bool]


class SQLitePlatformState:
    """Estado y auditoría persistentes; SQLite sigue siendo local-first."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS platform_state (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            connection.execute("CREATE TABLE IF NOT EXISTS audit_log (id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, principal TEXT NOT NULL, action TEXT NOT NULL, status TEXT NOT NULL, detail TEXT NOT NULL, correlation_id TEXT NOT NULL)")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def set_state(self, key: str, value: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute("INSERT INTO platform_state(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, json.dumps(value)))

    def get_state(self, key: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT value FROM platform_state WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def audit(self, principal: Principal, action: str, status: str, detail: dict[str, Any], correlation_id: str) -> None:
        with self._connect() as connection:
            connection.execute("INSERT INTO audit_log VALUES (?, datetime('now'), ?, ?, ?, ?, ?)", (str(uuid4()), principal.id, action, status, json.dumps(detail, default=str), correlation_id))

    def recent_audit(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT timestamp, principal, action, status, detail, correlation_id FROM audit_log ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
        return [{"timestamp": row[0], "principal": row[1], "action": row[2], "status": row[3], "detail": json.loads(row[4]), "correlation_id": row[5]} for row in rows]


class Gateway:
    """Coordina APIs y políticas; las superficies cliente no contienen lógica de dominio."""

    def __init__(self, state: SQLitePlatformState, *, voice: VoiceAPI | None = None, memory: SQLiteMemoryStore | None = None, tools: ToolRegistry | None = None, agents: MultiAgentOrchestrator | None = None, devices: DeviceAPI | None = None, event_bus: EventBus | None = None, approval: Approval | None = None, retry: RetryPolicy = RetryPolicy()) -> None:
        self.state = state
        self.voice = voice
        self.memory = memory
        self.tools = tools
        self.agents = agents
        self.devices = devices
        self.events = event_bus or EventBus()
        self.approval = approval
        self.retry = retry
        self._health_checks: dict[str, HealthCheck] = {}

    def register_health_check(self, name: str, check: HealthCheck) -> None:
        self._health_checks[name] = check

    def health(self) -> dict[str, dict[str, Any]]:
        report: dict[str, dict[str, Any]] = {}
        for name, check in self._health_checks.items():
            try:
                report[name] = {"status": "ok", **check()}
            except Exception as error:
                report[name] = {"status": "failed", "error": str(error)}
        return report

    def is_enabled(self) -> bool:
        state = self.state.get_state("system.enabled")
        return True if state is None else bool(state.get("enabled", True))

    def set_enabled(self, principal: Principal, enabled: bool, correlation_id: str = "") -> None:
        if Capability.SYSTEM_CONTROL not in principal.capabilities:
            self.state.audit(principal, "system.set_enabled", "denied", {"enabled": enabled}, correlation_id)
            raise PermissionError(f"{principal.id} no tiene capacidad {Capability.SYSTEM_CONTROL.value}")
        self.state.set_state("system.enabled", {"enabled": enabled})
        status = "enabled" if enabled else "disabled"
        self.state.audit(principal, "system.set_enabled", status, {"enabled": enabled}, correlation_id)
        self.events.publish(Event(f"system.{status}", {"enabled": enabled}, "gateway", correlation_id), self.retry)

    def voice_turn(self, principal: Principal, correlation_id: str = "") -> str | None:
        return self._perform(principal, Capability.VOICE, "voice.turn", {}, correlation_id, lambda: self._required(self.voice, "voice").run_turn())

    def memory_search(self, principal: Principal, query: str, limit: int, correlation_id: str = "") -> list[MemoryMatch]:
        return self._perform(principal, Capability.MEMORY_READ, "memory.search", {"query": query, "limit": limit}, correlation_id, lambda: self._required(self.memory, "memory").search(query, limit))

    def memory_save(self, principal: Principal, type: str, content: str, origin: str, confidence: float, correlation_id: str = "") -> Any:
        return self._perform(principal, Capability.MEMORY_WRITE, "memory.save", {"type": type, "origin": origin}, correlation_id, lambda: self._required(self.memory, "memory").save(type, content, origin, confidence))

    def invoke_tool(self, principal: Principal, call: ToolCall, correlation_id: str = "") -> ToolResult:
        registry = self._required(self.tools, "tools")
        definition = registry.definition(call.name)
        high_impact = bool(definition and definition.permission is Permission.CONFIRMATION_REQUIRED)
        return self._perform(principal, Capability.TOOLS, "tools.invoke", {"name": call.name, "arguments": call.arguments}, correlation_id, lambda: registry.invoke(call, lambda _call, _tool: True), high_impact)

    def run_agents(self, principal: Principal, request: str, correlation_id: str = "") -> WorkflowState:
        return self._perform(principal, Capability.AGENTS, "agents.run", {"request": request}, correlation_id, lambda: self._required(self.agents, "agents").run(request))

    def device_action(self, principal: Principal, command: str, arguments: dict[str, Any], correlation_id: str = "") -> Any:
        return self._perform(principal, Capability.DEVICES, "devices.execute", {"command": command, "arguments": arguments}, correlation_id, lambda: self._required(self.devices, "devices").execute(command, arguments), high_impact=True)

    def _perform(self, principal: Principal, capability: Capability, action: str, detail: dict[str, Any], correlation_id: str, operation: Callable[[], Any], high_impact: bool = False) -> Any:
        if capability not in principal.capabilities:
            self.state.audit(principal, action, "denied", detail, correlation_id)
            raise PermissionError(f"{principal.id} no tiene capacidad {capability.value}")
        if not self.is_enabled():
            self.state.audit(principal, action, "denied", {**detail, "reason": "system_disabled"}, correlation_id)
            raise RuntimeError("M.I.L.O. está desactivado.")
        if high_impact and not self._confirm(principal, action, detail, True):
            self.state.audit(principal, action, "denied", detail, correlation_id)
            raise PermissionError("La acción de alto impacto requiere confirmación humana.")
        event = Event(f"{action}.requested", detail, "gateway", correlation_id)
        self.events.publish(event, self.retry)
        try:
            result = self._retry(operation)
        except Exception as error:
            self.state.audit(principal, action, "failed", {**detail, "error": str(error)}, correlation_id)
            self.events.publish(Event(f"{action}.failed", {**detail, "error": str(error)}, "gateway", correlation_id), self.retry)
            raise
        self.state.audit(principal, action, "succeeded", detail, correlation_id)
        self.state.set_state(f"last:{action}", {"correlation_id": correlation_id, "status": "succeeded"})
        self.events.publish(Event(f"{action}.succeeded", detail, "gateway", correlation_id), self.retry)
        return result

    def _confirm(self, principal: Principal, action: str, detail: dict[str, Any], high_impact: bool) -> bool:
        return not high_impact or (Capability.HIGH_IMPACT in principal.capabilities and self.approval is not None and self.approval(action, detail))

    def _retry(self, operation: Callable[[], Any]) -> Any:
        last_error: Exception | None = None
        for attempt in range(1, self.retry.attempts + 1):
            try:
                return operation()
            except Exception as error:
                last_error = error
                if attempt < self.retry.attempts:
                    time.sleep(self.retry.delay_seconds * attempt)
        raise last_error or RuntimeError("La operación no se ejecutó.")

    @staticmethod
    def _required(value: Any, name: str) -> Any:
        if value is None:
            raise RuntimeError(f"API no configurada: {name}")
        return value
