"""Workflows mínimos con orquestación, especialistas aislados y trazabilidad."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Callable, Protocol
from uuid import uuid4

from tools import ConfirmationCallback, ToolCall, ToolRegistry, ToolResult


@dataclass(frozen=True)
class AgentTask:
    agent_name: str
    tool_call: ToolCall
    delegated_tools: frozenset[str] = frozenset()


@dataclass(frozen=True)
class TraceEvent:
    timestamp: str
    event: str
    detail: str


@dataclass(frozen=True)
class Checkpoint:
    stage: str
    timestamp: str
    summary: str


@dataclass
class WorkflowState:
    id: str
    request: str
    status: str = "received"
    summary: str = ""
    checkpoints: list[Checkpoint] = field(default_factory=list)
    trace: list[TraceEvent] = field(default_factory=list)


class DynamicRouter(Protocol):
    """Sólo se usa cuando no existe un workflow determinista conocido."""

    def route(self, request: str, available_agents: tuple[str, ...]) -> AgentTask | None: ...


class HumanApprovalGate(Protocol):
    def approve(self, call: ToolCall, reason: str) -> bool: ...


class SpecialistAgent:
    """Ejecuta sólo su toolset, salvo herramientas delegadas explícitamente."""

    def __init__(self, name: str, objective: str, registry: ToolRegistry, toolset: frozenset[str]) -> None:
        self.name = name
        self.objective = objective
        self._registry = registry
        self._toolset = toolset

    def execute(self, call: ToolCall, delegated_tools: frozenset[str] = frozenset(), approval: HumanApprovalGate | None = None) -> ToolResult:
        if call.name not in self._toolset and call.name not in delegated_tools:
            return ToolResult(call.call_id, call.name, call.arguments, None, 0, f"{self.name} no tiene delegación para usar {call.name}")
        definition = self._registry.definition(call.name)
        if definition is None:
            return ToolResult(call.call_id, call.name, call.arguments, None, 0, f"Herramienta no registrada: {call.name}")
        confirmation: ConfirmationCallback | None = None
        if definition.external_action:
            confirmation = None if approval is None else lambda tool_call, _tool: approval.approve(tool_call, f"Acción externa solicitada por {self.name}")
        return self._registry.invoke(call, confirmation)


class MultiAgentOrchestrator:
    """Enruta, conserva estado y resume; nunca ejecuta herramientas de dominio."""

    def __init__(self, agents: list[SpecialistAgent], dynamic_router: DynamicRouter | None = None, approval: HumanApprovalGate | None = None) -> None:
        self._agents = {agent.name: agent for agent in agents}
        self._dynamic_router = dynamic_router
        self._approval = approval

    def run(self, request: str) -> WorkflowState:
        state = WorkflowState(id=str(uuid4()), request=request)
        self._trace(state, "received", "Solicitud recibida")
        task = self._deterministic_route(request)
        if task is None and self._dynamic_router is not None:
            self._trace(state, "dynamic_route", "No había workflow conocido; se solicita decisión dinámica")
            task = self._dynamic_router.route(request, tuple(self._agents))
        if task is None or task.agent_name not in self._agents:
            state.status = "unroutable"
            state.summary = "No existe un workflow ni una delegación válida para esta solicitud."
            self._checkpoint(state, "routing", state.summary)
            return state
        self._trace(state, "delegated", f"Delegado a {task.agent_name}: {task.tool_call.name}")
        self._checkpoint(state, "routing", f"{task.agent_name} recibió {task.tool_call.name}")
        result = self._agents[task.agent_name].execute(task.tool_call, task.delegated_tools, self._approval)
        state.status = "completed" if result.error is None else "failed"
        state.summary = self._summarize(task.agent_name, result)
        self._trace(state, "result", state.summary)
        self._checkpoint(state, "completed", state.summary)
        return state

    @staticmethod
    def _deterministic_route(request: str) -> AgentTask | None:
        normalized = request.strip().lower()
        expression = re.search(r"(?:calcula|calcular)\s+(.+)$", request, re.IGNORECASE)
        if expression:
            return AgentTask("math", ToolCall("calculate", {"expression": expression.group(1)}))
        if "hora" in normalized or "fecha" in normalized:
            return AgentTask("system", ToolCall("get_current_time", {}))
        if "sistema" in normalized or "entorno" in normalized:
            return AgentTask("system", ToolCall("get_runtime_info", {}))
        return None

    @staticmethod
    def _summarize(agent_name: str, result: ToolResult) -> str:
        if result.error:
            return f"{agent_name} no completó la tarea: {result.error}"
        return f"{agent_name} completó la tarea con {result.result}."

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    def _trace(self, state: WorkflowState, event: str, detail: str) -> None:
        state.trace.append(TraceEvent(self._now(), event, detail))

    def _checkpoint(self, state: WorkflowState, stage: str, summary: str) -> None:
        state.checkpoints.append(Checkpoint(stage, self._now(), summary))


def default_multiagent_orchestrator(registry: ToolRegistry, dynamic_router: DynamicRouter | None = None, approval: HumanApprovalGate | None = None) -> MultiAgentOrchestrator:
    return MultiAgentOrchestrator(
        [
            SpecialistAgent("math", "Resolver aritmética segura", registry, frozenset({"calculate"})),
            SpecialistAgent("system", "Consultar hora y estado local no sensible", registry, frozenset({"get_current_time", "get_runtime_info"})),
        ],
        dynamic_router,
        approval,
    )
