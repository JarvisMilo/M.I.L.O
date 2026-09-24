"""Registro seguro de herramientas: sólo se ejecutan funciones registradas."""

from __future__ import annotations

import ast
import json
import logging
import operator
import platform
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Mapping

logger = logging.getLogger(__name__)
JsonObject = dict[str, Any]


class Permission(Enum):
    READ_ONLY = "read_only"
    CONFIRMATION_REQUIRED = "confirmation_required"


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: JsonObject
    call_id: str = ""


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    name: str
    arguments: JsonObject
    result: Any | None
    duration_ms: float
    error: str | None = None

    def as_message(self) -> JsonObject:
        return asdict(self)


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: JsonObject
    execute: Callable[[JsonObject], Any]
    permission: Permission = Permission.READ_ONLY
    external_action: bool = False

    def as_llm_tool(self) -> JsonObject:
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}


ConfirmationCallback = Callable[[ToolCall, ToolDefinition], bool]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        if not tool.name.replace("_", "").isalnum() or tool.name in self._tools:
            raise ValueError(f"Nombre de herramienta inválido o duplicado: {tool.name}")
        if tool.external_action and tool.permission is not Permission.CONFIRMATION_REQUIRED:
            raise ValueError("Las acciones externas deben requerir confirmación humana.")
        self._tools[tool.name] = tool

    def definition(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def specifications(self) -> list[JsonObject]:
        return [tool.as_llm_tool() for tool in self._tools.values()]

    def invoke(self, call: ToolCall, confirm: ConfirmationCallback | None = None) -> ToolResult:
        started = time.perf_counter()
        tool = self._tools.get(call.name)
        result: Any | None = None
        error: str | None = None
        try:
            if tool is None:
                raise PermissionError(f"Herramienta no registrada: {call.name}")
            _validate_arguments(call.arguments, tool.parameters)
            if tool.permission is Permission.CONFIRMATION_REQUIRED and (confirm is None or not confirm(call, tool)):
                raise PermissionError(f"Confirmación humana requerida para: {call.name}")
            result = tool.execute(call.arguments)
            json.dumps(result)
        except Exception as exc:
            error = str(exc)
        duration_ms = (time.perf_counter() - started) * 1000
        logged_result = ToolResult(call.call_id, call.name, call.arguments, result, duration_ms, error)
        logger.info("tool_call name=%s arguments=%s result=%s duration_ms=%.0f error=%s", call.name, call.arguments, result, duration_ms, error)
        return logged_result


def _validate_arguments(arguments: Mapping[str, Any], schema: JsonObject) -> None:
    if schema.get("type") != "object" or not isinstance(arguments, Mapping):
        raise ValueError("Los argumentos deben ser un objeto JSON.")
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    missing = [name for name in required if name not in arguments]
    unknown = set(arguments) - set(properties)
    if missing:
        raise ValueError(f"Faltan argumentos requeridos: {', '.join(missing)}")
    if schema.get("additionalProperties") is False and unknown:
        raise ValueError(f"Argumentos no permitidos: {', '.join(sorted(unknown))}")
    for name, value in arguments.items():
        expected = properties.get(name, {}).get("type")
        if expected == "string" and not isinstance(value, str):
            raise ValueError(f"{name} debe ser texto")
        if expected == "number" and (not isinstance(value, (int, float)) or isinstance(value, bool)):
            raise ValueError(f"{name} debe ser numérico")


_OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def _calculate(arguments: JsonObject) -> JsonObject:
    def evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            return _OPERATORS[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return evaluate(node.operand) if isinstance(node.op, ast.UAdd) else -evaluate(node.operand)
        raise ValueError("La expresión sólo permite números y +, -, * o /.")

    return {"value": evaluate(ast.parse(arguments["expression"], mode="eval").body)}


def default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    empty_object = {"type": "object", "properties": {}, "additionalProperties": False}
    registry.register(ToolDefinition("get_current_time", "Devuelve la fecha y hora local actual.", empty_object, lambda _args: {"local_time": datetime.now().astimezone().isoformat()}))
    registry.register(ToolDefinition("calculate", "Calcula una expresión aritmética básica sin ejecutar código.", {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"], "additionalProperties": False}, _calculate))
    registry.register(ToolDefinition("get_runtime_info", "Devuelve información no sensible del entorno de ejecución.", empty_object, lambda _args: {"system": platform.system(), "python": platform.python_version()}))
    return registry
