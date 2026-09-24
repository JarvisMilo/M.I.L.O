import unittest

from tools import Permission, ToolCall, ToolDefinition, ToolRegistry, default_registry


class ToolRegistryTests(unittest.TestCase):
    def test_calculate_is_registered_and_does_not_execute_python(self) -> None:
        registry = default_registry()
        result = registry.invoke(ToolCall("calculate", {"expression": "(2 + 3) * 4"}))
        self.assertEqual(result.result, {"value": 20})
        self.assertIsNone(result.error)

        blocked = registry.invoke(ToolCall("calculate", {"expression": "__import__('os').system('id')"}))
        self.assertIsNotNone(blocked.error)

    def test_unknown_tools_are_denied(self) -> None:
        result = default_registry().invoke(ToolCall("shell", {"command": "rm -rf /"}))
        self.assertIn("no registrada", result.error or "")

    def test_confirmation_is_required_for_irreversible_tools(self) -> None:
        registry = ToolRegistry()
        registry.register(ToolDefinition("delete_example", "Prueba", {"type": "object", "properties": {}, "additionalProperties": False}, lambda _args: "done", Permission.CONFIRMATION_REQUIRED))
        denied = registry.invoke(ToolCall("delete_example", {}))
        self.assertIn("Confirmación", denied.error or "")
        accepted = registry.invoke(ToolCall("delete_example", {}), lambda _call, _tool: True)
        self.assertEqual(accepted.result, "done")
