import unittest

from multiagent import AgentTask, MultiAgentOrchestrator, SpecialistAgent, default_multiagent_orchestrator
from tools import Permission, ToolCall, ToolDefinition, ToolRegistry, default_registry


class MultiAgentTests(unittest.TestCase):
    def test_known_math_request_uses_deterministic_workflow(self) -> None:
        state = default_multiagent_orchestrator(default_registry()).run("calcula (2 + 3) * 4")
        self.assertEqual(state.status, "completed")
        self.assertIn("20", state.summary)
        self.assertEqual(state.checkpoints[0].stage, "routing")
        self.assertEqual([event.event for event in state.trace], ["received", "delegated", "result"])

    def test_specialist_cannot_use_another_domain_tool_without_delegation(self) -> None:
        agent = SpecialistAgent("math", "aritmética", default_registry(), frozenset({"calculate"}))
        denied = agent.execute(ToolCall("get_runtime_info", {}))
        self.assertIn("no tiene delegación", denied.error or "")
        delegated = agent.execute(ToolCall("get_runtime_info", {}), delegated_tools=frozenset({"get_runtime_info"}))
        self.assertIsNone(delegated.error)

    def test_external_action_requires_human_approval(self) -> None:
        registry = ToolRegistry()
        registry.register(ToolDefinition("send_notification", "Acción externa", {"type": "object", "properties": {}, "additionalProperties": False}, lambda _args: "sent", Permission.CONFIRMATION_REQUIRED, external_action=True))
        agent = SpecialistAgent("notifier", "notificar", registry, frozenset({"send_notification"}))
        denied = agent.execute(ToolCall("send_notification", {}))
        self.assertIn("Confirmación", denied.error or "")

        class Approved:
            def approve(self, call: ToolCall, reason: str) -> bool:
                return True

        self.assertEqual(agent.execute(ToolCall("send_notification", {}), approval=Approved()).result, "sent")

    def test_dynamic_route_must_name_a_known_agent(self) -> None:
        class InvalidRouter:
            def route(self, request: str, available_agents: tuple[str, ...]) -> AgentTask:
                return AgentTask("unknown", ToolCall("calculate", {"expression": "1+1"}))

        state = default_multiagent_orchestrator(default_registry(), dynamic_router=InvalidRouter()).run("decide algo")
        self.assertEqual(state.status, "unroutable")
