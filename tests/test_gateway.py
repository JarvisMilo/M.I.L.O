from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from events import Event, EventBus, RetryPolicy
from gateway import Capability, Gateway, Principal, SQLitePlatformState


class GatewayTests(unittest.TestCase):
    def test_voice_api_publishes_events_retries_and_audits(self) -> None:
        class FlakyVoice:
            def __init__(self) -> None:
                self.attempts = 0

            def run_turn(self) -> str:
                self.attempts += 1
                if self.attempts == 1:
                    raise RuntimeError("temporal")
                return "hola"

        with TemporaryDirectory() as temporary:
            events: list[str] = []
            bus = EventBus()
            bus.subscribe("*", lambda event: events.append(event.type))
            state = SQLitePlatformState(Path(temporary) / "platform.sqlite3")
            gateway = Gateway(state, voice=FlakyVoice(), event_bus=bus, retry=RetryPolicy(attempts=2, delay_seconds=0))
            user = Principal("desktop", frozenset({Capability.VOICE}))
            self.assertEqual(gateway.voice_turn(user, "turn-1"), "hola")
            self.assertIn("voice.turn.requested", events)
            self.assertIn("voice.turn.succeeded", events)
            self.assertEqual(state.get_state("last:voice.turn"), {"correlation_id": "turn-1", "status": "succeeded"})
            self.assertEqual(state.recent_audit()[0]["status"], "succeeded")

    def test_capabilities_and_human_confirmation_gate_high_impact_actions(self) -> None:
        class Device:
            def execute(self, command: str, arguments: dict) -> str:
                return "done"

        with TemporaryDirectory() as temporary:
            gateway = Gateway(SQLitePlatformState(Path(temporary) / "platform.sqlite3"), devices=Device())
            no_access = Principal("mobile", frozenset())
            with self.assertRaises(PermissionError):
                gateway.device_action(no_access, "unlock", {})

            no_approval = Principal("mobile", frozenset({Capability.DEVICES, Capability.HIGH_IMPACT}))
            with self.assertRaises(PermissionError):
                gateway.device_action(no_approval, "unlock", {})

            approved = Gateway(SQLitePlatformState(Path(temporary) / "approved.sqlite3"), devices=Device(), approval=lambda _action, _detail: True)
            self.assertEqual(approved.device_action(no_approval, "unlock", {}), "done")

    def test_health_checks_do_not_fail_the_gateway(self) -> None:
        with TemporaryDirectory() as temporary:
            gateway = Gateway(SQLitePlatformState(Path(temporary) / "platform.sqlite3"))
            gateway.register_health_check("ok", lambda: {"model": "ready"})
            gateway.register_health_check("bad", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
            self.assertEqual(gateway.health()["ok"]["status"], "ok")
            self.assertEqual(gateway.health()["bad"]["status"], "failed")
