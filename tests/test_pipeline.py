from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest

from config import Settings
from main import PipelineState, VoicePipeline
from llm import LLMToolResponse
from memory import SQLiteMemoryStore


class Recorder:
    def record(
        self,
        destination: Path,
        seconds: float,
        sample_rate: int,
        stop_event: threading.Event | None = None,
    ) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"wav")
        return destination


class STT:
    def __init__(self, transcript: str = "hola") -> None:
        self.transcript = transcript

    def transcribe(self, audio_path: Path) -> str:
        return self.transcript


class LLM:
    def respond(self, text: str) -> str:
        return f"respuesta a {text}"


class ToolLLM:
    def respond(self, text: str) -> str:
        return text

    def respond_with_tools(self, text: str, tools: list[dict]) -> LLMToolResponse:
        return LLMToolResponse("", [{"id": "call-1", "function": {"name": "calculate", "arguments": {"expression": "2 + 2"}}}])

    def respond_after_tools(self, text: str, response: LLMToolResponse, results: list[dict]) -> str:
        return f"el resultado es {results[0]['result']['value']}"


class TTS:
    def synthesize(self, text: str, destination: Path) -> Path:
        destination.write_bytes(text.encode())
        return destination


class Speaker:
    def __init__(self) -> None:
        self.played: Path | None = None

    def play(self, source: Path) -> None:
        self.played = source


class VAD:
    def __init__(self, has_speech: bool) -> None:
        self._has_speech = has_speech

    def has_speech(self, audio_path: Path) -> bool:
        return self._has_speech


class PipelineTests(unittest.TestCase):
    def make_pipeline(self, directory: Path, transcript: str = "hola") -> tuple[VoicePipeline, Speaker]:
        speaker = Speaker()
        return (
            VoicePipeline(Settings(data_dir=directory), Recorder(), STT(transcript), LLM(), TTS(), speaker),
            speaker,
        )

    def test_full_turn_returns_response_and_returns_to_idle(self) -> None:
        with TemporaryDirectory() as temporary:
            pipeline, speaker = self.make_pipeline(Path(temporary))
            answer = pipeline.run_turn()
            self.assertEqual(answer, "respuesta a hola")
            self.assertEqual(pipeline.state, PipelineState.IDLE)
            self.assertIsNotNone(speaker.played)
            self.assertTrue(speaker.played.exists())

    def test_turn_passes_ptt_stop_event_to_recorder(self) -> None:
        class PTTRecorder(Recorder):
            def __init__(self) -> None:
                self.stop_event: threading.Event | None = None

            def record(self, destination: Path, seconds: float, sample_rate: int, stop_event: threading.Event | None = None) -> Path:
                self.stop_event = stop_event
                return super().record(destination, seconds, sample_rate, stop_event)

        with TemporaryDirectory() as temporary:
            recorder = PTTRecorder()
            pipeline = VoicePipeline(Settings(data_dir=Path(temporary)), recorder, STT(), LLM(), TTS(), Speaker())
            stop_event = threading.Event()
            pipeline.run_turn(stop_event)
            self.assertIs(recorder.stop_event, stop_event)

    def test_registered_tool_call_is_executed_and_returned_to_model(self) -> None:
        with TemporaryDirectory() as temporary:
            pipeline = VoicePipeline(Settings(data_dir=Path(temporary)), Recorder(), STT(), ToolLLM(), TTS(), Speaker())
            self.assertEqual(pipeline.run_turn(), "el resultado es 4")

    def test_relevant_memories_are_added_to_the_model_prompt(self) -> None:
        class ContextLLM(LLM):
            def respond(self, text: str) -> str:
                self.prompt = text
                return "respuesta"

        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            store = SQLiteMemoryStore(directory / "memory.sqlite3")
            store.save("preference", "El usuario prefiere té verde", "user", 0.9)
            llm = ContextLLM()
            pipeline = VoicePipeline(Settings(data_dir=directory), Recorder(), STT("quiero té"), llm, TTS(), Speaker(), memory=store)
            pipeline.run_turn()
            self.assertIn("El usuario prefiere té verde", llm.prompt)

    def test_empty_transcript_skips_llm_tts_and_playback(self) -> None:
        with TemporaryDirectory() as temporary:
            pipeline, speaker = self.make_pipeline(Path(temporary), transcript="")
            self.assertIsNone(pipeline.run_turn())
            self.assertEqual(pipeline.state, PipelineState.IDLE)
            self.assertIsNone(speaker.played)

    def test_vad_silence_skips_stt_and_playback(self) -> None:
        with TemporaryDirectory() as temporary:
            pipeline, speaker = self.make_pipeline(Path(temporary))
            pipeline.vad = VAD(False)
            self.assertIsNone(pipeline.run_turn())
            self.assertEqual(pipeline.state, PipelineState.IDLE)
            self.assertIsNone(speaker.played)

    def test_failures_return_to_idle(self) -> None:
        class BrokenSTT:
            def transcribe(self, audio_path: Path) -> str:
                raise RuntimeError("micrófono corrupto")

        with TemporaryDirectory() as temporary:
            speaker = Speaker()
            pipeline = VoicePipeline(Settings(data_dir=Path(temporary)), Recorder(), BrokenSTT(), LLM(), TTS(), speaker)
            self.assertIsNone(pipeline.run_turn())
            self.assertEqual(pipeline.state, PipelineState.IDLE)


if __name__ == "__main__":
    unittest.main()
