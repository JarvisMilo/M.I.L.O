from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from config import Settings
from main import PipelineState, VoicePipeline


class Recorder:
    def record(self, destination: Path, seconds: float, sample_rate: int) -> Path:
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
