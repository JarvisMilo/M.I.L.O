"""Punto de entrada PTT para el primer pipeline de voz de M.I.L.O."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Callable

from audio import AudioRecorder, AudioSpeaker, SoundDeviceRecorder, SoundDeviceSpeaker
from config import Settings
from llm import LLMProvider, OllamaLLM
from stt import FasterWhisperSTT, STTProvider
from tts import PiperTTS, TTSProvider
from vad import SileroVAD, VADProvider

logger = logging.getLogger(__name__)


class PipelineState(Enum):
    IDLE = auto()
    RECORDING = auto()
    ANALYZING = auto()
    TRANSCRIBING = auto()
    THINKING = auto()
    SYNTHESIZING = auto()
    PLAYING = auto()
    ERROR = auto()


@dataclass
class VoicePipeline:
    settings: Settings
    recorder: AudioRecorder
    stt: STTProvider
    llm: LLMProvider
    tts: TTSProvider
    speaker: AudioSpeaker
    vad: VADProvider | None = None
    clock: Callable[[], float] = time.perf_counter
    state: PipelineState = PipelineState.IDLE

    def _transition(self, state: PipelineState) -> None:
        logger.info("Estado: %s → %s", self.state.name, state.name)
        self.state = state

    def _timed(self, stage: str, operation: Callable[[], object]) -> object:
        started = self.clock()
        result = operation()
        logger.info("Latencia %-13s %.0f ms", stage, (self.clock() - started) * 1000)
        return result

    def run_turn(self, stop_recording: threading.Event | None = None) -> str | None:
        """Ejecuta un turno completo y garantiza el retorno a estado inactivo."""
        if self.state is not PipelineState.IDLE:
            raise RuntimeError("El pipeline ya está procesando un turno.")
        turn_started = self.clock()
        stamp = str(time.time_ns())
        recording = self.settings.data_dir / f"input-{stamp}.wav"
        response_audio = self.settings.data_dir / f"response-{stamp}.wav"
        try:
            self._transition(PipelineState.RECORDING)
            self._timed(
                "grabación",
                lambda: self.recorder.record(
                    recording,
                    self.settings.record_seconds,
                    self.settings.sample_rate,
                    stop_recording,
                ),
            )
            if self.vad is not None:
                self._transition(PipelineState.ANALYZING)
                has_speech = self._timed("vad", lambda: self.vad.has_speech(recording))
                if not has_speech:
                    logger.info("VAD no detectó habla; se omite el resto del turno.")
                    return None
            self._transition(PipelineState.TRANSCRIBING)
            transcript = self._timed("stt", lambda: self.stt.transcribe(recording))
            if not transcript:
                logger.info("No se detectó habla; se omite LLM y TTS.")
                return None
            logger.info("Usuario: %s", transcript)
            self._transition(PipelineState.THINKING)
            answer = self._timed("llm", lambda: self.llm.respond(transcript))
            logger.info("M.I.L.O.: %s", answer)
            self._transition(PipelineState.SYNTHESIZING)
            self._timed("tts", lambda: self.tts.synthesize(answer, response_audio))
            self._transition(PipelineState.PLAYING)
            self._timed("altavoz", lambda: self.speaker.play(response_audio))
            return answer
        except Exception:
            self._transition(PipelineState.ERROR)
            logger.exception("El turno de voz falló")
            return None
        finally:
            logger.info("Latencia turno total %.0f ms", (self.clock() - turn_started) * 1000)
            self._transition(PipelineState.IDLE)


def build_pipeline(settings: Settings) -> VoicePipeline:
    if settings.piper_model is None:
        raise ValueError("Define MILO_PIPER_MODEL con la ruta de una voz Piper .onnx.")
    return VoicePipeline(
        settings=settings,
        recorder=SoundDeviceRecorder(),
        stt=FasterWhisperSTT(settings.stt_model, settings.stt_device, settings.stt_compute_type),
        llm=OllamaLLM(settings.ollama_url, settings.ollama_model),
        tts=PiperTTS(settings.piper_model),
        speaker=SoundDeviceSpeaker(),
        vad=SileroVAD(settings.vad_threshold),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    pipeline = build_pipeline(Settings.from_env())
    print("PTT listo. Pulsa Intro para empezar; vuelve a pulsarlo para terminar. Escribe q para salir.")
    while input("> ").strip().lower() != "q":
        stop_recording = threading.Event()
        worker = threading.Thread(target=pipeline.run_turn, args=(stop_recording,), daemon=True)
        worker.start()
        input("Grabando… pulsa Intro para terminar (o espera el máximo configurado). ")
        stop_recording.set()
        worker.join()


if __name__ == "__main__":
    main()
