"""Punto de entrada PTT para el primer pipeline de voz de M.I.L.O."""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Callable

from audio import AudioRecorder, AudioSpeaker, SoundDeviceRecorder, SoundDeviceSpeaker
from config import Settings
from llm import LLMProvider, OllamaLLM
from memory import MemoryMatch, SQLiteMemoryStore
from stt import FasterWhisperSTT, STTProvider
from tts import PiperTTS, TTSProvider
from tools import ConfirmationCallback, ToolCall, ToolRegistry, default_registry
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
    tool_registry: ToolRegistry = field(default_factory=default_registry)
    confirm_tool: ConfirmationCallback | None = None
    memory: SQLiteMemoryStore | None = None
    memory_max_results: int = 3
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

    def _memory_context(self, transcript: str) -> str:
        if self.memory is None:
            return transcript
        matches: list[MemoryMatch] = self.memory.search(transcript, self.memory_max_results)
        for match in matches:
            logger.info("Memoria seleccionada id=%s type=%s motivo=%s", match.record.id, match.record.type, match.reason)
        if not matches:
            return transcript
        context = "\n".join(f"- ({match.record.type}) {match.record.content}" for match in matches)
        return f"Contexto relevante y persistente:\n{context}\n\nConsulta actual: {transcript}"

    def _save_event(self, content: str, origin: str) -> None:
        if self.memory is None:
            return
        try:
            self.memory.save("event", content, origin, confidence=0.5)
        except ValueError as error:
            logger.info("Evento no guardado en memoria: %s", error)

    def _respond(self, prompt: str) -> str:
        """Ejecuta exclusivamente llamadas solicitadas a herramientas registradas."""
        if not hasattr(self.llm, "respond_with_tools") or not hasattr(self.llm, "respond_after_tools"):
            return self.llm.respond(prompt)
        response = self.llm.respond_with_tools(prompt, self.tool_registry.specifications())
        calls = []
        for raw_call in response.tool_calls:
            function = raw_call.get("function", {})
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            calls.append(ToolCall(function.get("name", ""), arguments, raw_call.get("id", "")))
        if not calls:
            return response.content
        results = [self.tool_registry.invoke(call, self.confirm_tool).as_message() for call in calls]
        return self.llm.respond_after_tools(prompt, response, results)

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
            self._save_event(transcript, "user")
            self._transition(PipelineState.THINKING)
            prompt = self._memory_context(transcript)
            answer = self._timed("llm", lambda: self._respond(prompt))
            logger.info("M.I.L.O.: %s", answer)
            self._save_event(answer, "assistant")
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
        memory=SQLiteMemoryStore(settings.memory_db),
        memory_max_results=settings.memory_max_results,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = Settings.from_env()
    if not settings.enabled:
        logging.warning("M.I.L.O. está desactivado por MILO_ENABLED=false.")
        return
    pipeline = build_pipeline(settings)
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
