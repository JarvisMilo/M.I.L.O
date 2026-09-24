# M.I.L.O. — Nivel 01: haz que hable

Asistente de voz **local-first** y modular con *push-to-talk* (PTT). El flujo
del primer nivel es deliberadamente pequeño:

```text
micrófono → VAD → STT → LLM → TTS → altavoz
```

No incluye UI, memoria, agentes, wake word ni servicios de infraestructura. El
PTT se estabiliza primero; `openWakeWord` puede añadirse más adelante como otro
disparador de la misma máquina de estados.

## Arquitectura

Cada etapa depende de una interfaz (`Protocol`), no de una implementación de
proveedor. Cambiar Faster-Whisper por whisper.cpp, Ollama por un proveedor
cloud, o Piper por otro TTS no cambia el orquestador.

```text
main.VoicePipeline
 ├── audio.AudioRecorder      → graba WAV desde el micrófono
 ├── vad.VADProvider          → descarta grabaciones sin habla
 ├── stt.STTProvider          → texto transcrito
 ├── llm.LLMProvider          → respuesta de texto
 ├── tts.TTSProvider          → WAV sintetizado
 └── audio.AudioSpeaker       → reproduce el WAV
```

La máquina de estados es `IDLE → RECORDING → ANALYZING → TRANSCRIBING → THINKING →
SYNTHESIZING → PLAYING → IDLE`; cualquier error pasa por `ERROR` y vuelve a
`IDLE`. Se registra la latencia de cada etapa y del turno completo.

## Árbol de archivos

```text
.
├── main.py       # PTT, orquestación y máquina de estados
├── config.py     # configuración tipada desde variables de entorno
├── audio.py      # interfaces y adaptadores sounddevice
├── vad.py        # interfaz VAD y adaptador Silero VAD
├── stt.py        # interfaz STT y adaptador Faster-Whisper
├── llm.py        # interfaz LLM y adaptador local Ollama
├── tts.py        # interfaz TTS y adaptador Piper
├── requirements.txt
└── tests/
    └── test_pipeline.py
```

## Puesta en marcha

Se requiere Python 3.10+. Cree un entorno e instale las dependencias base:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Instale Piper y asegúrese de que `piper` está en `PATH`; descargue además una
voz `.onnx`. Arranque Ollama y descargue un modelo, por ejemplo `ollama pull
llama3.2`. Después configure y ejecute:

```bash
export MILO_PIPER_MODEL=/ruta/a/voz.onnx
export MILO_OLLAMA_MODEL=llama3.2
python main.py
```

Pulse Intro para iniciar cada turno PTT y vuelva a pulsarlo para terminar la
grabación antes del límite configurado. Pulse `q` y después Intro para salir.
Las grabaciones y respuestas WAV se guardan en `.milo/audio/` por defecto.

### Variables de entorno

| Variable | Predeterminado | Uso |
| --- | --- | --- |
| `MILO_RECORD_SECONDS` | `5` | Límite de seguridad para cada pulsación PTT. |
| `MILO_SAMPLE_RATE` | `16000` | Frecuencia de grabación WAV. |
| `MILO_VAD_THRESHOLD` | `0.5` | Umbral de detección de habla de Silero. |
| `MILO_STT_MODEL` | `base` | Modelo Faster-Whisper. |
| `MILO_STT_DEVICE` | `auto` | Dispositivo Faster-Whisper. |
| `MILO_STT_COMPUTE_TYPE` | `int8` | Tipo de cómputo Faster-Whisper. |
| `MILO_OLLAMA_URL` | `http://localhost:11434` | URL del servidor Ollama. |
| `MILO_OLLAMA_MODEL` | `llama3.2` | Modelo de conversación local. |
| `MILO_PIPER_MODEL` | *(obligatorio)* | Ruta a la voz ONNX de Piper. |
| `MILO_DATA_DIR` | `.milo/audio` | Directorio de WAV temporales. |

> `sounddevice` usa PortAudio del sistema. Si falta, instale el paquete de
> PortAudio de su distribución antes de ejecutar el asistente.
