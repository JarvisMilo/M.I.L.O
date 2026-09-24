# M.I.L.O. — Nivel 01: haz que hable

## Plataforma orientada a eventos (migración incremental)

```text
┌─────────────── Superficies sin cerebro ────────────────┐
│ escritorio / móvil / terminal: captura, muestra, envía │
└───────────────────────────┬────────────────────────────┘
                            │ APIs estables
                    ┌───────▼────────┐       eventos       ┌──────────────┐
                    │ Gateway local  │────────────────────►│ EventBus     │
                    │ capacidades    │                     │ reintentos   │
                    │ auditoría/HITL │◄────────────────────│ suscriptores │
                    └───┬────┬────┬──┘                     └──────────────┘
                        │    │    │
             ┌──────────▼┐ ┌─▼────▼──┐ ┌──────────▼────────┐
             │ voz       │ │ memoria │ │ tools/agentes/dev. │
             │ pipeline  │ │ SQLite  │ │ adaptadores locales│
             └───────────┘ └─────────┘ └───────────────────┘
```

### Contratos estables

* `Gateway.voice_turn`, `memory_search`, `memory_save`, `invoke_tool`,
  `run_agents` y `device_action` son las únicas APIs de plataforma; cada una
  exige una `Principal` con la `Capability` mínima necesaria.
* `Event(type, payload, source, correlation_id)` es el contrato de integración
  asíncrona. El bus entrega eventos con reintentos acotados y no convierte un
  error de suscriptor en una caída del gateway.
* `VoiceAPI` y `DeviceAPI` son puertos de proveedor; escritorio y móvil son
  superficies que llaman al gateway, nunca réplicas del orquestador ni de la
  lógica de dominio.
* `SQLitePlatformState` persiste estado operativo y auditoría. `health()`
  ejecuta comprobaciones registradas y aísla sus fallos.
* Acciones de dispositivo son de alto impacto por defecto. Tools con
  `CONFIRMATION_REQUIRED` también exigen capacidad `HIGH_IMPACT` y confirmación
  humana antes de que el gateway las ejecute.

### Plan de migración

1. **Ahora:** envolver los adaptadores existentes sin modificar el pipeline;
   ejecutar el gateway y su EventBus dentro del proceso local.
2. **Después:** mover las superficies de escritorio/móvil a clientes finos que
   usen las APIs del gateway y consuman eventos por `correlation_id`.
3. **Escala:** sustituir implementaciones de puertos por procesos o cloud sólo
   cuando mejoren calidad o latencia de forma medible; voz, memoria frecuente y
   auditoría permanecen locales por defecto.
4. **Operación:** añadir health checks de cada proveedor y suscriptores de
   auditoría; mantener los contratos para evitar un rewrite total.

## Checklist antes de subir de nivel

| Criterio | Estado actual | Verificación |
| --- | --- | --- |
| La capa funciona sin pasos manuales ocultos | Listo para prueba integrada | `python main.py` con los proveedores configurados; las pruebas usan dobles deterministas. |
| Latencia y fallos medibles | Implementado | Logs por etapa/turno, eventos, auditoría y health checks. |
| Permisos claros por acción | Implementado | `Gateway` exige capacidades y registra denegaciones. |
| Cambio de proveedor | Implementado | Protocolos de voz, dispositivo, memoria, tools y agentes desacoplados. |
| Desactivación del sistema | Implementado | `MILO_ENABLED=false` evita arrancar el PTT; `Gateway.set_enabled()` requiere `SYSTEM_CONTROL` y bloquea nuevas acciones. |
| El siguiente nivel resuelve un problema real | Pendiente de decisión | No añadir nivel hasta elegir una necesidad observable y medir su impacto. |

Asistente de voz **local-first** y modular con *push-to-talk* (PTT). El flujo
del primer nivel es deliberadamente pequeño:

```text
micrófono → VAD → STT → LLM → TTS → altavoz
```

No incluye UI, memoria, agentes, wake word ni servicios de infraestructura. El
PTT se estabiliza primero; `openWakeWord` puede añadirse más adelante como otro
disparador de la misma máquina de estados.

## Herramientas seguras

El LLM puede solicitar herramientas, pero nunca ejecuta código, comandos ni
rutas arbitrarias. Sólo se aceptan nombres registrados y argumentos validados
contra un esquema JSON. Las herramientas incluidas son de sólo lectura:
`get_current_time`, `calculate` (aritmética limitada, sin `eval`) y
`get_runtime_info`. Cada llamada registra herramienta, argumentos, resultado,
duración y error. Una herramienta futura marcada como irreversible o
destructiva requiere una devolución explícita de `confirm_tool` antes de
ejecutarse; si no existe o rechaza la llamada, se bloquea.

## Memoria persistente

SQLite es la fuente de verdad local en `.milo/memory.sqlite3`. Cada recuerdo
tiene `id`, `type` (`fact`, `preference` o `event`), `content`, `timestamp`,
`origin` y `confidence`. El almacén proporciona guardar, buscar, actualizar y
borrar. Antes de responder recupera por coincidencia de términos un máximo de
`MILO_MEMORY_MAX_RESULTS` recuerdos y registra el id, tipo y motivo de cada
selección. La interfaz `EmbeddingMemoryRetriever` queda disponible para migrar
la recuperación a embeddings/Qdrant cuando el volumen lo justifique.

Los eventos de conversación no secretos se guardan automáticamente. Patrones
de contraseñas, tokens, API keys y credenciales se rechazan por defecto, tanto
al guardar como al actualizar; no se conservan salvo una decisión explícita de
quien llame al almacén.

## Multiagente mínimo

`MultiAgentOrchestrator` sólo enruta, mantiene `WorkflowState` y resume; no
invoca tools. Incluye dos especialistas: `math` (sólo `calculate`) y `system`
(sólo `get_current_time` y `get_runtime_info`). Las solicitudes conocidas usan
workflows deterministas; las desconocidas sólo pasan a un `DynamicRouter`
inyectado. Cada workflow conserva checkpoints y trazas. Un especialista no
puede usar tools ajenas salvo que el orquestador las delegue explícitamente.
Toda tool marcada como acción externa debe declarar `CONFIRMATION_REQUIRED` y
recibir aprobación humana antes de ejecutarse.

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
| `MILO_MEMORY_DB` | `.milo/memory.sqlite3` | Archivo SQLite de memoria. |
| `MILO_MEMORY_MAX_RESULTS` | `3` | Máximo de recuerdos recuperados por turno. |

> `sounddevice` usa PortAudio del sistema. Si falta, instale el paquete de
> PortAudio de su distribución antes de ejecutar el asistente.
