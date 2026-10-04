# Technical Overview

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     CLIENT (OpenAI SDK / curl)                   │
│            POST /v1/audio/transcriptions  (multipart/form-data)  │
└───────────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                  Ray Serve  (serve.start)                        │
│  host: RAY_HOST   port: RAY_PORT                                 │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  ASRService  (@serve.deployment)                           │  │
│  │  num_replicas=NUM_REPLICAS                                 │  │
│  │  num_gpus=NUM_GPUS  per replica                            │  │
│  │  max_ongoing_requests=MAX_ONGOING_REQUESTS                 │  │
│  │  num_cpus=DECODE_WORKERS                                   │  │
│  │                                                            │  │
│  │  FastAPI ingress (@serve.ingress)                          │  │
│  │   POST /v1/audio/transcriptions                            │  │
│  │       │ validate model, read bytes, MIME check             │  │
│  │       │ run_in_executor(decode pool, load_audio_from_bytes)│  │
│  │       │ self.batched_transcribe(waveform, granularity)     │  │
│  │       ▼                                                    │  │
│  │  @serve.batch                                              │  │
│  │  batched_transcribe(waveforms, granularities)              │  │
│  │       │ run_in_executor(GPU thread, process_batch)         │  │
│  │       ▼                                                    │  │
│  │  process_batch_transcription()                             │  │
│  │       │ split ts / no-ts sub-batches                       │  │
│  │       ▼                                                    │  │
│  │  ParakeetRecognizer.transcribe()                           │  │
│  │       │ sort by length → autocast → model.transcribe()     │  │
│  │       ▼                                                    │  │
│  │  NeMo ASRModel  (GPU)                                      │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Processing Pipeline

### Stage 1 — Request validation

`transcribe_audio()` receives the uploaded file and:
1. Checks that `model` matches the loaded `asr_model.model_name` — raises `TranscriptedModelNotFoundException` otherwise.
2. Reads all audio bytes into memory.
3. Validates MIME type via `python-magic` — raises `UnsupportedAudioFormatException` on non-audio input.

### Stage 2 — Audio decoding (per request, CPU)

`load_audio_from_bytes()` runs on the replica's dedicated decode pool (`ThreadPoolExecutor(max_workers=DECODE_WORKERS)`), so concurrent requests decode in parallel without blocking the event loop:
- `AudioDecoder(audio_bytes, sample_rate=16000, num_channels=1)` (TorchCodec) decodes straight from the bytes, downmixing to mono and resampling to 16kHz in one FFmpeg pass
- Returns `(waveform: torch.Tensor, duration: float)`

Decoding happens before batching, so a corrupted or empty file raises `InvalidAudioException` (400) for that request only. Raw bytes are freed immediately after decoding.

### Stage 3 — Batch queuing

The decoded waveform and `timestamp_granularity` are passed to `batched_transcribe` with a direct method call. Ray Serve accumulates concurrent calls within the replica into a batch (up to `MAX_BATCH_SIZE`) waiting at most `BATCH_WAIT_TIMEOUT_S` seconds. Calling the method directly instead of through a deployment handle keeps each request at a single `max_ongoing_requests` slot.

### Stage 4 — GPU transcription (dedicated thread)

`process_batch_transcription()` runs in a single-thread `ThreadPoolExecutor` (one per replica) to keep GPU work on a single thread and avoid CUDA context migration:

| Batch composition | GPU calls |
|---|---|
| All no-timestamp | 1 call, `timestamps=False` |
| All with timestamps | 1 call, `timestamps=True` |
| Mixed | 2 sub-calls — ts items + no-ts items separately |

Within each GPU call, `ParakeetRecognizer.transcribe()`:
1. Sorts tensors by length (ascending) — minimizes padding waste across NeMo's internal sub-batches.
2. Runs `model.transcribe()` under `torch.autocast(device_type="cuda")` (FP16, CUDA only).
3. Restores original order before returning.

### Stage 5 — Response formatting

Based on `timestamp_granularity`:
- `None` → `TranscriptionResponse` — text + token usage
- `"word"` → `WordResponse` — text, detected language, duration, word-level timestamps
- `"segment"` → `SegmentResponse` — text, detected language, duration, segment-level timestamps

Language detection uses `pycld2` and runs only for timestamp responses.

---

## Configuration

All parameters live in `config/config.toml`. Changes require a container restart.

### `[serving]`

| Key | Default | Description |
|---|---|---|
| `NUM_GPUS` | `1` | GPUs allocated per replica |
| `NUM_REPLICAS` | `1` | Number of ASRService replicas |
| `MAX_ONGOING_REQUESTS` | `16` | Max in-flight requests per replica |
| `MAX_BATCH_SIZE` | `8` | Max items per GPU batch |
| `BATCH_WAIT_TIMEOUT_S` | `0.1` | Max wait time to fill a batch (seconds) |
| `DECODE_WORKERS` | `4` | Audio decode threads per replica (also the replica's Ray `num_cpus`) |

### `[asr]`

| Key | Default | Description |
|---|---|---|
| `ASR_MODEL_NAME` | `nvidia/parakeet-ctc-0.6b-vi` | HuggingFace model identifier |
| `ASR_DEVICE` | `auto` | `auto` / `cuda` / `cpu` |

### `[system]`

| Key | Description |
|---|---|
| `RAY_HOST` | HTTP host for Ray Serve |
| `RAY_PORT` | HTTP port for Ray Serve |
| `DEPLOYMENT_NAME` | Name registered with Ray Serve |

### Environment (`.env` / `docker-compose.yml`)

| Variable | Description |
|---|---|
| `RAY_FASTAPI_PORT` | Host port mapped to Ray Serve HTTP (default `8000`) |
| `RAY_DASHBOARD_PORT` | Host port mapped to Ray dashboard (default `8265`) |
| `RAY_LOG_LEVEL` | Ray internal log level (default `WARNING`) |
| `RAY_SERVE_LOG_TO_STDERR` | Forward Serve logs to stderr (default `1`) |
| `HF_HOME` | HuggingFace cache directory |

---

## Repository Structure

```
VoicePlatform/
├── app/
│   ├── app.py                          # Entry point: ray.init, serve.start, ASRService.bind
│   ├── core/config/
│   │   ├── asr.py                      # ASR_MODEL_NAME, ASR_DEVICE
│   │   ├── serving.py                  # NUM_GPUS, NUM_REPLICAS, MAX_BATCH_SIZE, …
│   │   └── system.py                   # RAY_HOST, RAY_PORT, DEPLOYMENT_NAME
│   ├── services/
│   │   ├── asr/
│   │   │   ├── factory.py              # RecognizerFactory — creates ParakeetRecognizer
│   │   │   └── nemo/recognizer.py      # ParakeetRecognizer — wraps NeMo ASRModel
│   │   └── deployments/
│   │       └── asr_deployment.py       # ASRService — Ray Serve deployment + FastAPI
│   ├── utils/
│   │   ├── audio/io.py                 # load_audio_from_bytes, is_audio_file
│   │   └── transcription/helper.py     # process_batch_transcription
│   └── schema/transcription/           # TranscriptionResult, response types
├── config/config.toml                  # All runtime configuration
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
└── examples/                           # Usage examples
```
