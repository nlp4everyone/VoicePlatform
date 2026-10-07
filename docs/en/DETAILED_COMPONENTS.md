# Core Components

## ASRService (`app/services/deployments/asr_deployment.py`)

Ray Serve deployment that owns the FastAPI ingress and the batch transcription logic.

**Deployment config** (set in `config/serve.yaml`, not in the decorator):
- `num_replicas` — number of independent replicas; each replica holds one model copy
- `ray_actor_options.num_gpus` — GPU resources reserved per replica
- `ray_actor_options.num_cpus` — one CPU reserved per audio decode thread (keep ≥ `DECODE_WORKERS`)
- `max_ongoing_requests` — maximum in-flight requests per replica before Ray Serve applies backpressure

**`__init__()`**
- Re-applies `ray.serve` log level to INFO (Ray Serve resets it during actor initialization)
- Calls `RecognizerFactory.create()` to load the ASR model
- Creates a `ThreadPoolExecutor(max_workers=DECODE_WORKERS)` for audio decoding — an explicit, bounded pool instead of asyncio's default executor
- Creates a `ThreadPoolExecutor(max_workers=1)` — a single dedicated thread keeps all GPU work serialized and prevents CUDA context migration across threads

**`transcribe_audio()`** — FastAPI endpoint handler
- Validates the requested model matches the loaded model
- Validates audio format via MIME check
- Decodes the audio on the decode executor; undecodable or empty audio is rejected with `InvalidAudioException` for this request only
- Calls `batched_transcribe` directly (no deployment handle) so each request holds a single `max_ongoing_requests` slot, and awaits the result
- Formats the response based on `timestamp_granularity`

**`batched_transcribe()`** — `@serve.batch` handler
- Receives a batch of `(waveform, granularity)` pairs aggregated by Ray Serve; audio is already decoded, so one bad upload cannot fail the batch
- Dispatches to `process_batch_transcription()` on the GPU executor

---

## RecognizerFactory (`app/services/asr/factory.py`)

Thin factory with class-level singleton storage. `create(model_name, device)`:
1. Logs a warning if `model_name` does not start with `"nvidia/parakeet"` (but proceeds anyway)
2. Instantiates `ParakeetRecognizer` and returns it
3. Stores the instance in `_recognizer` — subsequent calls replace it

`get_recognizer_model()` — returns `_recognizer.model_name` or `None`.

---

## ParakeetRecognizer (`app/services/asr/nemo/recognizer.py`)

Wraps `nemo_asr.models.ASRModel` with sorting, autocast, and structured output.

**Supported models:**
- `nvidia/parakeet-ctc-0.6b-vi`
- `nvidia/parakeet-tdt-0.6b-v3`

Unknown model names raise `ValueError` at startup, so the deployment fails instead of silently serving another model.

**Initialization:**
1. Resolves device: `"auto"` → `"cuda"` if `torch.cuda.is_available()` else `"cpu"`
2. `nemo_asr.models.ASRModel.from_pretrained(model_name)` — downloads/loads from HF cache
3. `model.cuda()` if CUDA
4. `model.eval()`

**`transcribe(audio, enable_timestamps, precision=3)`**
- Normalizes single inputs to `[item]`
- If batch is multi-item tensors: sorts ascending by length, records the permutation
- Runs inference under `torch.autocast(device_type="cuda")`, enabled only when the device is CUDA
- Restores original order after inference
- Returns `List[TranscriptionResult]`; timestamps are rounded to `precision` decimal places

**Properties:** `model_name`, `supported_models`

---

## Audio Utils (`app/utils/audio/io.py`)

**`load_audio_from_bytes(audio_bytes, target_sr=16000) → (Tensor, float)`**
- `torchcodec.decoders.AudioDecoder(audio_bytes, sample_rate=target_sr, num_channels=1)` — supports MP3, WAV, FLAC, OGG and other FFmpeg formats
- Passing `bytes` (not a `BytesIO`) lets TorchCodec decode from memory without Python read callbacks
- FFmpeg downmixes to mono and resamples to `target_sr` in the same pass
- Returns `(waveform, duration_seconds)`
- Raises `InvalidAudioException` when decoding fails or the audio has no samples

**`is_audio_file(data, buffer_size=2048) → bool`**
- Uses `python-magic` to sniff MIME type from the first 2048 bytes
- Returns `True` for `audio/*` and `application/ogg`; returns `False` otherwise and on any exception

**`estimate_audio_duration(audio_bytes) → float`**
- Uses `soundfile.info()` for a lightweight duration estimate without full decode

---

## Batch Processing Helper (`app/utils/transcription/helper.py`)

**`process_batch_transcription(asr_model, audio_data, timestamp_granularities, split_mixed_batch=True)`**

Dispatches to 1 or 2 GPU calls based on the batch composition:

```
ts_indices, no_ts_indices = get_timestamp_indices(timestamp_granularities)

Pure no-ts:  asr_model.transcribe(all, enable_timestamps=False)         → 1 GPU call
Pure ts:     asr_model.transcribe(all, enable_timestamps=True)          → 1 GPU call
Mixed (split_mixed_batch=True):
    asr_model.transcribe(ts_items, enable_timestamps=True)              → GPU call 1
    asr_model.transcribe(no_ts_items, enable_timestamps=False)          → GPU call 2
    merge results back to original indices
Mixed (split_mixed_batch=False):
    asr_model.transcribe(all, enable_timestamps=True)                   → 1 GPU call
    strip .words / .segments from no-ts results
```

**`get_timestamp_indices(timestamp_granularities) → (ts_indices, no_ts_indices)`**
- Partitions a list of granularity strings (`None`, `"word"`, `"segment"`) into two index lists

**`get_transcription_type(type) → TranscriptionType`**
- `None` → `Text`, `"word"` → `Word`, `"segment"` → `Segment`

---

## Response Schemas (`app/schema/transcription/`)

| Class | Used when | Fields |
|---|---|---|
| `TranscriptionResponse` | `timestamp_granularity=None` | `text`, `usage` (token counts) |
| `WordResponse` | `timestamp_granularity="word"` | `text`, `language`, `duration`, `usage` (seconds), `words` |
| `SegmentResponse` | `timestamp_granularity="segment"` | `text`, `language`, `duration`, `usage` (seconds), `segments` |

`TranscribedWord`: `start`, `end`, `word`
`AdvancedTranscribedSegment`: `id`, `start`, `end`, `text`

---

## Configuration Loader (`app/utils/config_loader/toml_loader.py`)

Reads `config/config.toml` and exposes sections as dicts. Used by `app/core/config/*.py` to populate module-level constants imported via `*`.

---

## Exception Handlers (`app/exceptions/`)

| Exception | Trigger | HTTP response |
|---|---|---|
| `TranscriptedModelNotFoundException` | Requested model ≠ loaded model | 404 |
| `UnsupportedAudioFormatException` | MIME check fails | 415 |
| `InvalidAudioException` | Audio cannot be decoded or has no samples | 400 |

All are registered via `asr_app.add_exception_handler()` and handled by `common_exception_handler`.
