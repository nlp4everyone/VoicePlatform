# Design Decisions

---

## 1. Ray Serve for Deployment

### Description

The ASR model is served through **Ray Serve** rather than a bare FastAPI/Uvicorn process. Ray Serve manages the lifecycle of `ASRService` replicas, GPU resource allocation, request routing, and automatic batching.

### Pros

- **Multi-replica scaling** — set `NUM_REPLICAS > 1` to run multiple independent model copies across GPUs with no code changes.
- **Built-in autobatching** — `@serve.batch` aggregates concurrent requests before the GPU call without manual queue management.
- **GPU resource isolation** — each replica declares `num_gpus` via `ray_actor_options`; Ray guarantees the allocation.
- **Dashboard** — Ray Serve dashboard provides live replica status, request throughput, and error rates.

### Cons

- **Higher startup overhead** — Ray cluster initialization and model loading take longer than a plain Uvicorn worker.
- **Operational complexity** — Ray adds its own logging, actor model, and process management on top of FastAPI.

### Alternatives considered

| Option | Reason not chosen |
|---|---|
| Uvicorn + FastAPI only | No built-in batching or multi-GPU replica management |
| Triton Inference Server | Heavier to configure; NeMo models require additional export steps |
| TorchServe | Less flexible for custom pre/post processing; smaller ecosystem around NeMo |

---

## 2. Automatic Request Batching (`@serve.batch`)

### Description

`batched_transcribe` is decorated with `@serve.batch(max_batch_size=MAX_BATCH_SIZE, batch_wait_timeout_s=BATCH_WAIT_TIMEOUT_S)`. Ray Serve groups concurrent calls within the replica into a single list before invoking the handler.

### Pros

- **GPU utilization** — a batch of N items in one `model.transcribe()` call is significantly faster than N sequential calls.
- **Throughput vs latency trade-off is configurable** — lower `BATCH_WAIT_TIMEOUT_S` favors latency; higher favors throughput.
- **Transparent to callers** — each request sees a single `TranscriptionResult`; batching is invisible.

### Cons

- **Added latency** — a request arriving just after a batch is dispatched waits up to `BATCH_WAIT_TIMEOUT_S` before the next batch forms.
- **Unequal batch sizes** — under low load, most batches are size 1, so the throughput gain is minimal.

---

## 3. Dedicated Single-Thread GPU Executor

### Description

`process_batch_transcription()` is dispatched via `asyncio.get_event_loop().run_in_executor(self._gpu_executor, ...)` where `self._gpu_executor` is a `ThreadPoolExecutor(max_workers=1)` created once per replica.

### Pros

- **No CUDA context migration** — CUDA operations are fastest on a single thread. Moving GPU work across threads forces context switching, which adds latency.
- **Prevents pool contention** — the default `asyncio` executor is shared across all async tasks; submitting GPU work there competes with CPU-bound tasks.
- **Serialized GPU access** — a single thread guarantees sequential batches with no interleaving.

### Cons

- **No parallelism within a replica** — a very large batch that takes a long time to process cannot be overlapped with decoding the next batch.

---

## 4. Mixed-Batch Splitting

### Description

When a batch contains both timestamp and non-timestamp requests, `process_batch_transcription()` issues two GPU sub-calls (`split_mixed_batch=True`):
1. `model.transcribe(ts_items, timestamps=True)`
2. `model.transcribe(no_ts_items, timestamps=False)`

### Pros

- **Avoids unnecessary GPU→CPU logit transfer** — NeMo's CTC alignment (needed for timestamps) transfers logits from GPU to CPU. Non-timestamp items incur this cost unnecessarily in a unified call with `timestamps=True`.
- **Lower latency for non-timestamp items** — they complete without waiting for CTC alignment of timestamp items.

### Cons

- **Two GPU kernel launches** — in a batch where all items are the same type, the extra dispatch overhead is avoided; but in a mixed batch, two calls are needed.
- **Controllable** — `SPLIT_MIXED_BATCH=False` falls back to a single call with stripping if the overhead of two calls outweighs the logit transfer cost.

---

## 5. Sort-by-Length Batching

### Description

Before calling `model.transcribe()`, audio tensors are sorted by length (ascending). The permutation is recorded and the results are un-sorted before returning.

### Pros

- **Minimizes padding** — NeMo internally splits large batches into sub-batches. Grouping similar-length items together reduces the amount of zero-padding added to shorter items, which reduces wasted GPU compute.

### Cons

- **Extra sort overhead** — `O(N log N)` per batch; negligible for typical batch sizes (≤ 32).
- **Requires restoring order** — an index permutation must be tracked and reversed.

---

## 6. AMP Autocast

### Description

GPU inference runs under `torch.cuda.amp.autocast()`, which automatically casts eligible operations to FP16 (or BF16 on Ampere+).

### Pros

- **Faster inference** — FP16 tensor cores are significantly faster than FP32 on modern NVIDIA GPUs.
- **Lower VRAM usage** — smaller intermediate tensors allow larger batches to fit in memory.
- **No accuracy degradation** — NeMo's ASR models are robust to FP16 inference.

### Cons

- **Device requirement** — has no effect on CPU; only beneficial on CUDA-capable hardware.

---

## 7. Decoding from Raw Bytes with TorchCodec

### Description

`load_audio_from_bytes()` passes the uploaded bytes straight to `torchcodec.decoders.AudioDecoder(audio_bytes, sample_rate=16000, num_channels=1)`. FFmpeg decodes, downmixes to mono and resamples to 16 kHz in one pass.

### Pros

- **No Python callbacks during decode** — given `bytes`, TorchCodec reads straight from memory. A `BytesIO` (what `torchaudio.load` received before) is treated as a file-like object, so the C++ decoder calls back into Python, and takes the GIL, for every read, which limits how well the decode threads run in parallel.
- **Fewer steps** — no separate `torchaudio.transforms.Resample` or channel mean; `torchaudio.load` already delegates to TorchCodec since TorchAudio 2.9.
- **Measured** — on 16 × 60 s stereo 48 kHz MP3s, decoding took 0.25 s instead of 0.34 s with 4 threads (0.90 s instead of 1.03 s with 1 thread), with identical output (correlation 1.0000).

### Cons

- **Direct TorchCodec dependency** — the decode path now depends on the TorchCodec API instead of the `torchaudio.load` wrapper.

---

## 8. Audio Format Validation via MIME Type

### Description

`is_audio_file()` uses `python-magic` to sniff the first 2048 bytes of the uploaded file and check whether the MIME type starts with `audio/` or matches `application/ogg`.

### Pros

- **Format-agnostic** — works for any audio container (MP3, WAV, FLAC, OGG, M4A) without an explicit allowlist.
- **Fast** — reads only 2048 bytes; no full decode required.
- **Rejects non-audio early** — invalid files are rejected before audio decode and before the GPU queue.

### Cons

- **False positives for edge cases** — some valid audio containers may have non-standard MIME types (e.g. video files with audio-only streams).
- **`python-magic` dependency** — requires `libmagic` system library in the Docker image.

---

## 9. OpenAI-Compatible API

### Description

The endpoint signature mirrors OpenAI's `POST /v1/audio/transcriptions`:
- `file` — audio file (multipart)
- `model` — model identifier
- `timestamp_granularities[]` — `"word"` or `"segment"`
- `response_format` — currently `verbose_json`

### Pros

- **Drop-in compatibility** — any client using the OpenAI Python SDK can switch to this service by changing `base_url` only.
- **Familiar interface** — developers do not need to learn a new API.

### Cons

- **Subset of OpenAI spec** — `response_format` options other than `verbose_json` are accepted but not meaningfully differentiated; some OpenAI fields (e.g. `language`, `prompt`) are not supported.
