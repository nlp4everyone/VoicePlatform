# 🎤 VoicePlatform - Private Audio Transcription Service (NeMo Parakeet + Ray Serve)

A high-performance, privacy-focused batch speech recognition (ASR) service powered by NVIDIA NeMo Parakeet models. It provides a secure, local deployment for transcribing audio files without relying on cloud services, keeping all data on your own infrastructure.

The service is built with FastAPI and Ray Serve and offers an OpenAI-compatible `/v1/audio/transcriptions` endpoint, so existing applications can switch to it by changing only the base URL.

<br />

# 🧠 Core Features

### 🎯 Advanced ASR Capabilities
- High-accuracy transcription with NVIDIA NeMo Parakeet (default: `nvidia/parakeet-ctc-0.6b-vi` for Vietnamese)
- Word-level and segment-level timestamps on demand
- OpenAI-compatible API, a drop-in replacement for `client.audio.transcriptions.create()`
- Audio format validation (MIME-type check via `python-magic`) before any processing

### 🚀 Performance Optimizations
- GPU-accelerated inference, roughly 47x faster than real time on a single RTX 3060 (see Real-Time Factor under Examples)
- Automatic request batching: Ray Serve groups concurrent requests into GPU batches (`MAX_BATCH_SIZE`, `BATCH_WAIT_TIMEOUT_S`)
- Mixed-batch splitting: timestamp and non-timestamp requests in one batch run as separate GPU sub-calls, avoiding unnecessary logit transfers
- Sort-by-length batching to minimize padding waste, and AMP autocast (`torch.cuda.amp.autocast`) for mixed-precision inference
- Cached resampler (`torchaudio.transforms.Resample` cached per `(src_sr, tgt_sr)` pair) and a dedicated single-thread GPU executor
- Multi-replica support: scale by setting `NUM_REPLICAS` in `config/config.toml`

### 🔒 Privacy & Security
- Complete local deployment - no audio leaves your infrastructure
- No API keys or external service dependencies required
- Models are downloaded once into your Hugging Face cache and never committed to the repository

### 🔌 Developer-Friendly API
- RESTful API with OpenAI-compatible endpoints and interactive docs at `/docs`
- Ready-to-run examples for single and concurrent requests
- Docker-based deployment driven by a single `Makefile`
- Ray dashboard on port `8265` for monitoring replicas and requests

<br />

# 📡 Capabilities

| Capability | Protocol | Endpoint | Status |
|------------|----------|----------|--------|
| Transcription | HTTP | `POST /v1/audio/transcriptions` | ✅ |
| Transcription with word timestamps | HTTP | `POST /v1/audio/transcriptions` with `timestamp_granularities=["word"]` | ✅ |
| Transcription with segment timestamps | HTTP | `POST /v1/audio/transcriptions` with `timestamp_granularities=["segment"]` | ✅ |
| Transcription (streaming text) | HTTP (SSE) | - | ❌ |
| Transcription (base64 audio) | HTTP | - | ❌ |
| Realtime (live audio) | WebSocket | - | ❌ |

**Legend**: ✅ supported · 🚧 in progress · ❌ not supported

> ℹ️ This branch is a batch service: a whole file is uploaded and the full transcript is returned. For live audio over WebSocket see the `vllm/qwen3_asr` branch.

<br />

# 🏗️ Architecture

```
Client (OpenAI SDK / HTTP)
        │
        │  POST /v1/audio/transcriptions
        ▼
FastAPI  (ASRService — Ray Serve ingress)
        │  validate model name, read bytes, validate audio MIME
        │
        │  .batched_transcribe.remote(audio_bytes, granularity)
        ▼
Ray Serve batch queue  (MAX_BATCH_SIZE, BATCH_WAIT_TIMEOUT_S)
        │
        │  load_audio_from_bytes() × N  [parallel asyncio.to_thread]
        │  process_batch_transcription() [dedicated GPU ThreadPoolExecutor]
        ▼
ParakeetRecognizer
        │  sort by length → autocast → model.transcribe()
        ▼
NeMo ASRModel  (nvidia/parakeet-ctc-0.6b-vi  or  nvidia/parakeet-tdt-0.6b-v3)
        │
        ▼
TranscriptionResult  →  TranscriptionResponse / WordResponse / SegmentResponse
```

<br />

# 🛠️ Prerequisites

1. **💻 Hardware Requirements**
   - 🖥️ **CPU**: x86_64 (AVX2 support recommended)
   - 🧠 **RAM**: 16GB minimum (the container may use up to 16GB of shared memory)
   - 🎮 **GPU**: NVIDIA GPU with CUDA support (recommended for optimal performance)
   - 💾 **Storage**: SSD recommended; the image is large (~30GB with NeMo and PyTorch) plus the model cache

2. **🛠️ Software Dependencies**
   - 🐳 **Docker and Docker Compose**
   - 🎮 **NVIDIA Container Toolkit** (for GPU support)
   - ⚡ **NVIDIA driver** compatible with CUDA 12.8, the version of the `nvidia/cuda:12.8.1` base image (a host CUDA Toolkit is not required, CUDA ships inside the image)
   - 📦 **[uv](https://docs.astral.sh/uv/)** (optional, only for running or developing outside Docker: `uv sync`)

<br />

# 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/nlp4everyone/VoicePlatform.git
   cd VoicePlatform
   ```

2. **Switch to the correct branch**
   ```bash
   git fetch
   git checkout ray/nvidia_asr
   ```

3. **Set up environment configuration**
   ```bash
   # Create .env from .env.sample (skipped if .env already exists)
   make env
   # Edit the .env file to customize ports
   # nano .env
   ```

4. **Configure the serving parameters** in `config/config.toml` to match your hardware (see Configuration below):
   ```toml
   [serving]
   NUM_GPUS = 1
   NUM_REPLICAS = 1
   MAX_ONGOING_REQUESTS = 16
   MAX_BATCH_SIZE = 8
   BATCH_WAIT_TIMEOUT_S = 0.1

   [asr]
   ASR_MODEL_NAME = "nvidia/parakeet-ctc-0.6b-vi"
   ASR_DEVICE = "auto"
   ```

5. **Build and start the service**
   ```bash
   # Build the image (dependencies are installed from uv.lock) and start in the background
   make up
   # Or run in the foreground
   make start
   ```
   The first start also downloads the model into `~/.cache/huggingface`, which can take a few minutes. Docker is invoked with `sudo` by default. Override with `make up DOCKER=docker` if your user is in the `docker` group.

6. **Verify the service is running**
   ```bash
   # Check that the API is serving
   make health
   # View service logs
   make logs
   ```

   Other useful targets: `make down`, `make restart`, `make ps`, `make clean`. Run `make help` for the full list.

7. **Access the service**
   - 🔌 **API Endpoint**: http://localhost:8005/v1 - OpenAI-compatible API (port set by `RAY_FASTAPI_PORT` in `.env`)
   - 📚 **API Documentation**: http://localhost:8005/docs - Interactive API docs
   - 📊 **Ray Dashboard**: http://localhost:8265 - Replicas, requests and logs

<br />

# 📝 Examples

All examples send `resources/sample_vi.wav` to `http://localhost:8005/v1`. Install the client dependency first: `pip install openai`. Run them from the project root, since the audio path is relative.

## 📄 Audio Transcription

**Transcription**: HTTP · `POST /v1/audio/transcriptions`

Check out the `examples/audio_transcription_example.py` file to see how to:
- 🎯 **Transcribe** an audio file using the OpenAI SDK
- 🕒 **Request** word-level timestamps with `timestamp_granularities=["word"]`
- ⏱️ **Measure** transcription time

```bash
python examples/audio_transcription_example.py
```

A minimal client looks like this:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8005/v1", api_key="token")

with open("resources/sample_vi.wav", "rb") as f:
    result = client.audio.transcriptions.create(
        model="nvidia/parakeet-ctc-0.6b-vi",
        file=f,
        timestamp_granularities=["word"]
    )
print(result)
```

### ⏱️ Real-Time Factor (RTF)

RTF = processing time ÷ audio duration. A value below 1 means faster than real time.

Measured with the OpenAI SDK (`client.audio.transcriptions.create`) on `resources/sample_vi.wav` (7.63 s of audio): one warm-up request, then 5 timed requests.

| Model | GPU | Mode | Audio | Mean time | RTF (mean) | RTF (min – max) |
|-------|-----|------|-------|-----------|------------|-----------------|
| `nvidia/parakeet-ctc-0.6b-vi` | NVIDIA RTX 3060 | Text only | 7.63 s | 0.162 s | 0.021 | 0.021 – 0.022 |
| `nvidia/parakeet-ctc-0.6b-vi` | NVIDIA RTX 3060 | Word timestamps | 7.63 s | 0.192 s | 0.025 | 0.024 – 0.026 |

> ℹ️ Times are end-to-end over localhost (upload + decode + inference) with `NUM_REPLICAS=1` and the defaults from `config/config.toml`; the first (warm-up) request took 0.177 s. RTF depends on the model, GPU and concurrent load, so re-measure on your own hardware.

## 🚦 Concurrent Requests

The `examples/concurrent_requests_example.py` sends several requests at once with the async OpenAI client so you can see Ray Serve batching in action.

```bash
python examples/concurrent_requests_example.py
```

Concurrent requests are grouped into GPU batches (up to `MAX_BATCH_SIZE`), so the total time barely grows with the number of requests. Measured with the same audio, one replica, 3 timed rounds after a warm-up round:

| Concurrent requests | Wall time per round | Mean latency per request | Aggregate RTF |
|---------------------|---------------------|--------------------------|---------------|
| 4 | 0.206 s | 0.204 s | 0.0068 |
| 8 | 0.187 s | 0.182 s | 0.0031 |

> Aggregate RTF = wall time ÷ (number of requests × audio duration), a measure of throughput.

> ⚠️ **Known limitation**: keep concurrency at or below `MAX_ONGOING_REQUESTS / 2` (8 with the default of 16). Each HTTP request occupies one slot while it calls `batched_transcribe` through the deployment's own handle, which needs a second slot. With 16 simultaneous requests at the default setting, all slots are taken by the outer requests and the service stalls (`Failed to route request after N attempts` in the logs) until the clients disconnect. If you expect more concurrent clients, raise `MAX_ONGOING_REQUESTS` to at least twice that number.

## 🎵 Sample Audio Files

Sample audio files are included in `resources/` (`sample_vi.wav` for Vietnamese, `sample_en.wav` for English) for testing the transcription capabilities immediately after deployment.

<br />

# ⚙️ Configuration

## 🎛️ Serving Parameters

Serving behavior is set in `config/config.toml`. Changes take effect after `make restart`.

| Key | Default | Description |
|-----|---------|-------------|
| `[serving] NUM_GPUS` | `1` | GPUs reserved per replica (fractions such as `0.5` let replicas share a GPU) |
| `[serving] NUM_REPLICAS` | `1` | Number of model replicas |
| `[serving] MAX_ONGOING_REQUESTS` | `16` | Maximum in-flight requests per replica (see the known limitation under Concurrent Requests) |
| `[serving] MAX_BATCH_SIZE` | `8` | Maximum number of requests grouped into one GPU batch |
| `[serving] BATCH_WAIT_TIMEOUT_S` | `0.1` | How long a batch waits to fill before it is dispatched |
| `[asr] ASR_MODEL_NAME` | `nvidia/parakeet-ctc-0.6b-vi` | NeMo Parakeet model to load (name must start with `nvidia/parakeet`) |
| `[asr] ASR_DEVICE` | `auto` | `auto`, `cuda` or `cpu` |
| `[asr] SPLIT_MIXED_BATCH` | `true` | Run timestamp and non-timestamp requests as separate GPU sub-calls |

## 🌍 Environment Variables

Set these in your `.env` file (created by `make env`):

```bash
# 🌐 Network configuration
RAY_FASTAPI_PORT=8005                   # Host port of the inference API (container listens on 8000)
RAY_DASHBOARD_PORT=8265                 # Host port of the Ray dashboard
```

<br />

# 📚 Documentation

- [Technical Overview](TECHNICAL_OVERVIEW.md) — full architecture, processing pipeline, configuration reference
- [Flow](FLOW.md) — step-by-step startup and per-request flow
- [Detailed Components](DETAILED_COMPONENTS.md) — component internals and API reference
- [Design Decisions](DESIGN_DECISIONS.md) — rationale and trade-offs for each major choice

<br />

# 📋 To-Do List
- [x] 🚀 Batch ASR service on Ray Serve with NeMo Parakeet
- [x] 🔌 OpenAI-compatible `/v1/audio/transcriptions` endpoint
- [x] 🕒 Word-level and segment-level timestamps
- [x] 📝 Example implementations (single and concurrent requests)
- [x] ⏱️ RTF measurement on the sample audio

<br />

# 💻 Technology Stack
- 🎯 **ASR Model**: NVIDIA NeMo Parakeet (default: `nvidia/parakeet-ctc-0.6b-vi`)
- ⚡ **Serving**: FastAPI + Ray Serve (`ray[serve]==2.50.0`)
- 📦 **Dependencies**: uv (`pyproject.toml` + `uv.lock`)
- 🐳 **Containerization**: Docker & Docker Compose
- 🔌 **API Compatibility**: OpenAI API format
- 🖥️ **GPU Support**: NVIDIA CUDA 12.8
- 🛠️ **Client Libraries**: OpenAI Python SDK

<br />

# 🙏 Acknowledgments
- 🚀 [Ray Team](https://github.com/ray-project/ray) for Ray Serve
- 🤖 [NVIDIA NeMo Team](https://github.com/NVIDIA/NeMo) for the Parakeet ASR models
