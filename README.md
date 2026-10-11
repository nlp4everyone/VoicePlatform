# 🎤 VoicePlatform - Private Text-to-Speech Service

A privacy-focused Vietnamese and English text-to-speech (TTS) service powered by the VieNeu-TTS v3 Turbo model. This project provides a secure, local deployment solution for synthesizing speech from text without relying on cloud services, ensuring complete data privacy and control.

Built on the VieNeu-TTS PyTorch engine (CUDA graphs, continuous batching across streams), this service offers an OpenAI-compatible `/v1/audio/speech` API for seamless integration with existing applications and workflows.

<br />

# 🧠 Core Features

### 🎯 Advanced TTS Capabilities
- Natural speech synthesis in Vietnamese and English with the VieNeu-TTS v3 Turbo model (48 kHz output)
- Preset voices out of the box (list them with `GET /v1/voices`)
- Voice cloning from a short reference clip (3-8 s) enrolled with `POST /v1/voices`
- OpenAI-compatible `/v1/audio/speech` API for easy integration with existing tools (OpenAI SDK, Pipecat, LiveKit)

### 🚀 Performance Optimizations
- GPU-accelerated inference with one CUDA graph per audio frame (no `torch.compile`, no compiler toolchain)
- Continuous batching: up to `VIENEU_MAX_STREAMS` streams are generated at once on one GPU
- Audio is streamed while it is generated (raw chunked PCM/WAV or Server-Sent Events)
- Reproducible image built from a pinned VieNeu-TTS commit and its `uv.lock`, so the dependency set matches the upstream benchmarks
- The CUDA graphs are captured by a warm-up at startup, so the first client does not pay for them

### 🔒 Privacy & Security
- Complete local deployment - no data leaves your infrastructure
- No external service dependencies; the model is downloaded once from Hugging Face and cached
- Optional Bearer authentication with `VIENEU_API_KEY`
- Ideal for sensitive content and compliance requirements

### 🔌 Developer-Friendly API
- RESTful API with OpenAI-compatible endpoints
- Comprehensive examples for different integration scenarios
- Docker-based deployment for consistent environments
- Container logs (`make logs`) and a `/health` endpoint that reports active streams and returns `503` if the GPU scheduler has died
- Overload protection: a bounded queue and HTTP `429` with `Retry-After` when the server is full

<br />

# 📡 Capabilities

| Capability | Protocol | Endpoint | Status |
|------------|----------|----------|--------|
| Speech synthesis | HTTP | `POST /v1/audio/speech` | ✅ |
| Voice cloning (enroll a clip, then use it as `voice`) | HTTP | `POST /v1/voices` + `POST /v1/audio/speech` | ✅ |
| Speech synthesis (streaming raw audio) | HTTP (chunked PCM/WAV) | `POST /v1/audio/speech` (default `stream_format="audio"`) | ✅ |
| Speech synthesis (streaming events) | HTTP (SSE) | `POST /v1/audio/speech` with `stream_format="sse"` | ✅ |
| Realtime (streaming text input) | WebSocket | - | ❌ |

**Legend**: ✅ supported · 🚧 in progress · ❌ not supported

> ℹ️ VieNeu-TTS has no WebSocket endpoint. For text that arrives incrementally (for example from an LLM), send each finished sentence as its own streaming request. Only `pcm` and `wav` are supported as `response_format`; `mp3`, `opus`, `aac` and `flac` return HTTP 400.

<br />

# 🛠️ Prerequisites

1. **💻 Hardware Requirements**
   - 🖥️ **CPU**: x86_64
   - 🧠 **RAM**: 8GB minimum, 16GB+ recommended
   - 🎮 **GPU**: NVIDIA GPU with CUDA support (required: `docker-compose.yml` reserves a GPU; upstream measures on a 12 GB RTX 3060 and reports about 1.1 GB peak VRAM at 16 streams)
   - 💾 **Storage**: SSD recommended for better model loading performance

2. **🛠️ Software Dependencies**
   - 🐳 **Docker and Docker Compose**
   - 🎮 **NVIDIA Container Toolkit** (for GPU support)
   - ⚡ **NVIDIA driver** compatible with CUDA 12.8 (the `nvidia/cuda:12.8.0-runtime` base image; a host CUDA Toolkit is not required, CUDA ships inside the image)

<br />

# 🚀 Quick Start

1. **Clone the repository**
   ```bash
   # Clone the repository
   git clone https://github.com/nlp4everyone/VoicePlatform.git
   # Navigate to project directory
   cd VoicePlatform
   ```

2. **Switch to the correct branch**
   ```bash
   # Fetch the latest changes and checkout the correct branch
   git fetch
   git checkout vieneu/vieneu-tts
   ```

3. **Set up environment configuration**
   ```bash
   # Create .env from .env.sample (skipped if it already exists)
   make env
   # Edit the .env file to customize settings
   # nano .env  # or use your preferred text editor
   ```

4. **Build and start the service**
   ```bash
   # Build the image (locked dependencies are installed once) and start in the background
   make up
   # Or run in the foreground
   make start
   ```
   Docker is invoked with `sudo` by default. Override with `make up DOCKER=docker` if your user is in the `docker` group.

   The first start downloads the model from Hugging Face and captures the CUDA graphs, so it can take a few minutes; later starts reuse the model cache stored in the Docker volume `hf-cache`.

5. **Verify the service is running**
   ```bash
   # Check the health endpoint
   make health
   # View service logs
   make logs
   ```

   Other useful targets: `make down`, `make restart`, `make ps`, `make clean` (also removes the built image; the model cache volume is kept) and `make clean-cache` (deletes the cached model, so it is downloaded again on the next start). Run `make help` for the full list.

6. **Access the service**
   - 🔌 **API Endpoint**: http://localhost:8000/v1 - OpenAI-compatible API
   - 📚 **API Documentation**: http://localhost:8000/docs - Interactive API docs
   - 📊 **Health Check**: http://localhost:8000/health - Service status

<br />

# 📝 Examples

All scripts talk to the server at `http://localhost:8000/v1` and write their `.wav` output into the `results/` directory. Install the client deps first: `pip install openai httpx requests`. If you set `VIENEU_API_KEY`, pass it as the `api_key` argument of the functions in the scripts.

The model name is reported back as-is by the server (`vieneu-v3-turbo`) and the examples use the preset voice `Mai Anh`. List the voices available on your server with:

```bash
curl http://localhost:8000/v1/voices
```

## 🔊 Speech Synthesis (non-streaming)

**Synthesis**: HTTP · `POST /v1/audio/speech`

The `examples/speech_synthesis_example.py` shows how to:
- 🎯 **Generate** a complete audio clip from text via `/v1/audio/speech`
- 🔀 **Compare** the OpenAI SDK and raw `httpx` requests, run concurrently
- ⏱️ **Measure** end-to-end synthesis time and RTF

```bash
# Run the non-streaming synthesis example
python examples/speech_synthesis_example.py
```

> ℹ️ The server always streams. A `wav` response carries a header with an unknown length so players can start at once, so the examples request `pcm` and write the WAV header themselves, which gives a regular seekable file.

### ⏱️ Real-Time Factor (RTF)

For synthesis, RTF = generation time ÷ duration of the generated audio. A value below 1 means audio is produced faster than it plays back.

Measured with the OpenAI SDK (`client.audio.speech.create`, `response_format="pcm"`, voice `Mai Anh`) on the example text (`"Xin chào, đây là VoicePlatform. Chúc bạn một ngày tốt lành."`): one warm-up request, then 5 timed requests, one at a time.

| Model | GPU | Mean time | Mean audio | RTF (mean) | RTF (min – max) |
|-------|-----|-----------|------------|------------|-----------------|
| `pnnbao-ump/VieNeu-TTS-v3-Turbo` | NVIDIA RTX 3060 | 1.563 s | 3.74 s | 0.417 | 0.405 – 0.426 |

> ℹ️ Output length differs from run to run (3.7 – 3.9 s for the same text), so compare RTF rather than raw time. Times are end-to-end over localhost with the defaults from `.env.sample` (`VIENEU_MAX_STREAMS=16`); results depend on GPU and concurrent load, so re-measure on your own hardware. The server also logs `ttfa`, `total`, `audio` and `rtf` for each request (`make logs`). Streaming RTF and time-to-first-audio are listed under Streaming Synthesis.

## 🎙️ Voice Cloning (non-streaming)

**Synthesis**: HTTP · `POST /v1/voices` then `POST /v1/audio/speech`

The `examples/voice_cloning_example.py` demonstrates:
- 📤 **Enrolling** a reference recording under a name with a multipart `POST /v1/voices` (3-8 s clip, up to 20 MB, optional denoising)
- 🧬 **Cloning** its voice by passing that name as `voice`
- 💾 **Saving** the generated clip to disk

```bash
# Run the voice cloning example (uses resources/sample_vi.wav as the reference)
python examples/voice_cloning_example.py
```

> ⚠️ Enrolled voices are kept in the server's memory only, so they are lost when the container restarts. Names of built-in voices are rejected with HTTP 409.

## 🌊 Streaming Synthesis

### 🔄 Asynchronous Streaming (raw audio)

**Synthesis**: HTTP (chunked PCM) · `POST /v1/audio/speech`

The `examples/async_streaming_example.py` demonstrates:
- ⚡ **Streaming** raw PCM through the async OpenAI client as it is generated
- 💾 **Writing** each chunk to a WAV file on arrival
- ⏱️ **Measuring** time-to-first-audio (TTFA) and RTF

```bash
# Run the async streaming example
python examples/async_streaming_example.py
```

### 🔄 Synchronous Streaming (SSE)

**Synthesis**: HTTP (SSE) · `POST /v1/audio/speech` with `stream_format="sse"`

The `examples/sync_streaming_example.py` demonstrates:
- 📡 **Consuming** OpenAI-style `speech.audio.delta` / `speech.audio.done` events with plain `requests`
- 🔓 **Decoding** the base64 PCM chunks and appending them to a WAV file
- ⏱️ **Measuring** time-to-first-audio (TTFA) and RTF

```bash
# Run the sync streaming example
python examples/sync_streaming_example.py
```

### ⏱️ Streaming RTF and Time-to-First-Audio

Measured with the OpenAI SDK raw-audio streaming (`response_format="pcm"`, default `stream_format="audio"`) and with raw `httpx` SSE (`stream_format="sse"`) on `"Xin chào, đây là VoicePlatform. Âm thanh được phát trực tiếp trong lúc mô hình đang tạo."`: one warm-up request, then 5 timed requests per mode, one at a time. Time-to-first-audio (TTFA) is the delay until the first audio chunk arrives.

| Mode | Mean total time | Mean audio | TTFA (mean) | RTF (mean) | RTF (min – max) |
|------|-----------------|------------|-------------|------------|-----------------|
| HTTP streaming (PCM) | 2.208 s | 5.18 s | 0.084 s | 0.426 | 0.402 – 0.446 |
| HTTP streaming (SSE) | 2.192 s | 5.10 s | 0.095 s | 0.429 | 0.412 – 0.456 |

> ℹ️ RTF is about the same as non-streaming (0.417 above), since streaming changes when audio arrives, not how fast it is generated. What improves is TTFA: playback can start after about 0.08 s instead of waiting for the whole clip (2.2 s on average for this text). SSE adds a little overhead from base64 and JSON framing.

## 🎵 Sample Audio Files

Sample audio files are included in `resources/` (`sample_vi.mp3`, `sample_vi.wav`). Only `sample_vi.wav` is used by the examples, as the reference voice in the cloning example; `sample_vi.mp3` is an extra copy in another format.

<br />

# ⚙️ Configuration

## 🌍 Environment Variables

Customize the service behavior using these environment variables in your `.env` file:

```bash
# 🐳 Image configuration
VIENEU_REF=85344322b7258b4e25479b692e8e3396baf9db34   # upstream VieNeu-TTS commit baked into the image

# 🎯 Serving configuration
VIENEU_MAX_STREAMS=16                   # Concurrent streams batched on the GPU
VIENEU_QUEUE=16                         # Extra requests allowed to wait for a free stream
VIENEU_QUEUE_TIMEOUT=10                 # Seconds a queued request waits before HTTP 429
VIENEU_API_KEY=                         # Bearer key for the API (empty = open API)
VIENEU_WATERMARK=1                      # Audio watermark switch (0 to disable)

# 🌐 Network configuration
API_PORT=8000                           # Host port; the container always listens on 8000
```

> ✅ The server binds to `0.0.0.0` inside the container, so set `VIENEU_API_KEY` before exposing the port beyond your machine. The key is checked on every `/v1/*` request but not on `/health`.

## ⚡ Performance Tuning

The upstream engine is already optimized: one CUDA graph per audio frame (`VIENEU_FUSED_FRAME=1` by default), continuous batching on a single GPU worker, and bfloat16 picked automatically on GPUs that support it. `torch.compile` is not used on purpose (it needs a C++ toolchain). What is left to tune, roughly in order of impact:

1. 🎯 **Match `VIENEU_MAX_STREAMS` to the real load.** The codec, not the backbone, is the cost: each codec call takes about 65 ms plus about 2.5 ms per *reserved* slot, even when the slot is idle. Upstream numbers on an RTX 3060:

   | `VIENEU_MAX_STREAMS` | Concurrent streams | First audio | RTF |
   |----------------------|--------------------|-------------|-----|
   | 16 | 1 | ~115 ms | |
   | 16 | 2–8 | 130–165 ms | |
   | 16 | 16 starting at once | ~185 ms (median) | 0.59 (worst) |
   | 32 | 1 | | 0.70 |
   | 32 | 32 at once | ~450 ms | 0.93 |

   If you rarely serve more than a few requests at once, `VIENEU_MAX_STREAMS=4` or `8` lowers both RTF and TTFA, at the cost of peak concurrency. Keep `VIENEU_QUEUE` in line with it.
2. 💧 **`VIENEU_WATERMARK=0`** skips the watermark applied to every audio chunk. Only disable it if you do not need AI-generated audio to be marked.
3. 📶 **Lower `sample_rate`** (`24000`, `16000`) in the request when the client allows it. Resampling runs per chunk with no extra latency, and the payload is 2–3× smaller. It does not make generation faster.
4. ✂️ **Keep inputs reasonably short.** Text is chunked at 256 characters (`max_chars`); splitting long text sensibly gives steadier time-to-first-audio.
5. 🎙️ **Enroll cloned voices once** with `POST /v1/voices` (about 2.7 s) and then request them by name, instead of sending reference audio on every request.
6. 🖥️ **Give the service the GPU to itself.** Other GPU workloads make RTF fluctuate; replace `count: all` with `device_ids` in `docker-compose.yml` to pin a specific GPU. Avoid frequent restarts, since each start re-captures the CUDA graphs.

> ⚠️ The codec's own `use_cuda_graph` mode is about 3× faster but produces wrong audio, so upstream does not use it. Do not try to enable it.

> ℹ️ Measure before changing defaults: run the examples with `VIENEU_MAX_STREAMS` set to `4`, `8` and `16`, then compare the `ttfa` and `rtf` values in `make logs`.

## 🔄 Upgrading VieNeu-TTS

The image installs the exact dependency set locked by one upstream commit. To upgrade, set `VIENEU_REF` in `.env` to another commit of [pnnbao97/VieNeu-TTS](https://github.com/pnnbao97/VieNeu-TTS) and run `make up`. The CUDA runtime version in the `Dockerfile` (`nvidia/cuda:12.8.0-runtime`) must match the torch build in that commit's `uv.lock`.

<br />

# 📋 To-Do List
- [x] 🚀 Basic TTS service deployment
- [x] 📝 Example requests
- [x] 🎙️ Voice cloning examples
- [x] 🌊 Streaming synthesis (raw audio and SSE)
- [x] ⏱️ RTF and time-to-first-audio measurements
- [ ] ⚡ WebSocket realtime endpoint (not provided by VieNeu-TTS)

<br />

# 💻 Technology Stack:
- 🎯 **TTS Model**: VieNeu-TTS v3 Turbo (`pnnbao-ump/VieNeu-TTS-v3-Turbo`)
- ⚡ **Inference Engine**: VieNeu-TTS PyTorch backend (CUDA graphs, `torch==2.8.0`)
- 🌐 **Server**: FastAPI + Uvicorn (`apps/openai_speech.py` from VieNeu-TTS)
- 🐳 **Containerization**: Docker & Docker Compose (multi-stage build on `nvidia/cuda:12.8.0-runtime`)
- 🔌 **API Compatibility**: OpenAI API format
- 🖥️ **GPU Support**: NVIDIA CUDA
- 📊 **Monitoring**: Docker logging and `/health`
- 🛠️ **Client Libraries**: OpenAI Python SDK, httpx, requests

<br />

# 🙏 Acknowledgments
- 🤖 [Phạm Nguyễn Ngọc Bảo](https://github.com/pnnbao97/VieNeu-TTS) for VieNeu-TTS, the model and the serving code this branch packages (Apache-2.0)
