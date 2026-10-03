# 🎤 VoicePlatform - Private Text-to-Speech Service

A high-performance, privacy-focused text-to-speech (TTS) service powered by OpenBMB's VoxCPM2 model. This project provides a secure, local deployment solution for synthesizing speech from text without relying on cloud services, ensuring complete data privacy and control.

Built with vLLM-Omni for optimal performance and GPU acceleration, this service offers OpenAI-compatible API endpoints for seamless integration with existing applications and workflows.

<br />

# 🧠 Core Features

### 🎯 Advanced TTS Capabilities
- Natural, expressive speech synthesis using the state-of-the-art VoxCPM2 model
- Support for 30 languages (including Vietnamese and English); no language tag needed, input text in any supported language directly
- Zero-shot voice cloning from a short reference recording
- OpenAI-compatible `/v1/audio/speech` API for easy integration with existing tools
- FP8 quantization out of the box for lower VRAM usage

### 🚀 Performance Optimizations
- GPU-accelerated inference with vLLM-Omni for real-time synthesis
- Configurable GPU memory utilization, KV cache budget and model parameters
- Concurrent request handling through vLLM batching (up to `MAX_NUM_SEQS` sequences)
- Optimized for low-latency speech generation workflows

### 🔒 Privacy & Security
- Complete local deployment - no data leaves your infrastructure
- No API keys or external service dependencies required
- Full control over model and data processing pipeline
- Ideal for sensitive content and compliance requirements

### 🔌 Developer-Friendly API
- RESTful API with OpenAI-compatible endpoints
- Comprehensive examples for different integration scenarios
- Docker-based deployment for consistent environments
- Container logs (`make logs`), a `/health` endpoint and Prometheus-style `/metrics` for monitoring

<br />

# 📡 Capabilities

| Capability | Protocol | Endpoint | Status |
|------------|----------|----------|--------|
| Speech synthesis | HTTP | `POST /v1/audio/speech` | ✅ |
| Voice cloning (base64 `ref_audio`) | HTTP | `POST /v1/audio/speech` | ✅ |
| Speech synthesis (streaming raw audio) | HTTP (chunked PCM) | `POST /v1/audio/speech` with `stream_format="audio"` | ✅ |
| Speech synthesis (streaming events) | HTTP (SSE) | `POST /v1/audio/speech` with `stream=true` | ✅ |
| Realtime (streaming text input) | WebSocket | `ws://<host>:8002/v1/audio/speech/stream` | ✅ |

**Legend**: ✅ supported · 🚧 in progress · ❌ not supported

> ℹ️ The WebSocket endpoint is provided by vLLM-Omni: the client sends text incrementally and receives audio back. See `examples/websocket_streaming_example.py` and the Realtime Synthesis (WebSocket) section in Examples for the message flow.

<br />

# 🛠️ Prerequisites

1. **💻 Hardware Requirements**
   - 🖥️ **CPU**: x86_64 (AVX2 support recommended)
   - 🧠 **RAM**: 8GB minimum, 16GB+ recommended
   - 🎮 **GPU**: NVIDIA GPU with CUDA support (required: `docker-compose.yml` reserves a GPU; defaults were tested on a 12 GB RTX 3060)
   - 💾 **Storage**: SSD recommended for better model loading performance

2. **🛠️ Software Dependencies**
   - 🐳 **Docker and Docker Compose**
   - 🎮 **NVIDIA Container Toolkit** (for GPU support)
   - ⚡ **NVIDIA driver** compatible with the CUDA version of the `vllm/vllm-omni:v0.26.0` image (a host CUDA Toolkit is not required, CUDA ships inside the image)

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
   git checkout vllm_omni/voxcpm
   ```

3. **Set up environment configuration**
   ```bash
   # Create .env from .env.sample (skipped if .env already exists)
   make env
   # Edit the .env file to customize settings
   # nano .env  # or use your preferred text editor
   ```

4. **Build and start the service**
   ```bash
   # Build the image (VoxCPM deps are installed once) and start in the background
   make up
   # Or run in the foreground
   make start
   ```
   Docker is invoked with `sudo` by default. Override with `make up DOCKER=docker` if your user is in the `docker` group.

5. **Verify the service is running**
   ```bash
   # Check the health endpoint
   make health
   # View service logs
   make logs
   ```

   Other useful targets: `make down`, `make restart`, `make ps`, `make clean` (also removes the built image; the model cache in `~/.cache/huggingface/hub` is kept). Run `make help` for the full list.

6. **Access the service**
   - 🔌 **API Endpoint**: http://localhost:8002/v1 - OpenAI-compatible API
   - 📚 **API Documentation**: http://localhost:8002/docs - Interactive API docs
   - 📊 **Health Check**: http://localhost:8002/health - Service status

<br />

# 📝 Examples

All scripts talk to the server at `http://localhost:8002/v1` and write their `.wav` output into the `results/` directory. Install the client deps first: `pip install openai httpx requests websockets`.

## 🔊 Speech Synthesis (non-streaming)

**Synthesis**: HTTP · `POST /v1/audio/speech`

The `examples/speech_synthesis_example.py` shows how to:
- 🎯 **Generate** a complete audio clip from text via `/v1/audio/speech`
- 🔀 **Compare** the OpenAI SDK and raw `httpx` requests, run concurrently
- ⏱️ **Measure** end-to-end synthesis time

```bash
# Run the non-streaming synthesis example
python examples/speech_synthesis_example.py
```

### ⏱️ Real-Time Factor (RTF)

For synthesis, RTF = generation time ÷ duration of the generated audio. A value below 1 means audio is produced faster than it plays back.

Measured with the OpenAI SDK (`client.audio.speech.create`, `response_format="wav"`) on the example text (`"Xin chào, đây là VoicePlatform. Chúc bạn một ngày tốt lành."`): one warm-up request, then 5 timed requests.

| Model | GPU | Mean time | Mean audio | RTF (mean) | RTF (min – max) |
|-------|-----|-----------|------------|------------|-----------------|
| `openbmb/VoxCPM2` (FP8) | NVIDIA RTX 3060 | 1.036 s | 4.10 s | 0.253 | 0.250 – 0.256 |

> ℹ️ Output length differs from run to run (3.4 – 5.3 s for the same text), so compare RTF rather than raw time. Times are end-to-end over localhost with the defaults from `.env.sample` (`QUANTIZATION=fp8`, `MAX_NUM_SEQS=4`); results depend on GPU and concurrent load, so re-measure on your own hardware. Streaming RTF and time-to-first-audio are listed under Streaming Synthesis.

## 🎙️ Voice Cloning (non-streaming)

**Synthesis**: HTTP · `POST /v1/audio/speech` with `ref_audio`

The `examples/voice_cloning_example.py` demonstrates:
- 🔐 **Encoding** a reference recording as a base64 `ref_audio` data URL
- 🧬 **Cloning** its voice zero-shot (optionally with `ref_text` for a closer match)
- 📤 **Saving** the generated clip to disk

```bash
# Run the voice cloning example (uses resources/sample_vi.wav as the reference)
python examples/voice_cloning_example.py
```

## 🌊 Streaming Synthesis

### 🔄 Asynchronous Streaming (raw audio)

**Synthesis**: HTTP (chunked PCM) · `POST /v1/audio/speech` with `stream_format="audio"`

The `examples/async_streaming_example.py` demonstrates:
- 🔄 **Streaming** raw PCM bytes with the async OpenAI client (`stream_format="audio"`)
- ⏱️ **Time-to-first-audio** measurement
- 💾 **Writing** chunks straight into a WAV file as they arrive

```bash
# Run the async streaming example
python examples/async_streaming_example.py
```

### 🔄 Synchronous Streaming (SSE)

**Synthesis**: HTTP (SSE) · `POST /v1/audio/speech` with `stream=true`

The `examples/sync_streaming_example.py` shows:
- 📡 **Server-Sent Events** handling (`speech.audio.delta` / `speech.audio.done`)
- 🔐 **Decoding** base64 PCM deltas into a WAV file
- 🔗 **Direct API** integration with `requests`, no OpenAI SDK

```bash
# Run the sync streaming example
python examples/sync_streaming_example.py
```

> ℹ️ Streaming only supports `response_format` of `pcm` or `wav`; VoxCPM2 outputs 16-bit mono at 48 kHz.

### ⏱️ Streaming RTF and Time-to-First-Audio

Measured with the OpenAI SDK raw-audio streaming (`response_format="pcm"`, `stream_format="audio"`) on `"Xin chào, đây là VoicePlatform. Âm thanh được phát trực tiếp trong lúc mô hình đang tạo."`: one warm-up request, then 5 timed requests. Time-to-first-audio (TTFA) is the delay until the first audio chunk arrives.

| Mode | Mean total time | Mean audio | TTFA (mean) | RTF (mean) | RTF (min – max) |
|------|-----------------|------------|-------------|------------|-----------------|
| HTTP streaming (PCM) | 1.358 s | 5.41 s | 0.167 s | 0.251 | 0.249 – 0.254 |

> ℹ️ RTF is about the same as non-streaming, since streaming changes when audio arrives, not how fast it is generated. What improves is TTFA: playback can start after about 0.17 s instead of waiting for the whole clip (1.4 s on average for this text).

## ⚡ Realtime Synthesis (WebSocket)

**Realtime**: WebSocket · `ws://localhost:8002/v1/audio/speech/stream`

The `examples/websocket_streaming_example.py` demonstrates:
- 🔌 **Connecting** to vLLM-Omni's streaming text-input endpoint, distinct from the REST endpoint used above
- ✍️ **Sending** text chunk by chunk (`input.text`) and flushing it with `input.done`
- 🔁 **Reusing** one connection for two utterances
- ⏱️ **Measuring** time-to-first-audio and RTF per utterance, and writing each result to a WAV file

Requires an extra client-side dependency: `pip install websockets`

```bash
# Run the websocket streaming example
python examples/websocket_streaming_example.py
```

vLLM-Omni accepts text incrementally over WebSocket and sends audio back. Message flow:

| Direction | Message |
|-----------|---------|
| Client → Server | `{"type": "session.config", "model": "openbmb/VoxCPM2", "voice": "default", "response_format": "pcm", "stream_audio": true}` (first message) |
| Client → Server | `{"type": "input.text", "text": "..."}` (one or more) |
| Client → Server | `{"type": "input.done"}` (flush: synthesize the buffered text, connection stays open) |
| Server → Client | `{"type": "audio.start", ...}`, binary audio frames, `{"type": "audio.done", ...}`, `{"type": "session.done", ...}` |
| Client → Server | `{"type": "session.close"}` (end of connection) |

`stream_audio: true` requires `response_format: "pcm"` and sends audio progressively; without it the audio of an utterance arrives once it is fully generated. The text is buffered until `input.done`, so the audio is generated once per flush.

Measured with the same flow (`stream_audio: true`, same text as above, 1 warm-up + 5 timed runs):

| Mode | Mean total time | Mean audio | TTFA (mean) | RTF (mean) | RTF (min – max) |
|------|-----------------|------------|-------------|------------|-----------------|
| WebSocket (`stream_audio: true`) | 1.409 s | 5.63 s | 0.169 s | 0.250 | 0.248 – 0.252 |

> ℹ️ TTFA and RTF match HTTP streaming, so WebSocket is worth using for its protocol (incremental text input, reusing one connection for several utterances), not for extra speed.

## 🎵 Sample Audio Files

Sample audio files are included in `resources/` (`sample_vi.mp3`, `sample_vi.wav`). Only `sample_vi.wav` is used by the examples, as the reference voice in the cloning example; `sample_vi.mp3` is an extra copy in another format.

<br />

# ⚙️ Configuration

## 🌍 Environment Variables

Customize the service behavior using these environment variables in your `.env` file:

```bash
# 🎯 Model configuration
MODEL_NAME=openbmb/VoxCPM2              # TTS model to use (configurable)
GPU_MEMORY_UTILIZATION=0.75             # GPU memory allocation (0.0-1.0)
MAX_MODEL_LEN=2048                      # Maximum model context length
MAX_NUM_SEQS=4                          # Maximum concurrent sequences
QUANTIZATION=fp8                        # Weight quantization (fp8 by default)
KV_CACHE_BYTES=2147483648               # KV cache budget for stage 0 in bytes (2 GiB)

# 🌐 Network configuration
VLLM_HOST=0.0.0.0                       # Service host address
VLLM_PORT=8002                          # Host port (container always listens on 8002)

# 💾 Storage configuration
HF_CACHE_DIR=~/.cache/huggingface/hub      # Host directory mounted as the model cache
```

> ✅ These defaults were tested on an **NVIDIA RTX 3060**. On GPUs with more VRAM you can raise `GPU_MEMORY_UTILIZATION`, `KV_CACHE_BYTES` and `MAX_NUM_SEQS`, or drop `QUANTIZATION=fp8`.

<br />

# 📋 To-Do List
- [x] 🚀 Basic TTS service deployment
- [x] 📝 Example requests
- [x] 🎙️ Voice cloning examples
- [x] 🌊 Streaming synthesis (raw audio and SSE)
- [x] ⏱️ RTF and time-to-first-audio measurements
- [x] ⚡ WebSocket example script (`/v1/audio/speech/stream`)

<br />

# 💻 Technology Stack:
- 🎯 **TTS Model**: Configurable (default: openbmb/VoxCPM2)
- ⚡ **Inference Engine**: vLLM-Omni (`vllm/vllm-omni:v0.26.0`)
- 🐳 **Containerization**: Docker & Docker Compose
- 🔌 **API Compatibility**: OpenAI API format
- 🖥️ **GPU Support**: NVIDIA CUDA
- 📊 **Monitoring**: Docker logging, `/health` and `/metrics`
- 🛠️ **Client Libraries**: OpenAI Python SDK, httpx, requests, websockets

<br />

# 🙏 Acknowledgments
- 🚀 [vLLM Team](https://github.com/vllm-project/vllm-omni) for the high-performance omni-modal inference engine
- 🤖 [OpenBMB Team](https://github.com/OpenBMB/VoxCPM) for the VoxCPM2 TTS model

