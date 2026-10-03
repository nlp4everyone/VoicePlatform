# 🎤 VoicePlatform - Private Audio Transcription Service

A high-performance, privacy-focused audio transcription service powered by Qwen's Automatic Speech Recognition (ASR) model. This project provides a secure, local deployment solution for transcribing audio files without relying on cloud services, ensuring complete data privacy and control.

Built with vLLM for optimal performance and GPU acceleration, this service offers OpenAI-compatible API endpoints for seamless integration with existing applications and workflows.

<br />

# 🧠 Core Features

### 🎯 Advanced ASR Capabilities
- High-accuracy audio transcription using state-of-the-art ASR models
- Support for multiple languages and audio formats
- OpenAI-compatible API for easy integration with existing tools
- Flexible input formats (file upload and base64 encoding)

### 🚀 Performance Optimizations
- GPU-accelerated inference with vLLM for real-time transcription
- Configurable GPU memory utilization and model parameters
- Concurrent request handling through vLLM batching (up to `MAX_NUM_SEQS` sequences)
- Optimized for low-latency transcription workflows

### 🔒 Privacy & Security
- Complete local deployment - no data leaves your infrastructure
- No API keys or external service dependencies required
- Full control over model and data processing pipeline
- Ideal for sensitive audio content and compliance requirements

### 🔌 Developer-Friendly API
- RESTful API with OpenAI-compatible endpoints
- Comprehensive examples for different integration scenarios
- Docker-based deployment for consistent environments
- Container logs (`make logs`) and a `/health` endpoint for monitoring

<br />

# 📡 Capabilities

| Capability | Protocol | Endpoint | Status |
|------------|----------|----------|--------|
| Transcription | HTTP | `POST /v1/audio/transcriptions` | ✅ |
| Transcription (base64 audio) | HTTP | `POST /v1/chat/completions` | ✅ |
| Transcription (streaming text) | HTTP (SSE) | `POST /v1/audio/transcriptions` with `stream=true` | ✅ |
| Realtime (live audio) | WebSocket | `ws://<host>:8001/v1/realtime` | ✅ |

**Legend**: ✅ supported · 🚧 in progress · ❌ not supported

> ℹ️ Realtime relies on the `--hf-overrides` flag in `docker-compose.yml`, which switches Qwen3-ASR to its realtime architecture so vLLM mounts the `/v1/realtime` route.

<br />

# 🛠️ Prerequisites

1. **💻 Hardware Requirements**
   - 🖥️ **CPU**: x86_64 (AVX2 support recommended)
   - 🧠 **RAM**: 8GB minimum, 16GB+ recommended
   - 🎮 **GPU**: NVIDIA GPU with CUDA support (recommended for optimal performance)
   - 💾 **Storage**: SSD recommended for better model loading performance

2. **🛠️ Software Dependencies**
   - 🐳 **Docker and Docker Compose**
   - 🎮 **NVIDIA Container Toolkit** (for GPU support)
   - ⚡ **NVIDIA driver** compatible with the CUDA version of the `vllm/vllm-openai:v0.24.0` image (a host CUDA Toolkit is not required, CUDA ships inside the image)

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
   git checkout vllm/qwen3_asr
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
   # Build the image (audio deps are installed once) and start in the background
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

   Other useful targets: `make down`, `make restart`, `make ps`, `make clean` (also removes the model cache). Run `make help` for the full list.

6. **Access the service**
   - 🔌 **API Endpoint**: http://localhost:8001/v1 - OpenAI-compatible API
   - 📚 **API Documentation**: http://localhost:8001/docs - Interactive API docs
   - 📊 **Health Check**: http://localhost:8001/health - Service status

<br />

# 📝 Examples

All examples read the model name from `MODEL_NAME` in your `.env` (falling back to
`Qwen/Qwen3-ASR-1.7B`), so they stay in sync with the served model. Install the client
dependencies first: `pip install openai httpx requests python-dotenv`. Run them from the project root,
since the audio path is relative.

## 📄 Audio Transcription (non-streaming)

**Transcription**: HTTP · `POST /v1/audio/transcriptions`

Check out the `examples/audio_transcription_example.py` file to see how to:
- 🎯 **Transcribe** audio files using the OpenAI-compatible API
- ⏱️ **Measure** transcription performance and processing time
- 🌍 **Handle** different audio formats and languages

```bash
# Run the audio transcription example
python examples/audio_transcription_example.py
```

### ⏱️ Real-Time Factor (RTF)

RTF = processing time ÷ audio duration. A value below 1 means faster than real time.

Measured with the OpenAI SDK (`client.audio.transcriptions.create`) on `resources/sample_vi.mp3` (7.81 s of audio): one warm-up request, then 5 timed requests.

| Model | GPU | Audio | Mean time | RTF (mean) | RTF (min – max) |
|-------|-----|-------|-----------|------------|-----------------|
| `Qwen/Qwen3-ASR-0.6B` | NVIDIA RTX 3060 | 7.81 s | 0.184 s | 0.024 | 0.023 – 0.024 |

> ℹ️ Times are end-to-end over localhost (upload + decode + inference) with `MODEL_NAME=Qwen/Qwen3-ASR-0.6B` and the default GPU/sequence limits from `docker-compose.yml`; the first (warm-up) request was slightly slower (0.216 s). RTF depends on the model, GPU and concurrent load, so re-measure on your own hardware.

## 🔐 Base64 Audio Transcription

**Transcription**: HTTP · `POST /v1/chat/completions`

The `examples/base64_audio_example.py` demonstrates:
- 🔐 **Encoding** audio files in base64 format
- 📤 **Sending** audio data via chat completions API
- ⚙️ **Processing** transcription results programmatically

```bash
# Run the base64 audio example
python examples/base64_audio_example.py
```

## 🌊 Streaming Transcription (SSE)

### 🔄 Asynchronous Streaming

**Transcription**: HTTP (SSE) · `POST /v1/audio/transcriptions` with `stream=true`

The `examples/async_streaming_example.py` demonstrates:
- 🔄 **Real-time** transcription using async OpenAI client
- 🌊 **Streaming** responses for immediate feedback
- ⚙️ **Configurable** transcription parameters (temperature, seed, top_p)

```bash
# Run the async streaming example
python examples/async_streaming_example.py
```

### 🔄 Synchronous Streaming

**Transcription**: HTTP (SSE) · `POST /v1/audio/transcriptions` with `stream=true`

The `examples/sync_streaming_example.py` shows:
- 🔄 **Streaming** transcription using raw HTTP requests
- 📡 **Server-Sent Events** (SSE) format handling
- 🔗 **Direct API** integration without OpenAI SDK

```bash
# Run the sync streaming example
python examples/sync_streaming_example.py
```

## ⚡ Realtime Transcription (WebSocket)

**Realtime**: WebSocket · `ws://localhost:8001/v1/realtime`

The `examples/websocket_streaming_example.py` demonstrates:
- 🔌 **Connecting** to vLLM's Realtime API over WebSocket (`/v1/realtime`), distinct from the REST/SSE endpoint used above
- 🎙️ **Converting** audio to PCM16 @ 16kHz and streaming it to the server in chunks
- 📥 **Receiving** `transcription.delta` / `transcription.done` events as they arrive

Requires extra client-side dependencies: `pip install websockets librosa numpy python-dotenv`

```bash
# Run the websocket streaming example
python examples/websocket_streaming_example.py
```

An alternative version, `examples/websocket_openai_sdk_example.py`, uses the OpenAI SDK's
`realtime.connect()` helper for the connection/auth handshake instead of the raw `websockets`
library (still sending/receiving vLLM's own event JSON directly, since vLLM's realtime events
aren't part of OpenAI's official Realtime API schema). Requires `pip install "openai[realtime]" librosa numpy python-dotenv`.

```bash
# Run the OpenAI SDK-based websocket example
python examples/websocket_openai_sdk_example.py
```

## 🎵 Sample Audio File

A sample audio file is included in `resources/sample_vi.mp3` for testing the transcription capabilities immediately after deployment.

<br />

# ⚙️ Configuration

## 🌍 Environment Variables

Customize the service behavior using these environment variables in your `.env` file:

```bash
# 🎯 Model configuration
MODEL_NAME=Qwen/Qwen3-ASR-1.7B          # ASR model to use (configurable)
GPU_MEMORY_UTILIZATION=0.8              # GPU memory allocation (0.0-1.0)
MAX_MODEL_LEN=8192                      # Maximum model context length
MAX_NUM_SEQS=4                          # Maximum concurrent sequences

# 🌐 Network configuration
VLLM_HOST=0.0.0.0                       # Service host address
VLLM_PORT=8001                          # Service port

# 💾 Storage configuration
DOWNLOAD_DIR=/root/.cache/huggingface/hub  # Model cache directory
```

<br />

# 📋 To-Do List
- [x] 🚀 Basic ASR service deployment
- [x] 📝 Example implementations
- [x] 🌊 Streaming transcription capabilities (async and sync)
- [x] 🔌 WebSocket realtime transcription (`/v1/realtime`)
- [x] ⏱️ RTF measurement on the sample audio

<br />

# 💻 Technology Stack:
- 🎯 **ASR Model**: Configurable (default: Qwen/Qwen3-ASR-1.7B)
- ⚡ **Inference Engine**: vLLM
- 🐳 **Containerization**: Docker & Docker Compose
- 🔌 **API Compatibility**: OpenAI API format
- 🖥️ **GPU Support**: NVIDIA CUDA
- 📊 **Monitoring**: Docker logging
- 🛠️ **Client Libraries**: OpenAI Python SDK, httpx, requests, websockets

<br />

# 🙏 Acknowledgments
- 🚀 [vLLM Team](https://github.com/vllm-project/vllm) for the high-performance inference engine
- 🤖 [Qwen Team](https://github.com/QwenLM/Qwen) for the advanced ASR model

