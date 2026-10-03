# 🎤 VoicePlatform - Private Speech-to-Text Service (Fun-ASR + WhisperLiveKit)

A high-performance, privacy-focused automatic speech recognition (ASR) service powered by FunAudioLLM's **Fun-ASR-MLT-Nano-2512** model. This project provides a secure, local deployment for transcribing audio without relying on cloud services, keeping all data on your own infrastructure.

The model is served with **vLLM** behind OpenAI-compatible endpoints, and [**WhisperLiveKit**](https://github.com/QuentinFuxa/WhisperLiveKit) is the real-time layer on top of it: it adds WebSocket streaming of microphone audio, a browser UI and live, incremental transcripts.

<br />

# 🧠 Core Features

### 🎯 Advanced ASR Capabilities
- Multilingual transcription (including Vietnamese) with Fun-ASR-MLT-Nano-2512; the spoken language is detected automatically
- OpenAI-compatible `/v1/audio/transcriptions` API, with optional streaming of the text (`stream=true`, Server-Sent Events)
- Any format ffmpeg can decode (wav, mp3, ...) is accepted, no client-side conversion needed
- `float32` inference, the precision FunAudioLLM validated for transcription fidelity

### 🔒 Privacy & Security
- Complete local deployment - no audio leaves your infrastructure
- No API keys or external service dependencies required
- The model is downloaded at setup time and never committed to the repository

### 🔌 Developer-Friendly
- One `make` command to download, convert and serve the model
- Examples for the OpenAI SDK, raw `httpx`/`requests`, streaming and base64 audio
- Docker-based deployment for consistent environments

<br />

# 📡 Capabilities

| Capability | Protocol | Endpoint | Status |
|------------|----------|----------|--------|
| Transcription | HTTP | `POST /v1/audio/transcriptions` | ✅ |
| Transcription (streaming text) | HTTP (SSE) | `POST /v1/audio/transcriptions` with `stream=true` | ✅ |
| Realtime (live audio) | WebSocket | `ws://<host>:8000/asr` | 🚧 |

**Legend**: ✅ supported · 🚧 in progress · ❌ not supported

> ℹ️ Realtime is 🚧 because vLLM has no WebSocket endpoint for Fun-ASR; it depends on connecting WhisperLiveKit to the vLLM server (see [Real-time Transcription](#-real-time-transcription-with-whisperlivekit)).

<br />

# 🏗️ Architecture

```
 Browser / mic ──WebSocket──▶ WhisperLiveKit ──HTTP──▶ vLLM (Fun-ASR-MLT-Nano) ◀──HTTP── Batch clients
                  /asr         :8000                   /v1/audio/transcriptions          (examples/)
                                                       :8002
```

- **vLLM server** (`fun-asr-mlt` service, port `8002`): runs the model and exposes the OpenAI-compatible REST API. vLLM does not offer a realtime WebSocket endpoint for Fun-ASR (only models implementing `SupportsRealtime`, e.g. Voxtral Realtime or Qwen3-ASR Realtime, get `/v1/realtime`).
- **WhisperLiveKit** (port `8000`): receives audio chunks over WebSocket, segments them and forwards each segment for transcription, then streams the transcript back to the client.

<br />

# 🛠️ Prerequisites

1. **💻 Hardware Requirements**
   - 🖥️ **CPU**: x86_64 (AVX2 support recommended)
   - 🧠 **RAM**: 8GB minimum, 16GB+ recommended
   - 🎮 **GPU**: NVIDIA GPU with CUDA support
   - 💾 **Storage**: SSD recommended, plus room for the downloaded checkpoint and the converted model

2. **🛠️ Software Dependencies**
   - 🐳 **Docker and Docker Compose**
   - 🎮 **NVIDIA Container Toolkit** (for GPU support)

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
   git checkout whisperlivekit/funasr
   ```

3. **Set up environment configuration**
   ```bash
   # Create .env from .env.example (skipped if .env already exists)
   make env
   # Edit the .env file to customize settings
   # nano .env
   ```

4. **Prepare the model (one-time)**
   ```bash
   make model
   ```
   `FunAudioLLM/Fun-ASR-MLT-Nano-2512` only ships a native FunASR checkpoint (`model.pt`) that vLLM cannot load. `make model` runs `scripts/convert_mlt_to_vllm.py` in a one-shot container, which:
   - downloads the MLT checkpoint into `HF_CACHE_DIR`
   - converts its weights to `model.safetensors` and validates every tensor against the official `FunAudioLLM/Fun-ASR-Nano-2512-vllm` layout
   - borrows the config, tokenizer and preprocessor files from that vLLM package
   - writes the result to `models/Fun-ASR-MLT-Nano-2512-vllm/` (git-ignored)

   Later runs are a no-op once the converted weights exist. You can skip this step: `make up` / `make start` run it automatically when needed.

5. **Start the service**
   ```bash
   # Convert the model if needed, then start in the background
   make up
   # Or run in the foreground
   make start
   ```
   Docker is invoked with `sudo` by default. Override with `make up DOCKER=docker` if your user is in the `docker` group.

6. **Verify the service is running**
   ```bash
   make health   # check the health endpoint
   make logs     # follow service logs
   ```

   Other useful targets: `make down`, `make restart`, `make ps`, `make model-clean` (deletes the converted model; re-converting reuses the download cache), `make clean` (also removes the built image; the model cache in `HF_CACHE_DIR` is kept). Run `make help` for the full list.

7. **Access the service**
   - 🔌 **API Endpoint**: http://localhost:8002/v1 - OpenAI-compatible API (served model name: `fun-asr-mlt-nano`)
   - 📚 **API Documentation**: http://localhost:8002/docs - Interactive API docs
   - 📊 **Health Check**: http://localhost:8002/health - Service status

<br />

# 📝 Examples

All scripts talk to the server at `http://localhost:8002/v1` and transcribe `resources/sample_vi.mp3`. Install the client deps first: `pip install openai httpx requests`. If the server was started with an API key, export it as `VLLM_API_KEY`.

## 📄 Audio Transcription (non-streaming)

The `examples/audio_transcription_example.py` shows how to:
- 🔍 **Discover** the served model name via `/v1/models`
- 🔀 **Compare** the OpenAI SDK and raw `httpx` requests, run concurrently
- ⏱️ **Measure** end-to-end processing time

```bash
python examples/audio_transcription_example.py
```

## 🌊 Streaming Transcription (SSE)

The `examples/async_streaming_example.py` demonstrates:
- 📡 **Streaming** the transcript with `stream=true` over Server-Sent Events
- 🧩 **Printing** each `choices[0].delta.content` piece as it arrives

```bash
python examples/async_streaming_example.py
```

> ℹ️ The whole file is uploaded first; only the text streams back. For live microphone audio, use WhisperLiveKit (below).

## 🔐 Base64 Audio

The `examples/base64_audio_example.py` shows how to handle audio that reaches you base64-encoded (e.g. from a browser or another API): decode it back to bytes and upload it as a multipart file. Fun-ASR is a transcription-only model in vLLM, so base64 audio inside a chat-completions JSON body is not supported.

```bash
python examples/base64_audio_example.py
```

## 🎵 Sample Audio Files

Sample Vietnamese audio is included in `resources/` (`sample_vi.mp3`, `sample_vi.wav`).

<br />

# ⚡ Real-time Transcription with WhisperLiveKit

[WhisperLiveKit](https://github.com/QuentinFuxa/WhisperLiveKit) provides the WebSocket front end and web UI.

```bash
pip install whisperlivekit
wlk --host 0.0.0.0 --port 8000 --language auto
```

- 🌐 **Web UI**: http://localhost:8000 - speak into the microphone and watch the transcript update live
- 🔌 **WebSocket**: `ws://localhost:8000/asr` (query parameters such as `?language=vi`)
- 📄 **REST**: `POST http://localhost:8000/v1/audio/transcriptions` (OpenAI-compatible)

> ⚠️ **Integration status**: connecting WhisperLiveKit to the vLLM Fun-ASR server is still in progress on this branch, so WhisperLiveKit is not part of `docker-compose.yml` yet. Known constraints:
> - WhisperLiveKit's built-in `funasr` backend runs **SenseVoiceSmall** in-process (languages: `auto`, `zh`, `yue`, `en`, `ja`, `ko`), not Fun-ASR-MLT-Nano, and has no Vietnamese support.
> - Its `openai-api` backend uses the `whisper-1` model name and requests `verbose_json` with word-level timestamps, which the vLLM Fun-ASR endpoint does not return. So a small adapter is still needed between the two.

<br />

# ⚙️ Configuration

## 🌍 Environment Variables

Customize the service with these variables in your `.env` file:

```bash
# 🌐 Network configuration
VLLM_PORT=8002                          # Host port (container always listens on 8002)

# 💾 Storage configuration
HF_CACHE_DIR=~/.cache/huggingface/hub   # Host directory mounted as the model download cache
```

## 🎛️ Server Parameters

vLLM serving flags are set in `docker-compose.yml` (`fun-asr-mlt` service):

| Flag | Default | Description |
|------|---------|-------------|
| `--served-model-name` | `fun-asr-mlt-nano` | Name clients pass as `model` |
| `--dtype` | `float32` | Inference precision |
| `--gpu-memory-utilization` | `0.7` | Fraction of GPU memory vLLM may use |
| `--max-model-len` | `4096` | Maximum model context length |
| `--enforce-eager` | on | Disables CUDA graphs (lower memory, slower startup) |

<br />

# 📋 To-Do List
- [x] 🚀 Fun-ASR-MLT-Nano served on vLLM
- [x] 🔄 One-time model download and conversion via `make model`
- [x] 📝 Example requests (batch, SSE streaming, base64)

<br />

# 💻 Technology Stack
- 🎯 **ASR Model**: FunAudioLLM/Fun-ASR-MLT-Nano-2512
- ⚡ **Inference Engine**: vLLM (`vllm/vllm-openai:v0.24.0`)
- 🔴 **Real-time Layer**: WhisperLiveKit
- 🐳 **Containerization**: Docker & Docker Compose
- 🔌 **API Compatibility**: OpenAI API format
- 🖥️ **GPU Support**: NVIDIA CUDA
- 🛠️ **Client Libraries**: OpenAI Python SDK, httpx, requests

<br />

# 🙏 Acknowledgments
- 🚀 [vLLM Team](https://github.com/vllm-project/vllm) for the high-performance inference engine
- 🤖 [FunAudioLLM Team](https://github.com/FunAudioLLM/Fun-ASR) for the Fun-ASR models
- ⚡ [Quentin Fuxa](https://github.com/QuentinFuxa/WhisperLiveKit) for WhisperLiveKit
