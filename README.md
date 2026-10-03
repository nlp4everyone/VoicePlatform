# 🎤 VoicePlatform - Private Text-to-Speech Service

A high-performance, privacy-focused text-to-speech (TTS) service powered by [KhanhTTS-OmniVoice](https://huggingface.co/kjanh/KhanhTTS-OmniVoice), a Vietnamese + English model fine-tuned from k2-fsa/OmniVoice. This project provides a secure, local deployment solution for synthesizing speech from text without relying on cloud services, ensuring complete data privacy and control.

Built with vLLM-Omni for optimal performance and GPU acceleration, this service offers OpenAI-compatible API endpoints for seamless integration with existing applications and workflows.

<br />

# 🧠 Core Features

### 🎯 Advanced TTS Capabilities
- Natural Vietnamese and English speech synthesis with KhanhTTS-OmniVoice
- Zero-shot voice cloning from a short reference recording (`ref_audio` + `ref_text`)
- OpenAI-compatible `/v1/audio/speech` API for easy integration with existing tools
- Lightweight ~0.6B diffusion model that fits comfortably on consumer GPUs

### 🚀 Performance Optimizations
- GPU-accelerated inference with vLLM-Omni
- Configurable GPU memory utilization
- Efficient handling of concurrent requests
- Client-side sentence-level streaming for an earlier first audio (at the cost of a longer total time, see RTF below)

### 🔒 Privacy & Security
- Complete local deployment - no data leaves your infrastructure
- No API keys or external service dependencies required
- Full control over model and data processing pipeline
- Ideal for sensitive content and compliance requirements

### 🔌 Developer-Friendly API
- RESTful API with OpenAI-compatible endpoints
- Comprehensive examples for different integration scenarios
- Docker-based deployment for consistent environments
- Container logs with rotation (`make logs`), a `/health` endpoint and Prometheus-style `/metrics` for monitoring

<br />

# 📡 Capabilities

| Capability | Protocol | Endpoint | Status |
|------------|----------|----------|--------|
| Speech synthesis | HTTP | `POST /v1/audio/speech` | ✅ |
| Voice cloning (base64 `ref_audio`) | HTTP | `POST /v1/audio/speech` | ✅ |
| Sentence-level streaming (client-side) | HTTP | one `POST /v1/audio/speech` per sentence | ✅ |
| Streaming audio from the server | HTTP (chunked PCM / SSE) | `stream` / `stream_format` options | ❌ |
| Realtime (streaming text input) | WebSocket | `ws://<host>:8002/v1/audio/speech/stream` | ❌ |

**Legend**: ✅ supported · 🚧 in progress · ❌ not supported

> ℹ️ OmniVoice returns every request as one complete WAV clip: `stream` and `stream_format` are ignored, and the WebSocket route answers `{"type":"error","message":"Streaming speech is not available"}`. Sentence-level streaming is therefore done on the client side (see the example below).

<br />

# 🛠️ Prerequisites

1. **💻 Hardware Requirements**
   - 🖥️ **CPU**: x86_64 (AVX2 support recommended)
   - 🧠 **RAM**: 8GB minimum, 16GB+ recommended
   - 🎮 **GPU**: NVIDIA GPU with CUDA support (required: `docker-compose.yml` reserves one GPU; defaults were tested on a 12 GB RTX 3060)
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
   git checkout vllm_omni/khanhtts-omnivoice
   ```

3. **Set up environment configuration**
   ```bash
   # Create .env from .env.sample (skipped if .env already exists)
   make env
   # Edit the .env file to customize settings
   # nano .env  # or use your preferred text editor
   ```

4. **Start the service**
   ```bash
   # Pull the vllm-omni image and start the service in the background
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

   Other useful targets: `make down`, `make restart`, `make ps`, `make clean` (removes the container and network; the pulled `vllm-omni` image and the model cache in `HF_CACHE_DIR` are kept). Run `make help` for the full list.

6. **Access the service**
   - 🔌 **API Endpoint**: http://localhost:8002/v1 - OpenAI-compatible API
   - 📚 **API Documentation**: http://localhost:8002/docs - Interactive API docs
   - 📊 **Health Check**: http://localhost:8002/health - Service status

<br />

# 📝 Examples

All scripts talk to the server at `http://localhost:8002/v1` and write their `.wav` output into the `results/` directory. Install the client deps first: `pip install openai httpx requests`.

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

Measured with the OpenAI SDK (`client.audio.speech.create`, `response_format="wav"`, `language="Vietnamese"`) on the example text (`"Xin chào, đây là VoicePlatform. Chúc bạn một ngày tốt lành."`): one warm-up request, then 5 timed requests, one at a time.

| Model | GPU | Mean time | Audio | RTF (mean) | RTF (min – max) |
|-------|-----|-----------|-------|------------|-----------------|
| `kjanh/KhanhTTS-OmniVoice` | NVIDIA RTX 3060 | 1.508 s | 3.48 s | 0.433 | 0.432 – 0.435 |

> ℹ️ Times are end-to-end over localhost with `GPU_MEMORY_UTILIZATION` from `.env.sample`. The output length was identical in all 5 runs (3.48 s). The example script itself fires the OpenAI SDK and `httpx` requests at the same time, so its per-client times are higher than the table above. Results depend on GPU and concurrent load, so re-measure on your own hardware.

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

## 🌊 Sentence-level Streaming

**Synthesis**: HTTP · one `POST /v1/audio/speech` per sentence (client-side streaming)

OmniVoice is a diffusion TTS model, so vLLM-Omni returns each request as one complete clip (`stream` / `stream_format` are ignored). The `examples/sentence_streaming_example.py` gets streaming-like latency anyway by:
- ✂️ **Splitting** the paragraph into sentences
- 🔀 **Synthesizing** them concurrently with a bounded number of in-flight requests
- 💾 **Appending** each clip to the output WAV in order as soon as it is ready, reporting time-to-first-audio

```bash
# Run the sentence-level streaming example
python examples/sentence_streaming_example.py
```

### ⏱️ Streaming RTF and Time-to-First-Audio

Measured on a 4-sentence Vietnamese paragraph (the one in the example, 10.9 – 11.2 s of audio), with the default `max_concurrency=2`: one warm-up run, then 5 timed runs. Time-to-first-audio (TTFA) is the delay until the first sentence is ready. "One request" sends the whole paragraph in a single non-streaming request for comparison.

| Mode | Mean total time | TTFA (mean) | Audio | RTF (mean) | RTF (min – max) |
|------|-----------------|-------------|-------|------------|-----------------|
| One request (whole paragraph) | 4.495 s | 4.495 s | 10.88 s | 0.413 | 0.401 – 0.439 |
| Sentence-level streaming | 7.346 s | 1.930 s | 11.24 s | 0.654 | 0.631 – 0.693 |

> ℹ️ Sentence-level streaming trades throughput for latency: the first audio arrives about 2.3× sooner (and usually after about 1.6 s; one run took 3.3 s), but the whole paragraph takes longer because the sentences compete for the GPU and each request carries its own overhead. Use it when playback should start early; use a single request when total time matters most.

> ℹ️ OmniVoice has no built-in speakers: omit `voice` for an automatic voice, or pass `ref_audio` (+ `ref_text`) to clone one. A `language` hint such as `"Vietnamese"` or `"English"` is optional. Output is 24 kHz mono.

## 🎵 Sample Audio Files

Sample audio files are included in `resources/` (`sample_vi.mp3`, `sample_vi.wav`). Only `sample_vi.wav` is used by the examples, as the reference voice in the cloning example; `sample_vi.mp3` is an extra copy in another format.

<br />

# ⚙️ Configuration

## 🌍 Environment Variables

Customize the service behavior using these environment variables in your `.env` file:

```bash
# 🎯 Model configuration
MODEL_NAME=kjanh/KhanhTTS-OmniVoice     # TTS model to use (configurable)
GPU_MEMORY_UTILIZATION=0.5              # GPU memory allocation (0.0-1.0); 0.85 if the variable is unset
HF_TOKEN=                               # Optional; only if you hit Hugging Face rate limits

# 🌐 Network configuration
VLLM_HOST=0.0.0.0                       # Service host address
VLLM_PORT=8002                          # Host port (container always listens on 8002)

# 💾 Storage configuration
HF_CACHE_DIR=~/.cache/huggingface/hub      # Host directory mounted as the model cache
```

> ✅ These defaults were tested on an **NVIDIA RTX 3060**. The model is small (~0.6B): with `GPU_MEMORY_UTILIZATION=0.5` on the 12 GB card the vLLM process used about 4.1 GB (the cap is about 6 GB), so this setting mostly controls how much VRAM is left for other workloads on the same GPU.

<br />

# 📋 To-Do List
- [x] 🚀 Basic TTS service deployment
- [x] 📝 Example requests (non-streaming and sentence-level streaming)
- [x] 🎙️ Voice cloning example
- [x] ⏱️ RTF and time-to-first-audio measurements

<br />

# 💻 Technology Stack:
- 🎯 **TTS Model**: Configurable (default: kjanh/KhanhTTS-OmniVoice)
- ⚡ **Inference Engine**: vLLM-Omni (`vllm/vllm-omni:v0.26.0`)
- 🐳 **Containerization**: Docker & Docker Compose
- 🔌 **API Compatibility**: OpenAI API format
- 🖥️ **GPU Support**: NVIDIA CUDA
- 📊 **Monitoring**: Docker logging, `/health` and `/metrics`
- 🛠️ **Client Libraries**: OpenAI Python SDK, httpx, requests

<br />

# 🙏 Acknowledgments
- 🚀 [vLLM Team](https://github.com/vllm-project/vllm-omni) for the high-performance omni-modal inference engine
- 🤖 [kjanh](https://huggingface.co/kjanh/KhanhTTS-OmniVoice) for the KhanhTTS-OmniVoice Vietnamese fine-tune
- 🎙️ [k2-fsa Team](https://github.com/k2-fsa/OmniVoice) for the OmniVoice base model

