# 🎤 VoicePlatform - Private Text-to-Speech Service

A high-performance, privacy-focused text-to-speech (TTS) service powered by OpenBMB's VoxCPM2 model. This project provides a secure, local deployment solution for synthesizing speech from text without relying on cloud services, ensuring complete data privacy and control.

Built with vLLM-Omni for optimal performance and GPU acceleration, this service offers OpenAI-compatible API endpoints for seamless integration with existing applications and workflows.

<br />

# 🧠 Core Features

### 🎯 Advanced TTS Capabilities
- Natural, expressive speech synthesis using the state-of-the-art VoxCPM2 model
- Support for multiple languages
- OpenAI-compatible `/v1/audio/speech` API for easy integration with existing tools
- FP8 quantization out of the box for lower VRAM usage

### 🚀 Performance Optimizations
- GPU-accelerated inference with vLLM-Omni for real-time synthesis
- Configurable GPU memory utilization, KV cache budget and model parameters
- Efficient batch processing of concurrent requests
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
- Detailed logging and monitoring capabilities

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
   - ⚡ **CUDA Toolkit 11.8+** (for GPU acceleration)

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

All scripts talk to the server at `http://localhost:8002/v1` and write their `.wav` output into the `results/` directory. Install the client deps first: `pip install openai httpx requests`.

## 🔊 Speech Synthesis (non-streaming)

The `examples/speech_synthesis_example.py` shows how to:
- 🎯 **Generate** a complete audio clip from text via `/v1/audio/speech`
- 🔀 **Compare** the OpenAI SDK and raw `httpx` requests, run concurrently
- ⏱️ **Measure** end-to-end synthesis time

```bash
# Run the non-streaming synthesis example
python examples/speech_synthesis_example.py
```

## 🎙️ Voice Cloning (non-streaming)

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

The `examples/async_streaming_example.py` demonstrates:
- 🔄 **Streaming** raw PCM bytes with the async OpenAI client (`stream_format="audio"`)
- ⏱️ **Time-to-first-audio** measurement
- 💾 **Writing** chunks straight into a WAV file as they arrive

```bash
# Run the async streaming example
python examples/async_streaming_example.py
```

### 🔄 Synchronous Streaming (SSE)

The `examples/sync_streaming_example.py` shows:
- 📡 **Server-Sent Events** handling (`speech.audio.delta` / `speech.audio.done`)
- 🔐 **Decoding** base64 PCM deltas into a WAV file
- 🔗 **Direct API** integration with `requests`, no OpenAI SDK

```bash
# Run the sync streaming example
python examples/sync_streaming_example.py
```

> ℹ️ Streaming only supports `response_format` of `pcm` or `wav`; VoxCPM2 outputs 16-bit mono at 48 kHz.

## 🎵 Sample Audio Files

Sample audio files are included in `resources/` (`sample_vi.mp3`, `sample_vi.wav`) and are used as the reference voice in the cloning example.

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
- [ ] 🎙️ Voice cloning examples

<br />

# 💻 Technology Stack:
- 🎯 **TTS Model**: Configurable (default: openbmb/VoxCPM2)
- ⚡ **Inference Engine**: vLLM-Omni
- 🐳 **Containerization**: Docker & Docker Compose
- 🔌 **API Compatibility**: OpenAI API format
- 🖥️ **GPU Support**: NVIDIA CUDA
- 📊 **Monitoring**: Docker logging
- 🛠️ **Client Libraries**: OpenAI Python SDK

<br />

# 🙏 Acknowledgments
- 🚀 [vLLM Team](https://github.com/vllm-project/vllm-omni) for the high-performance omni-modal inference engine
- 🤖 [OpenBMB Team](https://github.com/OpenBMB/VoxCPM) for the VoxCPM2 TTS model

