# 🎤 VoicePlatform

Private, self-hosted voice services: speech recognition (ASR), voice activity detection (VAD) and text-to-speech (TTS). Everything runs on your own GPU, so no audio leaves your infrastructure and no external API key is needed.

This repository is organized **one branch per service**. Each branch is a self-contained project with its own code, Docker setup and README. This `main` branch contains only this overview: pick a branch below, check it out and follow its README.

<br />

# 🧭 Branches at a Glance

| Branch | Task | Model | Engine | Protocol | RTF |
|--------|------|-------|--------|----------|-----|
| [`vllm/qwen3_asr`](https://github.com/nlp4everyone/VoicePlatform/tree/vllm/qwen3_asr) | ASR | Qwen3-ASR | vLLM | Transcription: HTTP, SSE · Realtime: WebSocket ✅ | 0.024 |
| [`whisperlivekit/funasr`](https://github.com/nlp4everyone/VoicePlatform/tree/whisperlivekit/funasr) | ASR | Fun-ASR-MLT-Nano-2512 | vLLM (+ WhisperLiveKit planned) | Transcription: HTTP, SSE · Realtime: WebSocket 🚧 | not measured |
| [`vllm_omni/voxcpm`](https://github.com/nlp4everyone/VoicePlatform/tree/vllm_omni/voxcpm) | TTS | VoxCPM2 | vLLM-Omni | Synthesis: HTTP, SSE · Realtime: WebSocket ✅ | 0.25 |
| [`vllm_omni/khanhtts-omnivoice`](https://github.com/nlp4everyone/VoicePlatform/tree/vllm_omni/khanhtts-omnivoice) | TTS (Vietnamese + English) | KhanhTTS-OmniVoice | vLLM-Omni | Synthesis: HTTP · Realtime: ❌ | 0.43 |
| [`ray/nvidia_asr_with_vad`](https://github.com/nlp4everyone/VoicePlatform/tree/ray/nvidia_asr_with_vad) | VAD + ASR | Pyannote + NeMo Parakeet | FastAPI + Ray Serve | Transcription: HTTP · Realtime: ❌ | not measured |
| [`ray/nvidia_asr`](https://github.com/nlp4everyone/VoicePlatform/tree/ray/nvidia_asr) | ASR | NeMo Parakeet | FastAPI + Ray Serve | Transcription: HTTP · Realtime: ❌ | not measured |
| [`ray/vad`](https://github.com/nlp4everyone/VoicePlatform/tree/ray/vad) | VAD | Pyannote | FastAPI + Ray Serve | Detection: HTTP · Realtime: ❌ | not measured |

**Legend**: ✅ supported · 🚧 in progress · ❌ not supported

**Reading the Protocol column**: *Transcription* (ASR), *Synthesis* (TTS) and *Detection* (VAD) are the request/response APIs; *Realtime* is live audio (or live text for TTS) over a persistent connection. HTTP means a regular REST call and SSE means streamed events over HTTP.

<br />

# 🔎 Which Branch Should I Use?

| I need... | Use |
|-----------|-----|
| Live transcription of microphone audio over WebSocket | [`vllm/qwen3_asr`](https://github.com/nlp4everyone/VoicePlatform/tree/vllm/qwen3_asr) |
| Fast transcription of audio files with an OpenAI-compatible API | [`vllm/qwen3_asr`](https://github.com/nlp4everyone/VoicePlatform/tree/vllm/qwen3_asr) |
| A multilingual ASR model (Fun-ASR-MLT-Nano) | [`whisperlivekit/funasr`](https://github.com/nlp4everyone/VoicePlatform/tree/whisperlivekit/funasr) |
| Speech synthesis with streaming audio and a WebSocket endpoint | [`vllm_omni/voxcpm`](https://github.com/nlp4everyone/VoicePlatform/tree/vllm_omni/voxcpm) |
| Vietnamese / English speech synthesis with voice cloning | [`vllm_omni/khanhtts-omnivoice`](https://github.com/nlp4everyone/VoicePlatform/tree/vllm_omni/khanhtts-omnivoice) |
| Transcription that skips silence and returns word or segment timestamps | [`ray/nvidia_asr_with_vad`](https://github.com/nlp4everyone/VoicePlatform/tree/ray/nvidia_asr_with_vad) |
| Only voice activity detection (speech segments) | [`ray/vad`](https://github.com/nlp4everyone/VoicePlatform/tree/ray/vad) |

<br />

# 🚀 Getting Started

```bash
# Clone the repository (you land on main, which only holds this overview)
git clone https://github.com/nlp4everyone/VoicePlatform.git
cd VoicePlatform

# Fetch all branches and check out the one you picked
git fetch
git checkout vllm/qwen3_asr     # replace with any branch from the table above
```

Then follow the **Quick Start** in that branch's `README.md`.

<br />

# 📚 Branch Details

## 🎧 ASR with vLLM

- **`vllm/qwen3_asr`**: Qwen3-ASR (default `Qwen/Qwen3-ASR-1.7B`) served by vLLM. Offers `POST /v1/audio/transcriptions`, SSE streaming of the text, base64 audio through chat completions, and a WebSocket realtime endpoint (`/v1/realtime`). Default port `8001`.
- **`whisperlivekit/funasr`**: Fun-ASR-MLT-Nano-2512 served by vLLM. The checkpoint is converted once with `make model` (nothing is stored in the repository). [WhisperLiveKit](https://github.com/QuentinFuxa/WhisperLiveKit) is planned as the WebSocket layer, which is why Realtime is 🚧. Default port `8002`.

## 🔊 TTS with vLLM-Omni

- **`vllm_omni/voxcpm`**: VoxCPM2 (30 languages, 48 kHz output, FP8 by default). Offers `POST /v1/audio/speech`, voice cloning, raw-audio and SSE streaming, and a WebSocket endpoint with streaming text input (`/v1/audio/speech/stream`). Default port `8002`.
- **`vllm_omni/khanhtts-omnivoice`**: KhanhTTS-OmniVoice, a Vietnamese + English fine-tune of OmniVoice (24 kHz output) with voice cloning. The server returns each request as one complete clip, so there is no server-side streaming or WebSocket; the branch ships a client-side sentence-level streaming example instead. Default port `8002`.

## 🧩 ASR / VAD with Ray Serve

- **`ray/nvidia_asr_with_vad`**: a two-stage pipeline in which Pyannote detects speech segments and NVIDIA NeMo Parakeet transcribes only those segments, with cross-request batching and timestamp offset adjustment. OpenAI-compatible `POST /v1/audio/transcriptions`. Includes detailed English and Vietnamese docs.
- **`ray/nvidia_asr`**: batch ASR with NeMo Parakeet (`nvidia/parakeet-ctc-0.6b-vi` or `nvidia/parakeet-tdt-0.6b-v3`), word and segment timestamps.
- **`ray/vad`**: voice activity detection with Pyannote (`pyannote/segmentation-3.0`).

The Ray branches use `run_service.sh` and `config/config.toml` instead of a Makefile and `.env`-only configuration; see their READMEs for details.

<br />

# 📏 About the RTF Column

RTF (Real-Time Factor) is processing time divided by audio duration; below 1 means faster than real time. For ASR the duration is that of the input audio, for TTS it is that of the generated audio, so ASR and TTS values are not directly comparable.

- Measured on an **NVIDIA RTX 3060** (12 GB) over localhost with the OpenAI SDK, after one warm-up request, as the mean of 5 requests.
- ASR: `Qwen/Qwen3-ASR-0.6B` on the 7.81 s sample in `resources/sample_vi.mp3`. The branch default is the 1.7B model, which was not measured.
- TTS: a short Vietnamese sentence (`vllm_omni/voxcpm`: 0.253, `vllm_omni/khanhtts-omnivoice`: 0.433). Streaming and WebSocket modes of `vllm_omni/voxcpm` give about the same RTF but a much earlier first audio (about 0.17 s).
- Branches marked *not measured* have not been benchmarked yet. Each branch README has the full tables and conditions.

<br />

# 🗂️ How the Repository Is Organized

- The branches are **independent projects**: the Ray branches and the vLLM branches do not share git history, so they are not meant to be merged into each other.
- `main` only holds this overview. Fixes and features go to the branch they belong to.
- When a branch gains or loses a capability (for example WebSocket support in `whisperlivekit/funasr`), update its row in the table above.
