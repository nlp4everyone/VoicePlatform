# Tổng quan kỹ thuật

## Kiến trúc

```
┌──────────────────────────────────────────────────────────────────┐
│                  CLIENT (OpenAI SDK / curl)                      │
│         POST /v1/audio/transcriptions  (multipart/form-data)     │
└───────────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                  Ray Serve  (serve run config/serve.yaml)        │
│  http_options: host / port  (config/serve.yaml)                  │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  ASRService  (@serve.deployment)                           │  │
│  │  num_replicas                                              │  │
│  │  ray_actor_options.num_gpus                                │  │
│  │  max_ongoing_requests                                      │  │
│  │  ray_actor_options.num_cpus (>= DECODE_WORKERS)            │  │
│  │                                                            │  │
│  │  FastAPI ingress (@serve.ingress)                          │  │
│  │   POST /v1/audio/transcriptions                            │  │
│  │       │ kiểm tra model, đọc bytes, MIME check              │  │
│  │       │ run_in_executor(decode pool, load_audio_from_bytes)│  │
│  │       │ self.batched_transcribe(waveform, granularity)     │  │
│  │       ▼                                                    │  │
│  │  @serve.batch                                              │  │
│  │  batched_transcribe(waveforms, granularities)              │  │
│  │       │ run_in_executor(GPU thread, process_batch)         │  │
│  │       ▼                                                    │  │
│  │  process_batch_transcription()                             │  │
│  │       │ tách ts / no-ts sub-batch                          │  │
│  │       ▼                                                    │  │
│  │  ParakeetRecognizer.transcribe()                           │  │
│  │       │ sắp xếp theo độ dài → autocast → model.transcribe()│  │
│  │       ▼                                                    │  │
│  │  NeMo ASRModel  (GPU)                                      │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Pipeline xử lý

### Giai đoạn 1 — Kiểm tra request

`transcribe_audio()` nhận file upload và:
1. Kiểm tra `model` có khớp với `asr_model.model_name` đang chạy không — nếu không raise `TranscriptedModelNotFoundException`.
2. Đọc toàn bộ audio bytes vào memory.
3. Kiểm tra MIME type qua `python-magic` — raise `UnsupportedAudioFormatException` nếu không phải audio.

### Giai đoạn 2 — Giải mã audio (theo từng request, CPU)

`load_audio_from_bytes()` chạy trên pool giải mã riêng của replica (`ThreadPoolExecutor(max_workers=DECODE_WORKERS)`), nên các request đồng thời được giải mã song song mà không chặn event loop:
- `AudioDecoder(audio_bytes, sample_rate=16000, num_channels=1)` (TorchCodec) giải mã thẳng từ bytes, gộp về mono và resample về 16kHz trong một lượt FFmpeg
- Trả về `(waveform: torch.Tensor, duration: float)`

Việc giải mã diễn ra trước khi gom batch, nên file hỏng hoặc rỗng chỉ khiến request đó nhận `InvalidAudioException` (400). Raw bytes được giải phóng ngay sau khi giải mã.

### Giai đoạn 3 — Xếp hàng batch

Waveform đã giải mã và `timestamp_granularity` được truyền vào `batched_transcribe` bằng lời gọi method trực tiếp. Ray Serve gom các call đồng thời trong replica thành một batch (tối đa `MAX_BATCH_SIZE`) trong thời gian tối đa `BATCH_WAIT_TIMEOUT_S` giây. Gọi trực tiếp thay vì qua deployment handle giúp mỗi request chỉ chiếm một slot `max_ongoing_requests`.

### Giai đoạn 4 — GPU transcription (luồng riêng)

`process_batch_transcription()` chạy trên `ThreadPoolExecutor` 1 luồng (mỗi replica một executor) để giữ GPU work trên một thread duy nhất, tránh CUDA context migration:

| Thành phần batch | Số GPU call |
|---|---|
| Toàn bộ không có timestamp | 1 call, `timestamps=False` |
| Toàn bộ có timestamp | 1 call, `timestamps=True` |
| Hỗn hợp | 2 sub-call — item có ts + item không có ts |

Trong mỗi GPU call, `ParakeetRecognizer.transcribe()`:
1. Sắp xếp tensor theo độ dài tăng dần — giảm padding thừa trong các sub-batch nội bộ của NeMo.
2. Chạy `model.transcribe()` dưới `torch.autocast(device_type="cuda")` (FP16, chỉ trên CUDA).
3. Khôi phục thứ tự gốc trước khi trả về.

### Giai đoạn 5 — Định dạng response

Dựa trên `timestamp_granularity`:
- `None` → `TranscriptionResponse` — text + token usage
- `"word"` → `WordResponse` — text, ngôn ngữ phát hiện, thời lượng, timestamp theo từ
- `"segment"` → `SegmentResponse` — text, ngôn ngữ phát hiện, thời lượng, timestamp theo đoạn

Phát hiện ngôn ngữ dùng `pycld2`, chỉ chạy cho response có timestamp.

---

## Cấu hình

Tham số model và batching nằm trong `config/config.toml`; replica, tài nguyên và host/port nằm trong `config/serve.yaml`. Mỗi khóa có thể được ghi đè bằng biến môi trường `ASR_<KEY>`, không lặp tiền tố (ưu tiên: env > `config.toml` > mặc định). Thay đổi yêu cầu restart container.

### Serving

| Key | Mặc định | Mô tả |
|---|---|---|
| `MAX_BATCH_SIZE` | `8` | Số item tối đa mỗi GPU batch |
| `BATCH_WAIT_TIMEOUT_S` | `0.1` | Thời gian chờ tối đa để điền đầy batch (giây) |
| `DECODE_WORKERS` | `4` | Số luồng giải mã audio mỗi replica |

### Model

| Key | Mặc định | Mô tả |
|---|---|---|
| `ASR_MODEL_NAME` | `nvidia/parakeet-ctc-0.6b-vi` | Định danh model trên HuggingFace |
| `ASR_DEVICE` | `auto` | `auto` / `cuda` / `cpu` |

### `config/serve.yaml`

| Key | Mô tả |
|---|---|
| `proxy_location` | Nơi chạy HTTP proxy (`EveryNode`) |
| `http_options.host` / `port` | HTTP host và port của Ray Serve |
| `http_options.request_timeout_s` | Proxy trả 408 và hủy request sau số giây này (mặc định `120`); công việc đã chạy trong thread decode/GPU không bị dừng |
| `applications[0].deployments[0].num_replicas` | Số replica ASRService |
| `...max_ongoing_requests` | Số request đang xử lý tối đa mỗi replica |
| `...max_queued_requests` | Số request chờ tối đa ngoài `max_ongoing_requests` (mặc định `32`); khi đầy, request mới nhận 503 |
| `...ray_actor_options.num_gpus` / `num_cpus` | GPU và CPU dành cho mỗi replica |

### Biến môi trường (`.env` / `docker-compose.yml`)

| Biến | Mô tả |
|---|---|
| `RAY_FASTAPI_PORT` | Host port map tới Ray Serve HTTP (mặc định `8000`) |
| `RAY_DASHBOARD_PORT` | Host port map tới Ray dashboard (mặc định `8265`) |
| `RAY_LOG_LEVEL` | Log level nội bộ của Ray (mặc định `WARNING`) |
| `RAY_SERVE_LOG_TO_STDERR` | Chuyển Serve logs ra stderr (mặc định `1`) |
| `RAY_memory_monitor_refresh_ms` | Chu kỳ giám sát OOM của Ray (mặc định `250`, `0` = tắt) |
| `RAY_memory_usage_threshold` | Tỷ lệ RAM khiến Ray kill worker (mặc định `0.95`) |
| `RAY_object_store_memory` | Dung lượng object store của Ray, tính bằng byte (mặc định 4 GiB) |
| `HF_HOME` | Thư mục cache HuggingFace |

---

## Cấu trúc thư mục

```
VoicePlatform/
├── app/
│   ├── app.py                          # Entry point: ASRService.bind (no ray.init / serve.start)
│   ├── core/config/
│   │   └── settings.py                 # Settings: MAX_BATCH_SIZE, BATCH_WAIT_TIMEOUT_S, DECODE_WORKERS, ASR_*
│   ├── services/
│   │   ├── asr/
│   │   │   ├── factory.py              # RecognizerFactory — tạo ParakeetRecognizer
│   │   │   └── nemo/recognizer.py      # ParakeetRecognizer — bọc NeMo ASRModel
│   │   └── deployments/
│   │       └── asr_deployment.py       # ASRService — Ray Serve deployment + FastAPI
│   ├── utils/
│   │   ├── audio/io.py                 # load_audio_from_bytes, is_audio_file
│   │   └── transcription/helper.py     # process_batch_transcription
│   └── schema/transcription/           # TranscriptionResult, các kiểu response
├── config/config.toml                  # Toàn bộ cấu hình runtime
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
└── examples/                           # Ví dụ sử dụng
```
