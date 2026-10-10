# 🎤 VoicePlatform - Dịch vụ chuyển giọng nói thành văn bản riêng tư (NeMo Parakeet + Ray Serve)

Dịch vụ nhận dạng giọng nói (ASR) theo batch, hiệu năng cao và đề cao quyền riêng tư, dùng các mô hình NVIDIA NeMo Parakeet. Dự án cung cấp giải pháp triển khai cục bộ, an toàn để chuyển file audio thành văn bản mà không phụ thuộc dịch vụ đám mây, toàn bộ dữ liệu nằm trên hạ tầng của bạn.

Dịch vụ xây dựng trên FastAPI và Ray Serve, cung cấp endpoint `/v1/audio/transcriptions` tương thích OpenAI, nên ứng dụng hiện có chỉ cần đổi base URL để chuyển sang dùng.

<br />

# 🧠 Tính năng chính

### 🎯 Khả năng ASR nâng cao
- Nhận dạng chính xác với NVIDIA NeMo Parakeet (mặc định: `nvidia/parakeet-ctc-0.6b-vi` cho tiếng Việt)
- Timestamp theo từng từ (word) và theo đoạn (segment) khi cần
- API tương thích OpenAI, thay thế trực tiếp cho `client.audio.transcriptions.create()`
- Kiểm tra định dạng audio (MIME-type qua `python-magic`) trước khi xử lý

### 🚀 Tối ưu hiệu năng
- Inference tăng tốc bằng GPU, nhanh hơn thời gian thực khoảng 47 lần trên một GPU RTX 3060 (xem Real-Time Factor ở mục Ví dụ)
- Tự động gộp batch: Ray Serve gom các request đồng thời thành GPU batch (`MAX_BATCH_SIZE`, `BATCH_WAIT_TIMEOUT_S`)
- Tách mixed-batch: request có/không có timestamp trong cùng batch chạy thành các GPU sub-call riêng, tránh transfer logit không cần thiết
- Sắp xếp batch theo độ dài để giảm padding lãng phí, và AMP autocast (`torch.autocast`, chỉ trên CUDA) cho inference mixed-precision
- Giải mã audio thẳng từ bytes bằng TorchCodec (mono 16 kHz trong một lượt FFmpeg) trên pool giải mã có giới hạn, và GPU executor đơn luồng riêng
- Hỗ trợ đa replica: scale bằng cách đặt `num_replicas` trong `config/serve.yaml`

### 🔒 Quyền riêng tư & bảo mật
- Triển khai hoàn toàn cục bộ - audio không rời khỏi hạ tầng của bạn
- Không cần API key hay dịch vụ bên ngoài
- Model được tải một lần vào cache Hugging Face và không bao giờ commit vào repository

### 🔌 API thân thiện với developer
- RESTful API với endpoint tương thích OpenAI và tài liệu tương tác tại `/docs`
- Ví dụ chạy ngay cho request đơn lẻ và request đồng thời
- Triển khai bằng Docker, điều khiển qua một `Makefile`
- Ray dashboard ở cổng `8265` để theo dõi replica và request

<br />

# 📡 Khả năng hỗ trợ

| Khả năng | Giao thức | Endpoint | Trạng thái |
|----------|-----------|----------|------------|
| Transcription | HTTP | `POST /v1/audio/transcriptions` | ✅ |
| Transcription kèm timestamp theo từ | HTTP | `POST /v1/audio/transcriptions` với `timestamp_granularities=["word"]` | ✅ |
| Transcription kèm timestamp theo đoạn | HTTP | `POST /v1/audio/transcriptions` với `timestamp_granularities=["segment"]` | ✅ |
| Health check | HTTP | `GET /health` (200 `ok` / 503 `unhealthy`) | ✅ |
| Transcription (stream text) | HTTP (SSE) | - | ❌ |
| Transcription (audio base64) | HTTP | - | ❌ |
| Realtime (audio trực tiếp) | WebSocket | - | ❌ |

**Chú thích**: ✅ đã hỗ trợ · 🚧 đang phát triển · ❌ chưa hỗ trợ

> ℹ️ Nhánh này là dịch vụ batch: upload cả file và nhận về toàn bộ transcript. Để dùng audio trực tiếp qua WebSocket, xem nhánh `vllm/qwen3_asr`.

<br />

# 🏗️ Kiến trúc

```
Client (OpenAI SDK / HTTP)
        │
        │  POST /v1/audio/transcriptions
        ▼
FastAPI  (ASRService — Ray Serve ingress)
        │  kiểm tra model, đọc bytes, kiểm tra MIME audio
        │
        │  load_audio_from_bytes()  [ThreadPoolExecutor giải mã riêng]
        │  self.batched_transcribe(waveform, granularity)
        ▼
Ray Serve batch queue  (MAX_BATCH_SIZE, BATCH_WAIT_TIMEOUT_S)
        │
        │  process_batch_transcription() [GPU ThreadPoolExecutor riêng]
        ▼
ParakeetRecognizer
        │  sắp xếp theo độ dài → autocast → model.transcribe()
        ▼
NeMo ASRModel  (nvidia/parakeet-ctc-0.6b-vi  hoặc  nvidia/parakeet-tdt-0.6b-v3)
        │
        ▼
TranscriptionResult  →  TranscriptionResponse / WordResponse / SegmentResponse
```

<br />

# 🛠️ Yêu cầu hệ thống

1. **💻 Phần cứng**
   - 🖥️ **CPU**: x86_64 (khuyến nghị có AVX2)
   - 🧠 **RAM**: tối thiểu 16GB; container bị giới hạn 24GB (`mem_limit`), gồm 16GB shared memory và object store của Ray
   - 🎮 **GPU**: NVIDIA GPU hỗ trợ CUDA (khuyến nghị để có hiệu năng tối ưu)
   - 💾 **Lưu trữ**: nên dùng SSD; image khá lớn (~30GB gồm NeMo và PyTorch) cộng thêm cache model

2. **🛠️ Phần mềm**
   - 🐳 **Docker và Docker Compose**
   - 🎮 **NVIDIA Container Toolkit** (để dùng GPU)
   - ⚡ **NVIDIA driver** tương thích CUDA 12.8, phiên bản của base image `nvidia/cuda:12.8.1` (không cần cài CUDA Toolkit trên máy host, CUDA đã nằm trong image)
   - 📦 **[uv](https://docs.astral.sh/uv/)** (tùy chọn, chỉ khi chạy hoặc phát triển ngoài Docker: `uv sync`)

<br />

# 🚀 Bắt đầu nhanh

1. **Clone repository**
   ```bash
   git clone https://github.com/nlp4everyone/VoicePlatform.git
   cd VoicePlatform
   ```

2. **Chuyển sang đúng nhánh**
   ```bash
   git fetch
   git checkout ray/nvidia_asr
   ```

3. **Thiết lập cấu hình môi trường**
   ```bash
   # Tạo .env từ .env.sample (bỏ qua nếu .env đã tồn tại)
   make env
   # Sửa file .env để đổi cổng nếu cần
   # nano .env
   ```

4. **Cấu hình tham số serving** trong `config/config.toml` cho phù hợp phần cứng (xem mục Cấu hình bên dưới):
   ```toml
   MAX_BATCH_SIZE = 8
   BATCH_WAIT_TIMEOUT_S = 0.1
   DECODE_WORKERS = 4
   ASR_MODEL_NAME = "nvidia/parakeet-ctc-0.6b-vi"
   ASR_DEVICE = "auto"
   ```

5. **Build và khởi động dịch vụ**
   ```bash
   # Build image (cài dependency từ uv.lock) và chạy nền
   make up
   # Hoặc chạy ở foreground
   make start
   ```
   Lần chạy đầu tiên cũng tải model vào `~/.cache/huggingface`, có thể mất vài phút. Để tải trước, chạy `make prefetch` trước `make up`; sau khi xong, đặt `HF_HUB_OFFLINE=1` trong `.env` để khi khởi động không gọi HuggingFace. Docker mặc định được gọi bằng `sudo`. Dùng `make up DOCKER=docker` nếu user của bạn đã thuộc nhóm `docker`.

6. **Kiểm tra dịch vụ đang chạy**
   ```bash
   # Kiểm tra replica còn khỏe không (GET /health); docker ps cũng hiển thị trạng thái health của container
   make health
   # Xem log dịch vụ
   make logs
   ```

   Các target hữu ích khác: `make down`, `make restart`, `make ps`, `make clean`. Chạy `make help` để xem đầy đủ.

7. **Truy cập dịch vụ**
   - 🔌 **API Endpoint**: http://localhost:8005/v1 - API tương thích OpenAI (cổng đặt bởi `RAY_FASTAPI_PORT` trong `.env`)
   - 📚 **Tài liệu API**: http://localhost:8005/docs - Tài liệu API tương tác
   - 📊 **Ray Dashboard**: http://localhost:8265 - Replica, request và log

<br />

# 📝 Ví dụ

Tất cả ví dụ gửi `resources/sample_vi.wav` tới `http://localhost:8005/v1`. Cài thư viện client trước: `pip install openai`. Chạy từ thư mục gốc của project vì đường dẫn audio là đường dẫn tương đối.

## 📄 Transcription audio

**Transcription**: HTTP · `POST /v1/audio/transcriptions`

Xem file `examples/audio_transcription_example.py` để biết cách:
- 🎯 **Transcribe** file audio bằng OpenAI SDK
- 🕒 **Yêu cầu** timestamp theo từ với `timestamp_granularities=["word"]`
- ⏱️ **Đo** thời gian transcription

```bash
python examples/audio_transcription_example.py
```

Một client tối giản:

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

RTF = thời gian xử lý ÷ thời lượng audio. Giá trị nhỏ hơn 1 nghĩa là nhanh hơn thời gian thực.

Đo bằng OpenAI SDK (`client.audio.transcriptions.create`) trên `resources/sample_vi.wav` (7.63 s audio): một request warm-up, sau đó 5 request được tính giờ.

| Model | GPU | Chế độ | Audio | Thời gian TB | RTF (TB) | RTF (min – max) |
|-------|-----|--------|-------|--------------|----------|-----------------|
| `nvidia/parakeet-ctc-0.6b-vi` | NVIDIA RTX 3060 | Chỉ text | 7.63 s | 0.162 s | 0.021 | 0.021 – 0.022 |
| `nvidia/parakeet-ctc-0.6b-vi` | NVIDIA RTX 3060 | Timestamp theo từ | 7.63 s | 0.192 s | 0.025 | 0.024 – 0.026 |

> ℹ️ Thời gian là end-to-end qua localhost (upload + decode + inference) với `num_replicas=1` và giá trị mặc định trong `config/config.toml`; request đầu tiên (warm-up) mất 0.177 s. RTF phụ thuộc model, GPU và tải đồng thời, nên hãy đo lại trên phần cứng của bạn.

## 🚦 Request đồng thời

File `examples/concurrent_requests_example.py` gửi nhiều request cùng lúc bằng async OpenAI client để bạn thấy Ray Serve gộp batch hoạt động.

```bash
python examples/concurrent_requests_example.py
```

Các request đồng thời được gom thành GPU batch (tối đa `MAX_BATCH_SIZE`), nên tổng thời gian gần như không tăng theo số request. Đo với cùng audio, một replica, 3 vòng tính giờ sau một vòng warm-up:

| Số request đồng thời | Thời gian mỗi vòng | Độ trễ TB mỗi request | RTF tổng hợp |
|----------------------|--------------------|-----------------------|--------------|
| 4 | 0.206 s | 0.204 s | 0.0068 |
| 8 | 0.187 s | 0.182 s | 0.0031 |

> RTF tổng hợp = thời gian mỗi vòng ÷ (số request × thời lượng audio), thể hiện thông lượng.

## 🎵 File audio mẫu

File audio mẫu nằm trong `resources/` (`sample_vi.wav` tiếng Việt, `sample_en.wav` tiếng Anh) để thử transcription ngay sau khi triển khai.

<br />

# 🩺 Health Check & Warm-up

- **`GET /health`** trả `200 {"status": "ok"}` khi replica còn chạy được model, ngược lại trả `503 {"status": "unhealthy"}`. `make health` và healthcheck của Docker đều gọi endpoint này, nên `docker ps` hiển thị container là `healthy` hoặc `unhealthy`.
- **Restart replica**: Ray Serve gọi `ASRService.check_health()` mỗi `health_check_period_s` (`config/serve.yaml`). Trên CUDA, hàm này cấp phát một tensor nhỏ, nên khi CUDA context hỏng (ví dụ lỗi Xid hoặc ECC) nó raise và Serve restart replica. Healthcheck của Docker chỉ báo trạng thái, không restart container.
- **Warm-up**: mỗi replica transcribe `resources/sample_vi.wav` lúc khởi động (batch size 1 và `MAX_BATCH_SIZE`, có và không có timestamp) trước khi Serve chuyển traffic tới, để request thật đầu tiên không phải trả chi phí khởi tạo CUDA. Replica được restart sau khi health check lỗi cũng được warm-up lại. Đặt `ASR_WARMUP=false` để bỏ qua.

<br />

# ⚙️ Cấu hình

## 🎛️ Tham số serving

Tham số model và batching được đặt trong `config/config.toml`; mỗi khóa có thể được ghi đè bằng biến môi trường `ASR_<KEY>`, không lặp tiền tố (ví dụ `ASR_MAX_BATCH_SIZE`, `ASR_DEVICE`) trong `.env`, giá trị này được ưu tiên hơn. Số replica, GPU/CPU, `max_ongoing_requests` và host/port nằm trong `config/serve.yaml` (`num_replicas`, `ray_actor_options.num_gpus`, `max_ongoing_requests`, `http_options`). Thay đổi có hiệu lực sau `make restart`.

| Khóa | Mặc định | Mô tả |
|------|----------|-------|
| `MAX_BATCH_SIZE` | `8` | Số request tối đa được gộp vào một GPU batch |
| `BATCH_WAIT_TIMEOUT_S` | `0.1` | Thời gian một batch chờ cho đầy trước khi được gửi đi |
| `DECODE_WORKERS` | `4` | Số luồng giải mã audio mỗi replica; đồng thời là `num_cpus` Ray dành cho replica |
| `ASR_MODEL_NAME` | `nvidia/parakeet-ctc-0.6b-vi` | Model NeMo Parakeet cần load (tên phải bắt đầu bằng `nvidia/parakeet`) |
| `ASR_DEVICE` | `auto` | `auto`, `cuda` hoặc `cpu` |
| `SPLIT_MIXED_BATCH` | `true` | Chạy request có/không có timestamp thành các GPU sub-call riêng |
| `WARMUP` | `true` | Transcribe `resources/sample_vi.wav` lúc khởi động (batch size 1 và `MAX_BATCH_SIZE`, có/không timestamp) để mỗi replica được làm nóng trước khi nhận traffic |

## 🌍 Biến môi trường

Đặt trong file `.env` (tạo bởi `make env`):

```bash
# 🌐 Cấu hình mạng
RAY_FASTAPI_PORT=8005                   # Cổng host của inference API (container lắng nghe ở 8000)
RAY_DASHBOARD_PORT=8265                 # Cổng host của Ray dashboard

# 🎛️ Ghi đè tùy chọn config/config.toml (để trống = giữ giá trị trong file)
ASR_MAX_BATCH_SIZE=16
ASR_DEVICE=cuda
```

<br />

# 🧪 Kiểm thử

Bộ test chạy trên CPU, không cần GPU hay NeMo: NeMo được thay bằng module giả trong `tests/conftest.py`, và `ASRService` được chạy qua `TestClient` của FastAPI với recognizer giả. Test bao phủ helper batching, giải mã audio, settings, recognizer và API (transcription, response lỗi, `/health`, warm-up). Test không load model thật, nên cần kiểm tra phần đó trên máy có GPU bằng `make up`.

```bash
# Cài một lần (cần FFmpeg và libmagic, ví dụ apt install ffmpeg libmagic1)
uv venv
uv pip install --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match -r tests/requirements.txt

make test PYTHON=.venv/bin/python   # pytest
make lint PYTHON=.venv/bin/python   # ruff
```

GitHub Actions (`.github/workflows/ci.yml`) chạy lint và test này mỗi lần push lên nhánh `ray/nvidia_asr`. Nó không deploy gì.

<br />

# 📚 Tài liệu

- [Technical Overview](TECHNICAL_OVERVIEW_vi.md) — kiến trúc đầy đủ, pipeline xử lý, tham chiếu cấu hình
- [Flow](FLOW_vi.md) — luồng khởi động và luồng xử lý từng request
- [Detailed Components](DETAILED_COMPONENTS_vi.md) — chi tiết từng thành phần và tham chiếu API
- [Design Decisions](DESIGN_DECISIONS_vi.md) — lý do và đánh đổi của từng quyết định thiết kế chính

<br />

# 📋 To-Do List
- [x] 🚀 Dịch vụ ASR batch trên Ray Serve với NeMo Parakeet
- [x] 🔌 Endpoint `/v1/audio/transcriptions` tương thích OpenAI
- [x] 🕒 Timestamp theo từ và theo đoạn
- [x] 📝 Ví dụ minh họa (request đơn lẻ và đồng thời)
- [x] ⏱️ Đo RTF trên audio mẫu
- [x] 🧪 Unit test và integration test trên CPU, CI bằng GitHub Actions
- [ ] 🔐 Xác thực: thêm API key (qua header) hoặc đặt service sau một gateway
- [ ] 🔎 Request ID và log có cấu trúc, để dễ truy vết khi lỗi
- [ ] 📊 Metrics (độ trễ, RTF, độ dài audio, kích thước batch, tỉ lệ lỗi) qua Prometheus exporter của Ray và Grafana

<br />

# 💻 Công nghệ sử dụng
- 🎯 **Mô hình ASR**: NVIDIA NeMo Parakeet (mặc định: `nvidia/parakeet-ctc-0.6b-vi`)
- ⚡ **Serving**: FastAPI + Ray Serve (`ray[serve]==2.50.0`)
- 📦 **Dependency**: uv (`pyproject.toml` + `uv.lock`)
- 🐳 **Container hóa**: Docker & Docker Compose
- 🔌 **Tương thích API**: định dạng OpenAI API
- 🖥️ **Hỗ trợ GPU**: NVIDIA CUDA 12.8
- 🛠️ **Thư viện client**: OpenAI Python SDK
- 🧪 **Kiểm thử**: pytest + ruff, GitHub Actions

<br />

# 🙏 Lời cảm ơn
- 🚀 [Ray Team](https://github.com/ray-project/ray) cho Ray Serve
- 🤖 [NVIDIA NeMo Team](https://github.com/NVIDIA/NeMo) cho các mô hình ASR Parakeet
