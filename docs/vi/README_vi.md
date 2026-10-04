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
- Sắp xếp batch theo độ dài để giảm padding lãng phí, và AMP autocast (`torch.cuda.amp.autocast`) cho inference mixed-precision
- Cache resampler (`torchaudio.transforms.Resample` được cache theo cặp `(src_sr, tgt_sr)`) và GPU executor đơn luồng riêng
- Hỗ trợ đa replica: scale bằng cách đặt `NUM_REPLICAS` trong `config/config.toml`

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
        │  .batched_transcribe.remote(audio_bytes, granularity)
        ▼
Ray Serve batch queue  (MAX_BATCH_SIZE, BATCH_WAIT_TIMEOUT_S)
        │
        │  load_audio_from_bytes() × N  [song song asyncio.to_thread]
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
   - 🧠 **RAM**: tối thiểu 16GB (container có thể dùng tới 16GB shared memory)
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

5. **Build và khởi động dịch vụ**
   ```bash
   # Build image (cài dependency từ uv.lock) và chạy nền
   make up
   # Hoặc chạy ở foreground
   make start
   ```
   Lần chạy đầu tiên cũng tải model vào `~/.cache/huggingface`, có thể mất vài phút. Docker mặc định được gọi bằng `sudo`. Dùng `make up DOCKER=docker` nếu user của bạn đã thuộc nhóm `docker`.

6. **Kiểm tra dịch vụ đang chạy**
   ```bash
   # Kiểm tra API đã phục vụ chưa
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

> ℹ️ Thời gian là end-to-end qua localhost (upload + decode + inference) với `NUM_REPLICAS=1` và giá trị mặc định trong `config/config.toml`; request đầu tiên (warm-up) mất 0.177 s. RTF phụ thuộc model, GPU và tải đồng thời, nên hãy đo lại trên phần cứng của bạn.

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

> ⚠️ **Giới hạn đã biết**: giữ số request đồng thời không vượt quá `MAX_ONGOING_REQUESTS / 2` (8 với mặc định 16). Mỗi request HTTP chiếm một slot trong lúc gọi `batched_transcribe` qua handle của chính deployment, và lời gọi này cần thêm một slot nữa. Với 16 request cùng lúc ở cấu hình mặc định, mọi slot bị các request bên ngoài chiếm hết và dịch vụ bị kẹt (log báo `Failed to route request after N attempts`) cho tới khi client ngắt kết nối. Nếu dự kiến có nhiều client đồng thời hơn, hãy tăng `MAX_ONGOING_REQUESTS` lên ít nhất gấp đôi số đó.

## 🎵 File audio mẫu

File audio mẫu nằm trong `resources/` (`sample_vi.wav` tiếng Việt, `sample_en.wav` tiếng Anh) để thử transcription ngay sau khi triển khai.

<br />

# ⚙️ Cấu hình

## 🎛️ Tham số serving

Hành vi serving được đặt trong `config/config.toml`. Thay đổi có hiệu lực sau `make restart`.

| Khóa | Mặc định | Mô tả |
|------|----------|-------|
| `[serving] NUM_GPUS` | `1` | Số GPU dành cho mỗi replica (giá trị phân số như `0.5` cho phép các replica dùng chung GPU) |
| `[serving] NUM_REPLICAS` | `1` | Số replica của model |
| `[serving] MAX_ONGOING_REQUESTS` | `16` | Số request đang xử lý tối đa trên mỗi replica (xem giới hạn đã biết ở mục Request đồng thời) |
| `[serving] MAX_BATCH_SIZE` | `8` | Số request tối đa được gộp vào một GPU batch |
| `[serving] BATCH_WAIT_TIMEOUT_S` | `0.1` | Thời gian một batch chờ cho đầy trước khi được gửi đi |
| `[asr] ASR_MODEL_NAME` | `nvidia/parakeet-ctc-0.6b-vi` | Model NeMo Parakeet cần load (tên phải bắt đầu bằng `nvidia/parakeet`) |
| `[asr] ASR_DEVICE` | `auto` | `auto`, `cuda` hoặc `cpu` |
| `[asr] SPLIT_MIXED_BATCH` | `true` | Chạy request có/không có timestamp thành các GPU sub-call riêng |

## 🌍 Biến môi trường

Đặt trong file `.env` (tạo bởi `make env`):

```bash
# 🌐 Cấu hình mạng
RAY_FASTAPI_PORT=8005                   # Cổng host của inference API (container lắng nghe ở 8000)
RAY_DASHBOARD_PORT=8265                 # Cổng host của Ray dashboard
```

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

<br />

# 💻 Công nghệ sử dụng
- 🎯 **Mô hình ASR**: NVIDIA NeMo Parakeet (mặc định: `nvidia/parakeet-ctc-0.6b-vi`)
- ⚡ **Serving**: FastAPI + Ray Serve (`ray[serve]==2.50.0`)
- 📦 **Dependency**: uv (`pyproject.toml` + `uv.lock`)
- 🐳 **Container hóa**: Docker & Docker Compose
- 🔌 **Tương thích API**: định dạng OpenAI API
- 🖥️ **Hỗ trợ GPU**: NVIDIA CUDA 12.8
- 🛠️ **Thư viện client**: OpenAI Python SDK

<br />

# 🙏 Lời cảm ơn
- 🚀 [Ray Team](https://github.com/ray-project/ray) cho Ray Serve
- 🤖 [NVIDIA NeMo Team](https://github.com/NVIDIA/NeMo) cho các mô hình ASR Parakeet
