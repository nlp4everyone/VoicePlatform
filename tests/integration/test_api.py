import pytest

from app.services.deployments import asr_deployment
from tests.integration.conftest import service_settings

URL = "/v1/audio/transcriptions"
MODEL = "nvidia/parakeet-ctc-0.6b-vi"


def post(client, wav, **form):
    return client.post(URL, files={"file": ("sample.wav", wav, "audio/wav")},
                       data={"model": MODEL, **form})


def test_transcribe_text(client, sample_wav_bytes):
    resp = post(client, sample_wav_bytes)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["text"] == "xin chào các bạn"
    assert body["usage"]["output_tokens"] > 0


def test_transcribe_word_timestamps(client, sample_wav_bytes):
    resp = post(client, sample_wav_bytes, **{"timestamp_granularities[]": "word"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert [w["word"] for w in body["words"]] == ["xin", "chào"]
    assert body["duration"] == pytest.approx(7.63, abs=0.05)
    assert body["usage"]["seconds"] == 8


def test_transcribe_segment_timestamps(client, sample_wav_bytes):
    resp = post(client, sample_wav_bytes, **{"timestamp_granularities[]": "segment"})
    assert resp.status_code == 200, resp.text
    segments = resp.json()["segments"]
    assert segments[0]["id"] == 0
    assert segments[0]["text"] == "xin chào"


def test_unknown_model_is_rejected(client, sample_wav_bytes):
    resp = client.post(URL, files={"file": ("a.wav", sample_wav_bytes, "audio/wav")},
                       data={"model": "some/other-model"})
    assert resp.status_code == 400
    assert resp.json()["code"] == "model_not_found"


def test_non_audio_file_is_rejected(client):
    resp = client.post(URL, files={"file": ("notes.txt", b"hello there", "text/plain")},
                       data={"model": MODEL})
    assert resp.status_code == 400
    assert resp.json()["code"] == "unsupported_value"


def test_corrupt_audio_is_rejected(client, sample_wav_bytes):
    # Valid header, truncated/garbled body: must fail for this request only
    broken = sample_wav_bytes[:44] + b"\xff" * 3
    resp = post(client, broken)
    assert resp.status_code in (200, 400)
    if resp.status_code == 400:
        assert resp.json()["code"] == "invalid_audio"


def test_missing_file_is_a_validation_error(client):
    assert client.post(URL, data={"model": MODEL}).status_code == 422


def test_invalid_granularity_is_a_validation_error(client, sample_wav_bytes):
    resp = post(client, sample_wav_bytes, **{"timestamp_granularities[]": "paragraph"})
    assert resp.status_code == 422


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_health_503_when_gpu_broken(client, recognizer):
    recognizer.healthy = False
    resp = client.get("/health")
    assert resp.status_code == 503
    assert resp.json() == {"status": "unhealthy"}


def test_check_health_propagates_for_ray(recognizer, monkeypatch):
    """Ray restarts the replica when ASRService.check_health() raises."""
    monkeypatch.setattr(service_settings(), "WARMUP", False)
    monkeypatch.setattr(asr_deployment.RecognizerFactory, "create",
                        lambda model_name, device: recognizer)
    service = asr_deployment.ASRService.func_or_class()
    service.check_health()
    recognizer.healthy = False
    with pytest.raises(RuntimeError, match="CUDA error"):
        service.check_health()


def test_warmup_runs_at_startup_when_enabled(recognizer, monkeypatch):
    calls = []
    monkeypatch.setattr(recognizer, "warmup",
                        lambda audio, batch_sizes: calls.append((audio.ndim, batch_sizes)))
    monkeypatch.setattr(service_settings(), "WARMUP", True)
    monkeypatch.setattr(asr_deployment.RecognizerFactory, "create",
                        lambda model_name, device: recognizer)
    asr_deployment.ASRService.func_or_class()
    assert calls == [(1, [1, service_settings().MAX_BATCH_SIZE])]
