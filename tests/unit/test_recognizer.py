from unittest import mock

import pytest
import torch

from app.schema.transcription.base import BaseRecognizer
from app.services.asr.nemo import recognizer as rec_module
from app.services.asr.nemo.recognizer import ParakeetRecognizer


class FakeResult:
    def __init__(self, text, timestamp=None):
        self.text = text
        self.timestamp = timestamp


@pytest.fixture
def recognizer(monkeypatch):
    """ParakeetRecognizer on CPU with a fake NeMo model."""
    model = mock.MagicMock(name="model")
    model.eval.return_value = model
    fake_nemo = mock.MagicMock()
    fake_nemo.models.ASRModel.from_pretrained.return_value = model
    monkeypatch.setattr(rec_module, "nemo_asr", fake_nemo)
    r = ParakeetRecognizer(model_name="nvidia/parakeet-ctc-0.6b-vi", device="cpu")
    return r, model


def test_unsupported_model_is_rejected():
    with pytest.raises(ValueError, match="Unsupported ASR model"):
        ParakeetRecognizer(model_name="not/a-model", device="cpu")


def test_load_failure_becomes_runtime_error(monkeypatch):
    fake_nemo = mock.MagicMock()
    fake_nemo.models.ASRModel.from_pretrained.side_effect = OSError("no network")
    monkeypatch.setattr(rec_module, "nemo_asr", fake_nemo)
    with pytest.raises(RuntimeError, match="Failed to load"):
        ParakeetRecognizer(model_name="nvidia/parakeet-ctc-0.6b-vi", device="cpu")


def test_cuda_falls_back_to_cpu_when_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    fake_nemo = mock.MagicMock()
    monkeypatch.setattr(rec_module, "nemo_asr", fake_nemo)
    r = ParakeetRecognizer(model_name="nvidia/parakeet-ctc-0.6b-vi", device="cuda")
    assert r._device == "cpu"


def test_transcribe_restores_input_order(recognizer):
    r, model = recognizer
    # lengths 30, 10, 20 -> the model sees them sorted: 10, 20, 30
    audio = [torch.zeros(30), torch.zeros(10), torch.zeros(20)]
    model.transcribe.return_value = [FakeResult("short"), FakeResult("mid"), FakeResult("long")]
    results = r.transcribe(audio)
    assert [len(a) for a in model.transcribe.call_args.args[0]] == [10, 20, 30]
    assert [x.text for x in results] == ["long", "short", "mid"]


def test_transcribe_single_tensor_is_wrapped(recognizer):
    r, model = recognizer
    model.transcribe.return_value = [FakeResult("hi")]
    assert r.transcribe(torch.zeros(5))[0].text == "hi"


def test_transcribe_with_timestamps_rounds_and_maps(recognizer):
    r, model = recognizer
    model.transcribe.return_value = [FakeResult("xin chào", {
        "word": [{"start": 0.123456, "end": 0.5, "word": "xin"}],
        "segment": [{"start": 0.0, "end": 1.23456, "segment": "xin chào"}],
    })]
    result = r.transcribe([torch.zeros(5)], enable_timestamps=True, precision=2)[0]
    assert result.words[0].start == 0.12 and result.words[0].word == "xin"
    assert result.segments[0].end == 1.23 and result.segments[0].text == "xin chào"


def test_transcribe_error_propagates(recognizer):
    r, model = recognizer
    model.transcribe.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError, match="boom"):
        r.transcribe([torch.zeros(5)])


def test_check_health_noop_on_cpu(recognizer):
    r, _ = recognizer
    r.check_health()


def test_check_health_raises_when_gpu_broken(recognizer, monkeypatch):
    r, _ = recognizer
    r._device = "cuda"

    def broken(*args, **kwargs):
        raise RuntimeError("CUDA error: unspecified launch failure")

    monkeypatch.setattr(torch, "empty", broken)
    with pytest.raises(RuntimeError, match="CUDA error"):
        r.check_health()


def test_warmup_runs_each_batch_size_with_and_without_timestamps(recognizer):
    r, model = recognizer
    model.transcribe.side_effect = lambda audio, timestamps: [
        FakeResult("x", {"word": [], "segment": []}) for _ in audio]
    r.warmup(torch.zeros(100), batch_sizes=[1, 4, 4])
    calls = [(len(c.args[0]), c.kwargs["timestamps"]) for c in model.transcribe.call_args_list]
    assert calls == [(1, False), (1, True), (4, False), (4, True)]


def test_base_recognizer_defaults_are_noops():
    base = BaseRecognizer(model_name="m")
    assert base.check_health() is None
    assert base.warmup(torch.zeros(1), [1]) is None
