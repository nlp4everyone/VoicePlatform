"""Drive the real ASRService (routes, validation, batching) with a fake recognizer.

The Serve class is instantiated directly and its ASGI app is served through
FastAPI's TestClient, so no Ray cluster is started. Class-based routes look the
instance up through serve.get_replica_context(), which is stubbed here.
"""
import types

import pytest
from fastapi.testclient import TestClient

from app.schema.transcription.base import BaseRecognizer, TranscribedSegment, TranscribedWord
from app.schema.transcription.response import TranscriptionResult
from app.services.deployments import asr_deployment

MODEL_NAME = "nvidia/parakeet-ctc-0.6b-vi"


class FakeRecognizer(BaseRecognizer):
    def __init__(self):
        super().__init__(model_name=MODEL_NAME)
        self.healthy = True
        self.batches = []

    def check_health(self):
        if not self.healthy:
            raise RuntimeError("CUDA error: unspecified launch failure")

    def transcribe(self, audio, enable_timestamps=False, precision=3):
        self.batches.append((len(audio), enable_timestamps))
        results = []
        for _ in audio:
            if enable_timestamps:
                results.append(TranscriptionResult(
                    text="xin chào các bạn",
                    words=[TranscribedWord(start=0.0, end=0.4, word="xin"),
                           TranscribedWord(start=0.4, end=0.8, word="chào")],
                    segments=[TranscribedSegment(start=0.0, end=0.8, text="xin chào")]))
            else:
                results.append(TranscriptionResult(text="xin chào các bạn"))
        return results


def service_settings():
    """The settings object ASRService actually reads.

    serve.deployment round-trips the class through cloudpickle, which copies
    module-level instances such as `settings` by value, so patching
    asr_deployment.settings would not reach the service.
    """
    user_class = asr_deployment.ASRService.func_or_class.__mro__[1]
    return user_class.__init__.__globals__["settings"]


def _reset_batch_queue():
    """Drop the serve.batch queue so each test builds one on its own event loop.

    The queue lives in a wrapper shared by every instance of the class and binds
    to the first event loop that uses it; a later TestClient (new loop) would hang.
    """
    lazy_queue = asr_deployment.ASRService.func_or_class.batched_transcribe.set_max_batch_size.__self__
    lazy_queue._queue = None


@pytest.fixture
def recognizer():
    return FakeRecognizer()


@pytest.fixture
def client(recognizer, monkeypatch):
    # Warm-up has its own unit test; keep service start-up instant here
    monkeypatch.setattr(service_settings(), "WARMUP", False)
    monkeypatch.setattr(asr_deployment.RecognizerFactory, "create",
                        lambda model_name, device: recognizer)

    service = asr_deployment.ASRService.func_or_class()
    monkeypatch.setattr(asr_deployment.serve, "get_replica_context",
                        lambda: types.SimpleNamespace(
                            servable_object=service,
                            _deployment_config=types.SimpleNamespace(max_ongoing_requests=16)))

    _reset_batch_queue()
    with TestClient(service._asgi_app) as test_client:
        yield test_client
    _reset_batch_queue()
