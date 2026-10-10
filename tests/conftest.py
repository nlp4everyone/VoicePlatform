"""Shared fixtures. NeMo is replaced by a fake module so the suite runs on CPU
without installing NeMo or downloading a model."""
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_WAV = ROOT / "resources" / "sample_vi.wav"


def _install_fake_nemo() -> types.ModuleType:
    """Register nemo.collections.asr in sys.modules, exposing a mockable ASRModel."""
    nemo = types.ModuleType("nemo")
    collections = types.ModuleType("nemo.collections")
    asr = types.ModuleType("nemo.collections.asr")
    asr.models = types.SimpleNamespace(ASRModel=mock.MagicMock(name="ASRModel"))
    nemo.collections = collections
    collections.asr = asr
    sys.modules.update({"nemo": nemo,
                        "nemo.collections": collections,
                        "nemo.collections.asr": asr})
    return asr


try:
    import nemo.collections.asr as _nemo_asr  # noqa: F401  (real NeMo, if installed)
    FAKE_NEMO = None
except ImportError:
    FAKE_NEMO = _install_fake_nemo()


@pytest.fixture
def sample_wav_bytes() -> bytes:
    return SAMPLE_WAV.read_bytes()
