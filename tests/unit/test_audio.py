import pytest
import torch

from app.exceptions.audio import InvalidAudioException
from app.utils.audio import (estimate_audio_duration, is_audio_file,
                             load_audio_from_bytes)


def test_is_audio_file_accepts_wav(sample_wav_bytes):
    assert is_audio_file(sample_wav_bytes)


@pytest.mark.parametrize("data", [b"", b"just some plain text", b"\x00" * 64])
def test_is_audio_file_rejects_non_audio(data):
    assert not is_audio_file(data)


def test_load_audio_resamples_to_mono_16k(sample_wav_bytes):
    # sample_vi.wav is 44.1 kHz, ~7.6 s
    waveform, duration = load_audio_from_bytes(sample_wav_bytes)
    assert isinstance(waveform, torch.Tensor)
    assert waveform.ndim == 1
    assert duration == pytest.approx(7.63, abs=0.05)
    assert waveform.shape[-1] == pytest.approx(duration * 16000, abs=20)  # duration is rounded to 3 decimals


def test_load_audio_rejects_corrupt_bytes():
    with pytest.raises(InvalidAudioException):
        load_audio_from_bytes(b"RIFF....not really a wav file")


def test_estimate_audio_duration(sample_wav_bytes):
    assert estimate_audio_duration(sample_wav_bytes) == pytest.approx(7.63, abs=0.05)
