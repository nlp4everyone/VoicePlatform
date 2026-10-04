from typing import Tuple, Union
from functools import lru_cache
import soundfile as sf
import io, torch, torchaudio, magic, logging
from app.exceptions.audio import InvalidAudioException
magic_mime = magic.Magic(mime=True)
logger = logging.getLogger("ray.serve")


@lru_cache(maxsize=8)
def _get_resampler(sr_src: int, sr_tgt: int) -> torchaudio.transforms.Resample:
    return torchaudio.transforms.Resample(sr_src, sr_tgt)

def estimate_audio_duration(audio_bytes: bytes) -> float:
    """
    Estimate the duration of audio data in seconds.
    
    Args:
        audio_bytes: Raw audio data bytes
        
    Returns:
        Duration in seconds as a float
    """
    with io.BytesIO(audio_bytes) as f:
        info = sf.info(f)
        return info.frames / info.samplerate

def load_audio_from_bytes(audio_bytes: bytes,
                          target_sr :int = 16000) -> Tuple[torch.Tensor, float]:
    """
    Load audio from bytes and convert to mono tensor at target sample rate.

    Args:
        audio_bytes: Raw audio data bytes
        target_sr: Target sample rate (default: 16000)

    Returns:
        Tuple of (waveform tensor, duration in seconds)

    Raises:
        InvalidAudioException: If the audio cannot be decoded or has no samples
    """
    buffer = io.BytesIO(audio_bytes)
    try:
        waveform, sr = torchaudio.load(buffer)
    except Exception as e:
        # The decoder error names internal objects, so log it and keep the client message generic
        logger.warning(f"Audio decoding failed: {e}")
        raise InvalidAudioException(reason="the file is corrupted or uses an unsupported codec") from e

    # A valid header with no samples would fail later inside the model
    if waveform.numel() == 0:
        raise InvalidAudioException(reason="audio contains no samples")

    waveform = waveform[0] if waveform.shape[0] == 1 else waveform.mean(dim=0)

    if sr != target_sr:
        waveform = _get_resampler(sr, target_sr)(waveform)

    duration = round(waveform.shape[-1] / target_sr, 3)
    return waveform, duration

def is_audio_file(data: bytes,
                  buffer_size: int = 2048) -> bool:
    """
    Check if the given bytes represent an audio file.

    Args:
        data: Raw bytes to check for audio content
        buffer_size: Number of bytes to read from the data for MIME detection (default: 2048)

    Returns:
        True if the data appears to be an audio file, False otherwise
    """
    try:
        if not data:
            return False

        # Only need a small prefix
        mime = magic_mime.from_buffer(data[:buffer_size])

        return mime.startswith("audio/") or mime in {
            "application/ogg",  # common edge case
        }

    except Exception:
        return False