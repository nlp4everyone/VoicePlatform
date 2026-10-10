import pytest
import torch

from app.schema.transcription import TranscriptionType
from app.schema.transcription.response import TranscriptionResult
from app.utils.transcription.helper import (get_timestamp_indices,
                                            get_transcription_type,
                                            process_batch_transcription)


class FakeModel:
    """Records each transcribe call and echoes the audio index as the text."""

    def __init__(self):
        self.calls = []

    def transcribe(self, audio, enable_timestamps):
        self.calls.append((len(audio), enable_timestamps))
        return [TranscriptionResult(text=f"{int(a.item())}",
                                    words=[] if enable_timestamps else None,
                                    segments=[] if enable_timestamps else None)
                for a in audio]


def make_audio(n):
    # one-element tensor per item; the value identifies it
    return [torch.tensor([float(i)]) for i in range(n)]


def test_get_timestamp_indices_partitions():
    ts, no_ts = get_timestamp_indices(["word", None, "segment", None])
    assert ts == [0, 2]
    assert no_ts == [1, 3]


def test_get_timestamp_indices_empty():
    assert get_timestamp_indices([]) == ([], [])


@pytest.mark.parametrize("value,expected", [
    (None, TranscriptionType.Text),
    ("word", TranscriptionType.Word),
    ("segment", TranscriptionType.Segment),
    ("anything-else", TranscriptionType.Text),
])
def test_get_transcription_type(value, expected):
    assert get_transcription_type(value) == expected


def test_pure_no_timestamp_batch_is_one_call():
    model = FakeModel()
    results = process_batch_transcription(model, make_audio(3), [None] * 3)
    assert model.calls == [(3, False)]
    assert [r.text for r in results] == ["0", "1", "2"]


def test_pure_timestamp_batch_is_one_call():
    model = FakeModel()
    process_batch_transcription(model, make_audio(2), ["word", "segment"])
    assert model.calls == [(2, True)]


def test_mixed_batch_split_keeps_original_order():
    model = FakeModel()
    results = process_batch_transcription(model, make_audio(4),
                                          ["word", None, "segment", None],
                                          split_mixed_batch=True)
    assert sorted(model.calls) == [(2, False), (2, True)]
    assert [r.text for r in results] == ["0", "1", "2", "3"]
    assert results[0].words is not None and results[1].words is None


def test_mixed_batch_without_split_strips_unused_timestamps():
    model = FakeModel()
    results = process_batch_transcription(model, make_audio(3),
                                          ["word", None, "word"],
                                          split_mixed_batch=False)
    assert model.calls == [(3, True)]
    assert results[1].words is None and results[1].segments is None
    assert results[0].words is not None


def test_path_inputs_are_converted_to_str():
    from pathlib import Path

    seen = {}

    class PathModel:
        def transcribe(self, audio, enable_timestamps):
            seen["audio"] = audio
            return [TranscriptionResult(text="x")]

    process_batch_transcription(PathModel(), [Path("a.wav")], [None])
    assert seen["audio"] == ["a.wav"]
