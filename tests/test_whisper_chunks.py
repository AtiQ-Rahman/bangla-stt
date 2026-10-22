"""Chunked Whisper (the Bangla model): chunks cut at quiet moments, silence skipped, times offset."""
from types import SimpleNamespace

import numpy as np

from stt import transcribe as T
from stt.audio import SAMPLE_RATE


class FakeWhisper:
    """Records each chunk's length and returns one word in its middle, then any extra words."""

    def __init__(self, extra=()):
        self.chunks, self.extra = [], list(extra)

    def transcribe(self, chunk, **kw):
        self.chunks.append(len(chunk) / SAMPLE_RATE)
        mid = len(chunk) / SAMPLE_RATE / 2
        words = [SimpleNamespace(start=mid, end=mid + 0.2, word="শব্দ", probability=1.0)] + self.extra
        return [SimpleNamespace(words=words)], None


def test_chunks_cut_skip_silence_and_offset():
    t = np.arange(40 * SAMPLE_RATE) / SAMPLE_RATE
    wav = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    wav[13 * SAMPLE_RATE:13 * SAMPLE_RATE + SAMPLE_RATE // 2] = 0  # a pause to cut at
    wav[28 * SAMPLE_RATE:40 * SAMPLE_RATE] = 0  # silence at the end
    model = FakeWhisper()
    spec = {"language": "bn", "chunk_sec": 14.0}
    out = list(T._whisper_chunks(model, spec, wav, 100.0, None, 0.0))
    assert abs(model.chunks[0] - 13.25) < 0.4  # cut in the pause, not at 14 s
    assert out[-1][2] is None  # the silent end was skipped, not sent to Whisper
    first = out[0][2]
    assert first.words[0].start > 100 and first.text == "শব্দ"


def test_invented_words_at_chunk_end_are_dropped():
    wav = (0.3 * np.sin(2 * np.pi * 220 * np.arange(5 * SAMPLE_RATE) / SAMPLE_RATE)).astype(np.float32)
    model = FakeWhisper([SimpleNamespace(start=3.0, end=3.4, word=" আর", probability=0.95),
                         SimpleNamespace(start=3.4, end=3.6, word=" রাত", probability=0.5),
                         SimpleNamespace(start=3.6, end=3.6, word=" সাড়ে", probability=0.9)])
    out = list(T._whisper_chunks(model, {"language": "bn", "chunk_sec": 14.0}, wav, 0.0, None, 0.0))
    assert out[0][2].text == "শব্দ আর"  # zero-length and unsure words at the end are gone


def test_text_is_nfc():
    assert T._nfc("য়") == T._nfc("য়")  # য় written two ways ends up the same
