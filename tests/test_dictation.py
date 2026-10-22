"""Live dictation: phrases are cut at pauses (or when long) and transcribed in order."""
import numpy as np

from stt.audio import SAMPLE_RATE
from stt.dictation import Dictation, to_16k_mono
from stt.transcribe import Segment

SR = 48000  # what browsers usually record at


def tone(sec, loud=0.3):
    t = np.arange(int(sec * SR)) / SR
    return (loud * np.sin(2 * np.pi * 220 * t) * 32767).astype(np.int16)


def quiet(sec):
    return (np.random.default_rng(0).normal(0, 30, int(sec * SR))).astype(np.int16)  # mic hiss


def dictate(*parts):
    heard = []

    def fake(wav, offset):
        heard.append((round(offset, 1), round(len(wav) / SAMPLE_RATE, 1)))
        return [Segment(offset, offset + len(wav) / SAMPLE_RATE, f"phrase{len(heard)}", [])]

    d = Dictation(fake)
    audio = np.concatenate(parts)
    for i in range(0, len(audio), SR // 2):  # 0.5 s chunks, like the page sends
        d.feed(SR, audio[i:i + SR // 2])
    during = list(heard)
    d.finish()
    return d, during, heard


def test_phrases_end_at_pauses():
    d, during, heard = dictate(quiet(1), tone(1.5), quiet(1.2), tone(2), quiet(1.2), tone(1))
    assert len(during) == 2  # the first two were written while "speaking"
    assert len(heard) == 3 and heard[0][0] < 1.0 < heard[0][0] + heard[0][1]
    assert d.text() == "phrase1 phrase2 phrase3" and not d.pending
    assert abs(len(d.audio) / SAMPLE_RATE - 7.9) < 0.01


def test_long_speech_is_cut_and_silence_skipped():
    _, _, heard = dictate(tone(32))
    assert all(sec <= 10 for _, sec in heard) and sum(sec for _, sec in heard) == 32
    _, _, heard = dictate(quiet(10))
    assert heard == []


def test_chunk_conversion():
    out = to_16k_mono(44100, np.stack([tone(1), tone(1)], axis=1))
    assert out.dtype == np.float32 and abs(len(out) - SAMPLE_RATE * SR / 44100) < 2
    assert 0.25 < np.abs(out).max() < 0.35
