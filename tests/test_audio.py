"""Decoding, cutting and splitting audio."""
from stt.audio import SAMPLE_RATE, cut_clip, duration, load_audio, split_on_silence


def test_load_and_range(tone_gap_tone):
    assert abs(len(load_audio(tone_gap_tone)) / SAMPLE_RATE - 6.5) < 0.05
    assert abs(len(load_audio(tone_gap_tone, 1, 4)) / SAMPLE_RATE - 3.0) < 0.05


def test_cut_clip_and_duration(tone_gap_tone, tmp_path):
    clip = cut_clip(tone_gap_tone, tmp_path / "part.mp3", 1, 3)
    assert abs(duration(clip) - 2.0) < 0.1


def test_split_on_silence_finds_the_gap(tone_gap_tone):
    spans = split_on_silence(load_audio(tone_gap_tone), max_sec=4)
    assert len(spans) == 2
    assert abs(spans[0][1] / SAMPLE_RATE - 3.0) < 0.2


def test_split_keeps_long_speech_whole():
    import numpy as np
    wav = np.sin(np.arange(30 * SAMPLE_RATE) / 10).astype(np.float32)  # 30 s, no silence
    assert split_on_silence(wav, max_sec=20) == [(0, len(wav))]
