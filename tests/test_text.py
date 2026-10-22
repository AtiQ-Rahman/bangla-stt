"""Turning timed segments into text, subtitles and other formats."""
from stt.export import bangla_digits, to_vtt
from stt.transcribe import Segment, Word, _join, to_srt, to_text


def test_srt_numbers_and_times(segments):
    srt = to_srt(segments)
    assert srt.startswith("1\n00:00:00,000 --> 00:00:03,000\nআমি ভাত খাই\n")
    assert "2\n00:00:05,000 --> 00:00:08,000\n" in srt


def test_srt_splits_long_cues():
    words = [Word(i * 0.5, i * 0.5 + 0.4, f"w{i}") for i in range(40)]  # 20 s, no pauses
    srt = to_srt([Segment(0, 20, " ".join(w.text for w in words), words)])
    assert srt.count("-->") >= 4  # cues are at most 6 s / 12 words


def test_vtt_uses_dots(segments):
    vtt = to_vtt(segments)
    assert vtt.startswith("WEBVTT\n\n00:00:00.000 --> 00:00:03.000\n")


def test_paragraph_breaks_on_pauses():
    a = Segment(0, 2, "one", [])
    b = Segment(2.1, 4, "two", [])
    c = Segment(9, 10, "three", [])  # 5 s pause before
    assert to_text([a, b, c]) == "one two\n\nthree"


def test_whisper_words_keep_their_spacing():
    assert _join([Word(0, 1, " Visit"), Word(1, 2, " stanford"), Word(2, 3, ".edu")]) == "Visit stanford.edu"
    assert _join([Word(0, 1, "আমি"), Word(1, 2, "যাব")]) == "আমি যাব"


def test_bangla_digits():
    assert bangla_digits("20 অক্টোবর 2026") == "২০ অক্টোবর ২০২৬"


def test_translation_reply_parsing():
    from stt.translate import _parse
    assert _parse("1. Hello\n2) World\n", 2) == ["Hello", "World"]
    assert _parse("1. Hello\nWorld", 2) is None  # a missing number: redo line by line


def test_srt_from_cues():
    from stt.export import srt_from_cues
    assert srt_from_cues([(0, 1.5, "Hi")]) == "1\n00:00:00,000 --> 00:00:01,500\nHi\n"
