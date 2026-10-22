"""Edited or AI-fixed text keeps the timing of the original words."""
from stt.retime import retime
from stt.transcribe import to_srt


def test_unchanged_text_keeps_times(segments):
    out = retime("আমি ভাত খাই জন প্রশাসন মন্ত্রণালয়", segments)
    assert [(s.start, s.end) for s in out] == [(0, 3), (5, 8)]


def test_joined_and_respelled_words(segments):
    out = retime("আমি ভাত খাই। জনপ্রশাসন মন্ত্রণালয়ে", segments)
    words = {w.text: (w.start, w.end) for s in out for w in s.words}
    assert words["খাই।"] == (2, 3)
    assert words["জনপ্রশাসন"] == (5, 7)  # took the time of "জন প্রশাসন"
    assert "জনপ্রশাসন মন্ত্রণালয়ে" in to_srt(out)


def test_added_word_sits_after_the_previous_one(segments):
    out = retime("আমি ভাত খাই জন প্রশাসন মন্ত্রণালয় আজ", segments)
    last = out[-1].words[-1]
    assert last.text == "আজ" and last.start == 8


def test_nothing_to_time():
    assert retime("text", []) == []
