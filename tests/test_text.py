"""Turning timed segments into text, subtitles and other formats."""
from stt.export import bangla_digits, parse_srt, to_vtt
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


def test_speaker_turns_in_text_and_subtitles():
    words = [Word(0, 1, "hello", 1), Word(1, 2, "there", 1), Word(2.2, 3, "hi", 2), Word(3, 4, "back", 2),
             Word(4.1, 5, "ok", 1)]
    segs = [Segment(0, 5, "hello there hi back ok", words)]
    assert to_text(segs) == "Speaker 1: hello there\n\nSpeaker 2: hi back\n\nSpeaker 1: ok"
    srt = to_srt(segs)
    assert srt.count("-->") == 3 and "Speaker 2: hi back" in srt


def test_label_assigns_words_to_turns():
    from stt.speakers import label
    segs = [Segment(0, 6, "a b c", [Word(0, 1, "a"), Word(2, 3, "b"), Word(4.5, 5.5, "c")])]
    out = label(segs, [(0, 2.5, 1), (4, 6, 2)])
    assert [w.speaker for w in out[0].words] == [1, 1, 2]


def test_retime_ignores_speaker_labels():
    from stt.retime import retime
    segs = [Segment(0, 2, "hi there", [Word(0, 1, "hi", 1), Word(1, 2, "there", 1)])]
    out = retime("Speaker 1: Hi there.", segs)
    assert [w.text for w in out[0].words] == ["Hi", "there."] and out[0].words[0].speaker == 1


def test_speaker_turns_keep_spaces_between_whisper_chunks():
    """A Whisper chunk's first word has no leading space: it must not stick to the last one."""
    segs = [Segment(0, 2, "করা হয়েছে", [Word(0, 1, "করা", 1), Word(1, 2, " হয়েছে", 1)]),
            Segment(2, 4, "বিশ অক্টোবর", [Word(2, 3, "বিশ", 1), Word(3, 4, " অক্টোবর", 1)])]
    assert to_text(segs) == "Speaker 1: করা হয়েছে বিশ অক্টোবর"


def test_parse_srt_edited_elsewhere():
    # BOM and Windows line ends (Notepad), a cue over two lines, a VTT-style "." in the time
    text = "﻿1\r\n00:00:01,000 --> 00:00:02,500\r\nআমি\r\nভালো\r\n\r\n2\n00:00:04.2 --> 00:00:05,000\nআছি\n"
    assert parse_srt(text) == [(1.0, 2.5, "আমি ভালো"), (4.2, 5.0, "আছি")]
