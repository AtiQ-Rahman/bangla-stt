"""Learning the user's corrected words (stt.dictionary) and using them as hotwords."""
import pytest

from stt import dictionary
from stt.transcribe import DEFAULT_MODEL
from webui.transcribing import make_job


@pytest.fixture(autouse=True)
def own_file(tmp_path, monkeypatch):
    monkeypatch.setattr(dictionary, "FILE", tmp_path / "dictionary.json")


def test_corrections_are_the_changed_words(segments):
    edited = "Speaker 1: আমি ভাত খাই।\nজন প্রশাসন রাজশাহী"  # one real change; punctuation is no change
    assert dictionary.corrections(segments, edited) == ["রাজশাহী"]
    assert dictionary.corrections(segments, "আমি ভাত খাই 20 জন প্রশাসন মন্ত্রণালয়") == []  # added only


def test_learn_counts_once_per_transcript(segments):
    edited = "আমি ভাত খাই জন প্রশাসন রাজশাহী"
    assert dictionary.learn(segments, edited, "a") == ["রাজশাহী"]
    assert dictionary.learn(segments, edited, "a") == []  # the same edit saved again
    dictionary.learn(segments, "আমি Dhaka খাই জন প্রশাসন রাজশাহী", "b")
    assert dictionary.words("Bangla") == ["রাজশাহী"]
    assert dictionary.words("English") == ["Dhaka"]


def test_edit_the_list_and_use_it(segments):
    dictionary.learn(segments, "আমি ভাত খাই জন প্রশাসন রাজশাহী", "a")
    dictionary.set_words("Bangla", "রাজশাহী, তারেক রহমান,, ")
    assert dictionary.words("Bangla") == ["রাজশাহী", "তারেক রহমান"]
    job = make_job("x.wav", "t", DEFAULT_MODEL, True, True, "প্রথম আলো, রাজশাহী", "", "", False)
    assert job["words"] == ["প্রথম আলো", "রাজশাহী", "তারেক রহমান"]
    off = make_job("x.wav", "t", DEFAULT_MODEL, True, True, "", "", "", False, use_learned=False)
    assert off["words"] == []
