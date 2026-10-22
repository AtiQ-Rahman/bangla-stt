"""History keeps runs on disk so they can be reopened and resumed."""
import pytest

from stt import history


@pytest.fixture(autouse=True)
def tmp_history(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "HISTORY_DIR", tmp_path / "history")


def job(tmp_path):
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"fake audio")
    return {"audio": str(audio), "title": "My talk: part 1?", "model": "MMS 1B (Meta) - best accuracy",
            "quantize": True, "use_lm": True, "words": [], "start": None, "end": None}


def test_create_save_load_roundtrip(tmp_path, segments):
    j = job(tmp_path)
    entry = history.create(j)
    assert j["audio"].startswith(str(entry))  # the audio was copied in
    history.save(entry, j, segments, 3.0, "stopped")
    meta = history.load(entry)
    assert meta["status"] == "stopped" and meta["at"] == 3.0
    assert meta["segments"][1].words[1].text == "প্রশাসন"
    assert (entry / "transcript.srt").read_text(encoding="utf-8").count("-->") == 2


def test_entries_list_and_delete(tmp_path, segments):
    entry = history.create(job(tmp_path))
    label, folder = history.entries()[0]
    assert "My talk_ part 1_" in folder and label.startswith("⏸")  # running = not finished
    history.save(entry, job(tmp_path), segments, 8.0, "done")
    assert history.entries()[0][0].startswith("✅")
    history.delete(entry)
    assert history.entries() == []


def test_delete_stays_inside_history(tmp_path):
    outside = tmp_path / "keep"
    outside.mkdir()
    history.delete(outside)
    assert outside.exists()


def test_search_finds_words_in_transcripts(tmp_path, segments):
    a = history.create(job(tmp_path))
    history.save(a, job(tmp_path), segments, 8.0, "done")
    history.save_text(a, "corrected.txt", "জনপ্রশাসন মন্ত্রণালয়ে আজ বৈঠক")
    assert len(history.entries("প্রশাসন")) == 1  # in the transcript
    assert len(history.entries("বৈঠক")) == 1  # only in the AI-fixed text
    assert history.entries("নেই") == []
    assert len(history.entries("my talk")) == 1  # the title, any case
    assert "ভাত" in history.snippet(a, "ভাত")
