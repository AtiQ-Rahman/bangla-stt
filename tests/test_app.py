"""The UI builds, and handlers return as many values as their event has outputs.

Mismatched counts (a button added to a list but not to every handler) only fail at click time
in the browser, so they are checked here.
"""
import os

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
app = pytest.importorskip("app")


def outputs_of(fn_name: str) -> list[int]:
    """Output counts of every event that runs the function called fn_name."""
    return [len(block.outputs) for block in app.demo.fns.values()
            if getattr(block.fn, "__name__", "") == fn_name]


def test_button_state_handlers_match_controls():
    n = len(outputs_of("idle_buttons") and app.demo.fns[next(
        k for k, b in app.demo.fns.items() if getattr(b.fn, "__name__", "") == "idle_buttons")].outputs)
    assert len(app.busy_buttons("transcribe")()) == n
    assert len(app.idle_buttons("a.mp3", "text", None, None)) == n
    assert len(app.stopped("a.mp3", "text", None, None, "transcribe")) == n + 3


@pytest.mark.parametrize("fn", ["clear_results"])
def test_fixed_handlers_match_their_outputs(fn):
    assert {len(getattr(app, fn)())} == set(outputs_of(fn))


def test_open_from_history_matches_outputs(tmp_path, monkeypatch, segments):
    from stt import history
    monkeypatch.setattr(history, "HISTORY_DIR", tmp_path / "h")
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"x")
    job = {"audio": str(audio), "title": "t", "model": app.DEFAULT_MODEL, "quantize": True,
           "use_lm": True, "words": [], "start": None, "end": None}
    entry = history.create(job)
    history.save(entry, job, segments, 3.0, "stopped")
    values = app.history_open(str(entry))
    assert {len(values)} == set(outputs_of("history_open")) == set(outputs_of("open_queue_item"))


def test_parse_time():
    assert app.parse_time("90") == 90 and app.parse_time("1:30") == 90 and app.parse_time("0:01:30") == 90
    assert app.parse_time("") is None


def test_open_and_follow(tmp_path, monkeypatch, segments):
    """An entry being transcribed right now is followed live, then shown finished."""
    from stt import history
    monkeypatch.setattr(history, "HISTORY_DIR", tmp_path / "h")
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"x")
    job = {"audio": str(audio), "title": "t", "model": app.DEFAULT_MODEL, "quantize": True,
           "use_lm": True, "words": [], "start": None, "end": None, "source": "https://youtu.be/x"}
    entry = history.create(job)
    folder = str(entry)
    field = dict(zip(app.OPEN_FIELDS, range(len(app.OPEN_FIELDS))))

    app.LIVE[folder] = {"segments": segments[:1], "frac": 0.4, "audio_sec": 10.0}
    try:
        values = app.history_open(folder)
        assert values[field["follow"]] == folder and values[field["partial"]] is None
        assert "আমি ভাত খাই" in values[field["transcript"]]
        segments_now = segments
        app.LIVE[folder]["segments"] = segments_now
        tick = app.follow_tick(folder)
        assert "মন্ত্রণালয়" in tick[field["transcript"]]
    finally:
        app.LIVE.pop(folder, None)

    history.save(entry, job, segments, 8.0, "done")
    done = app.follow_tick(folder)
    assert done[field["follow"]] is None
    assert "youtu.be/x" in done[field["status"]] and "finished" in str(done[field["pbar"]].value)
