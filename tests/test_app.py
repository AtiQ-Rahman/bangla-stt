"""The UI builds, and handlers return as many values as their event has outputs.

Mismatched counts (a button added to a list but not to every handler) only fail at click time
in the browser, so they are checked here.
"""
import os

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
app = pytest.importorskip("app")
from webui import common, controls, opening  # noqa: E402  (after the offline switch)


def outputs_of(fn_name: str) -> list[int]:
    """Output counts of every event that runs the function called fn_name."""
    return [len(block.outputs) for block in app.demo.fns.values()
            if getattr(block.fn, "__name__", "") == fn_name]


def test_button_state_handlers_match_controls():
    n = len(app.controls)
    assert set(outputs_of("idle_buttons")) == {n}
    assert len(controls.busy_buttons("transcribe")()) == n
    assert len(controls.idle_buttons("a.mp3", "text", None, None)) == n
    assert len(controls.stopped("a.mp3", "text", None, None, "transcribe")) == n + 4


@pytest.mark.parametrize("fn", ["clear_results"])
def test_fixed_handlers_match_their_outputs(fn):
    assert {len(getattr(controls, fn)())} == set(outputs_of(fn))


def test_open_from_history_matches_outputs(tmp_path, monkeypatch, segments):
    from stt import history
    monkeypatch.setattr(history, "HISTORY_DIR", tmp_path / "h")
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"x")
    job = {"audio": str(audio), "title": "t", "model": app.DEFAULT_MODEL, "quantize": True,
           "use_lm": True, "words": [], "start": None, "end": None}
    entry = history.create(job)
    history.save(entry, job, segments, 3.0, "stopped")
    values = opening.history_open(str(entry))
    assert len(values) == len(opening.OPEN_FIELDS)
    assert {len(values)} == set(outputs_of("history_open")) == set(outputs_of("open_queue_item"))


def test_parse_time():
    parse = common.parse_time
    assert parse("90") == 90 and parse("1:30") == 90 and parse("0:01:30") == 90
    assert parse("") is None


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
    field = dict(zip(opening.OPEN_FIELDS, range(len(opening.OPEN_FIELDS))))

    common.LIVE[folder] = {"segments": segments[:1], "frac": 0.4, "audio_sec": 10.0}
    try:
        values = opening.history_open(folder)
        assert values[field["follow"]] == folder and values[field["partial"]] is None
        assert "আমি ভাত খাই" in values[field["transcript"]]
        segments_now = segments
        common.LIVE[folder]["segments"] = segments_now
        tick = opening.follow_tick(folder)
        assert "মন্ত্রণালয়" in tick[field["transcript"]]
    finally:
        common.LIVE.pop(folder, None)

    history.save(entry, job, segments, 8.0, "done")
    done = opening.follow_tick(folder)
    assert done[field["follow"]] is None
    assert "youtu.be/x" in done[field["status"]] and "finished" in str(done[field["pbar"]].value)


def test_defaults_match_their_outputs(tmp_path, monkeypatch):
    from webui import settings
    monkeypatch.setattr(settings, "FILE", tmp_path / "s.json")
    assert {len(settings.apply_defaults())} == set(outputs_of("apply_defaults"))
    assert {len(settings.reset_defaults())} == set(outputs_of("reset_defaults"))
