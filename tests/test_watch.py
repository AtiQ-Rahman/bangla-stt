"""Folder watch: new audio/video files join the batch queue once, after they stop changing."""
import os
import time

import pytest

from webui import settings, watch


@pytest.fixture(autouse=True)
def own_files(tmp_path, monkeypatch):
    monkeypatch.setattr(watch, "SEEN", tmp_path / "watched.json")
    monkeypatch.setattr(settings, "FILE", tmp_path / "settings.json")


def put(folder, name, age=60):
    path = folder / name
    path.write_bytes(b"x" * 10)
    os.utime(path, (time.time() - age,) * 2)
    return path


def test_new_settled_media_files_are_queued_once(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    put(inbox, "talk.mp3"), put(inbox, "clip.MP4"), put(inbox, "notes.txt"), put(inbox, ".hidden.mp3")
    put(inbox, "copying.wav", age=0)  # still arriving
    queue, rows, _, start = watch.tick(True, str(inbox), [], "")
    assert sorted(item["name"] for item in queue) == ["clip.MP4", "talk.mp3"] and start
    assert len(rows) == 2
    assert not isinstance(watch.tick(True, str(inbox), queue, "")[0], list)  # nothing new
    assert [p.name for p in watch.new_files(inbox)] == []  # both seen; copying.wav not settled yet


def test_nothing_while_busy_or_off(tmp_path):
    put(tmp_path, "a.mp3")
    assert all(not isinstance(v, list) for v in watch.tick(True, str(tmp_path), [], "batch"))
    assert all(not isinstance(v, list) for v in watch.tick(False, str(tmp_path), [], ""))
    assert len(watch.tick(True, str(tmp_path), [], "")[0]) == 1


def test_toggle_remembers_the_folder(tmp_path):
    folder = tmp_path / "new inbox"
    watch.toggle(True, str(folder))
    assert folder.is_dir() and settings.load()["watch_folder"] == str(folder)
