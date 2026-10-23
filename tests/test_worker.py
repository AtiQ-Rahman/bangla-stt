"""The worker process protocol: pieces stream back, Stop ends a job, a dead worker is reported."""
import time

import pytest

from stt import activity, worker as worker_mod
from stt.worker import Worker, WorkerDied

from conftest import sample


def _fake_serve(requests, replies, cancel):
    """A stand-in worker: 5 pieces, 0.2 s apart; "hang" never stops; "die" exits mid-job."""
    while True:
        job_id, args = requests.get()
        cancel.clear()
        if args.get("task") == "speakers":  # one long step that can't stop by itself
            time.sleep(30)
            continue
        replies.put(("start", job_id, 16000 * 5))
        if args["model"] == "loading":
            replies.put(("activity", job_id, (7, "🧠 Loading the test model…")))
            replies.put(("activity", job_id, (8, "🌐 Downloading…")))
            replies.put(("activity", job_id, (8, None)))
        if args["model"] == "die":
            raise SystemExit(1)
        stopped = None
        for i in range(1, 6):
            time.sleep(10 if args["model"] == "hang" else 0.2)
            replies.put(("piece", job_id, (i, 5, f"piece {i}")))
            if cancel.is_set():
                stopped = "cancelled"
                break
        replies.put(("end", job_id, stopped))


def run(w, model, **kw):
    return w.transcribe("x.wav", 0.0, None, model, True, False, [], 0.0, **kw)


def test_pieces_stream_back():
    w = Worker(target=_fake_serve)
    try:
        replies = list(run(w, "ok"))
        assert replies[0] == ("start", 16000 * 5)
        assert [v[2] for k, v in replies[1:]] == [f"piece {i}" for i in range(1, 6)]
        assert w.is_loaded("ok", True)
    finally:
        w.kill()


def test_cancel_ends_the_job_and_keeps_the_worker():
    w = Worker(target=_fake_serve)
    try:
        replies = []
        for kind, value in run(w, "ok"):
            replies.append(kind)
            if kind == "piece":
                w.cancel()
        assert replies[-1] == "cancelled" and replies.count("piece") <= 2
        assert w.alive()  # the model stays loaded for the next job
        assert [k for k, _ in run(w, "ok")].count("piece") == 5
    finally:
        w.kill()


def test_abandoned_job_that_wont_stop_kills_the_worker(monkeypatch):
    monkeypatch.setattr(worker_mod, "STOP_GRACE_SEC", 0.5)
    w = Worker(target=_fake_serve)
    gen = run(w, "hang")
    assert next(gen)[0] == "start"
    t = time.monotonic()
    gen.close()  # what Stop does to a running job
    assert time.monotonic() - t < 3 and not w.alive()


def test_dead_worker_is_reported():
    w = Worker(target=_fake_serve)
    with pytest.raises(WorkerDied):
        list(run(w, "die"))


@pytest.mark.models
def test_real_transcription_in_the_worker():
    path = sample("bn")
    w = Worker()
    try:
        pieces = [v for k, v in w.transcribe(path, 0.0, 20.0, "XLS-R 300M Bengali - faster", True, False,
                                              [], 0.0) if k == "piece"]
        text = " ".join(seg.text for _, _, seg in pieces if seg)
        assert len(text) > 50
    finally:
        w.kill()


def _orphan_check(requests, replies, cancel):
    """Start the parent-watch like the real worker, then report our pid and wait."""
    import os
    import threading
    threading.Thread(target=worker_mod._exit_with_parent, daemon=True).start()
    replies.put(("start", 0, os.getpid()))
    time.sleep(60)


def test_worker_exits_when_its_parent_dies(tmp_path):
    """A worker whose app was killed (not shut down) ends by itself instead of keeping models."""
    import os
    import subprocess
    import sys
    script = tmp_path / "parent.py"
    script.write_text(
        "import sys, time\n"
        f"sys.path[:0] = [{str(worker_mod.__file__.rsplit('/stt/', 1)[0])!r}, {str(__file__.rsplit('/', 1)[0])!r}]\n"
        "from stt.worker import Worker\n"
        "from test_worker import _orphan_check\n"
        "w = Worker(target=_orphan_check)\n"
        "w._start()\n"
        "print(w._replies.get(timeout=30)[2], flush=True)\n"
        "time.sleep(60)\n")
    parent = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, text=True)
    child = int(parent.stdout.readline())
    parent.kill()  # SIGKILL: no clean-up runs in the parent
    parent.wait()
    for _ in range(50):
        time.sleep(0.2)
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            return
    os.kill(child, 9)
    raise AssertionError("the worker outlived its parent")


def test_stop_while_finding_speakers_kills_the_worker(monkeypatch):
    import threading
    monkeypatch.setattr(worker_mod, "STOP_GRACE_SEC", 0.5)
    w = Worker(target=_fake_serve)
    threading.Timer(1.0, w.cancel).start()  # Stop pressed a second in
    t = time.monotonic()
    assert w.speakers("x.wav", 0, None) is None
    assert time.monotonic() - t < 4 and not w.alive()


def test_doing_shows_while_running():
    with activity.doing("🌐 Downloading…") as now:
        assert activity.current() == ["🌐 Downloading…"]
        now("🤖 writing…")
        assert activity.current() == ["🤖 writing…"]
    assert activity.current() == []


def test_worker_activity_shows_in_the_app_until_the_job_ends():
    w = Worker(target=_fake_serve)
    try:
        seen = []
        for kind, _ in run(w, "loading"):
            if kind == "piece":
                seen.append(activity.current())
        assert seen[0] == ["🧠 Loading the test model…"]  # the download ended, the loading not yet
        assert activity.current() == []  # the job's end clears what the worker left shown
    finally:
        w.kill()
