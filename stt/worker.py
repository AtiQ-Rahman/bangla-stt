"""Run transcription in a separate worker process.

The speech models are big and CPU-bound. In their own process they can't freeze the web UI
(Python's GIL), Stop can end a job at once (by killing the worker if it doesn't stop by itself),
unloading frees all their memory, and running out of memory kills only the worker.
"""
import multiprocessing as mp
import os
import queue
import signal
import sys
import threading
import time
from typing import Iterator

from stt import activity


STOP_GRACE_SEC = 3.0  # how long a job may take to stop by itself before the worker is killed


class WorkerDied(RuntimeError):
    """The worker process ended in the middle of a job (most likely out of memory)."""


def _exit_with_parent():
    """End this process when the app that started it is gone, however it ended (killed,
    crashed), so the models never linger in memory."""
    parent = os.getppid()
    while True:
        time.sleep(2)
        if os.getppid() != parent:
            os._exit(0)


def _serve(requests, replies, cancel):
    """Worker process: run transcribe() for each request, streaming the pieces back."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl+C is for the app; it ends us
    threading.Thread(target=_exit_with_parent, daemon=True).start()
    from stt import activity
    from stt.audio import load_audio
    from stt.transcribe import transcribe

    job_id = None
    activity.forward = lambda key, text: replies.put(("activity", job_id, (key, text)))
    while True:
        job_id, args = requests.get()
        cancel.clear()
        try:
            wav = load_audio(args["path"], args["load_from"], args["end"])
            if args.get("task") == "speakers":
                from stt.speakers import diarize
                def progress(frac):
                    replies.put(("progress", job_id, frac))
                    return cancel.is_set()
                turns = diarize(wav, offset=args["load_from"], progress=progress)
                replies.put(("end", job_id, "cancelled") if cancel.is_set() else ("speakers", job_id, turns))
                replies.put(("end", job_id, None))
                continue
            replies.put(("start", job_id, len(wav)))
            stopped = None
            if len(wav):
                for done, total, seg in transcribe(
                        wav, args["model"], args["quantize"], args["use_lm"], offset=args["load_from"],
                        hotwords=args["words"], keep_from_sec=args["keep_from_sec"],
                        whisper=args.get("whisper")):
                    replies.put(("piece", job_id, (done, total, seg)))
                    if cancel.is_set():
                        stopped = "cancelled"
                        break
            replies.put(("end", job_id, stopped))
        except Exception as e:  # reported to the app, which shows it
            replies.put(("error", job_id, f"{type(e).__name__}: {e}"))


class Worker:
    """One worker process, started on first use; jobs run one at a time."""

    def __init__(self, target=_serve):
        self._target = target
        self._proc = None
        self._lock = threading.Lock()  # one job at a time
        self._ids = iter(range(1, 10**9))
        self.loaded: tuple[str, bool] | None = None  # (model, quantize) the worker holds

    def _start(self):
        ctx = mp.get_context("spawn")  # a clean interpreter: no copy of the web server
        self._requests, self._replies, self._cancel = ctx.Queue(), ctx.Queue(), ctx.Event()
        self._proc = ctx.Process(target=self._target, name="stt-worker", daemon=True,
                                 args=(self._requests, self._replies, self._cancel))
        # A spawned process first re-runs the main script (app.py builds the whole UI on import).
        # Hiding the main script from multiprocessing while starting makes the worker import only
        # what it needs (stt.worker); it is read only at start().
        main = sys.modules["__main__"]
        saved = {k: main.__dict__[k] for k in ("__file__", "__spec__") if k in main.__dict__}
        try:
            main.__dict__.pop("__file__", None)
            main.__spec__ = None
            self._proc.start()
        finally:
            main.__dict__.update(saved)
            if "__spec__" not in saved:
                del main.__spec__
        self.loaded = None

    def alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def is_loaded(self, model: str, quantize: bool) -> bool:
        return self.alive() and self.loaded in ((model, quantize), (model, None))

    def transcribe(self, path, load_from, end, model, quantize, use_lm, words, keep_from_sec,
                   whisper=None) -> Iterator[tuple[str, object]]:
        """Yields ("start", samples) once the audio is read, then ("piece", (done, total, segment))
        for each piece, and ("cancelled", None) last if cancel() stopped it."""
        yield from self._job({"path": str(path), "load_from": load_from, "end": end, "model": model,
                              "quantize": quantize, "use_lm": use_lm, "words": words,
                              "keep_from_sec": keep_from_sec, "whisper": whisper}, model, quantize)

    def speakers(self, path, start, end) -> list[tuple[float, float, int]] | None:
        """Who speaks when in path from start to end: (start, end, speaker) turns; None if Stop
        ended it."""
        for kind, value in self.speaker_steps(path, start, end):
            if kind != "progress":
                return value
        return []

    def speaker_steps(self, path, start, end) -> Iterator[tuple[str, object]]:
        """speakers() step by step: ("progress", fraction) as it goes, then ("speakers", turns),
        or ("cancelled", None) if Stop ended it."""
        yield from self._job({"task": "speakers", "path": str(path), "load_from": start or 0.0,
                              "end": end}, None, None)

    def _job(self, args, model, quantize) -> Iterator[tuple[str, object]]:
        """Send one request to the worker and yield its replies until it ends."""
        with self._lock:
            if not self.alive():
                self._start()
            job_id = next(self._ids)
            self._requests.put((job_id, args))
            finished, cancelled_at, shown = False, None, set()
            try:
                while True:
                    try:
                        kind, reply_id, value = self._replies.get(timeout=0.5)
                    except queue.Empty:
                        if not self.alive():
                            self._proc = None
                            raise WorkerDied("the transcription process stopped unexpectedly "
                                             "(probably out of memory)")
                        # Stop pressed during one long step (finding speakers, loading a model):
                        # if the worker doesn't end the job soon, end the worker.
                        if self._cancel.is_set():
                            cancelled_at = cancelled_at or time.monotonic()
                            if time.monotonic() - cancelled_at > STOP_GRACE_SEC:
                                self.kill()
                                finished = True
                                yield "cancelled", None
                                return
                        continue
                    if reply_id != job_id:  # left over from a job that was stopped
                        continue
                    if kind == "activity":  # e.g. loading or downloading a model
                        shown.add(value[0])
                        activity.show((self, value[0]), value[1])
                        continue
                    if kind == "end":
                        finished = True
                        if value == "cancelled":
                            yield "cancelled", None
                        return
                    if kind == "error":
                        finished = True
                        raise RuntimeError(value)
                    if kind == "piece" and model:
                        self.loaded = (model, quantize)
                    yield kind, value
            finally:
                for key in shown:  # the worker may have been killed while showing it
                    activity.show((self, key), None)
                if not finished:  # stopped from outside: end the job, or the worker
                    self._stop_job(job_id)

    def _stop_job(self, job_id):
        self._cancel.set()
        deadline = time.monotonic() + STOP_GRACE_SEC
        while time.monotonic() < deadline and self.alive():
            try:
                kind, reply_id, _ = self._replies.get(timeout=0.2)
            except queue.Empty:
                continue
            if reply_id == job_id and kind in ("end", "error"):
                return
        self.kill()  # still busy (e.g. loading a model): free it right away

    def cancel(self):
        """Ask the running job to stop (Stop button); the job's generator does the rest."""
        if self.alive():
            self._cancel.set()

    def kill(self):
        """End the worker and free all of its memory (unloading the models)."""
        if self._proc is not None:
            self._proc.kill()
            self._proc.join(timeout=5)
        self._proc = None
        self.loaded = None


WORKER = Worker()
