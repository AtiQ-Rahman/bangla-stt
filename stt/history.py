"""Keep every transcription on disk (audio, text, subtitles, progress) so it survives restarts.

Each run is a folder under HISTORY_DIR holding audio.mp3, transcript.txt/.srt, corrected.txt
(after an AI fix) and meta.json with the settings, status and timed segments that Resume
needs.
"""
import json
import shutil
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from stt.transcribe import Segment, Word, to_srt, to_text

HISTORY_DIR = Path(__file__).resolve().parent.parent / "history"


def _safe(title: str) -> str:
    return "".join(c if c.isalnum() or c in " -_" else "_" for c in title).strip()[:60] or "audio"


def create(job: dict) -> Path:
    """A new entry for job; its audio is copied in, and job["audio"] then points at the copy."""
    entry = HISTORY_DIR / f"{datetime.now():%Y-%m-%d %H-%M-%S} {_safe(job['title'])}"
    entry.mkdir(parents=True)
    copy = entry / ("audio" + Path(job["audio"]).suffix)
    shutil.copyfile(job["audio"], copy)
    job["audio"] = str(copy)
    save(entry, job, [], job["start"] or 0.0, "running")
    return entry


def save(entry: Path, job: dict, segments: list[Segment], at: float, status: str):
    """Write progress: status is "running", "stopped" or "done"; at is where Resume continues."""
    meta = {"job": job, "status": status, "at": at, "updated": time.time(),
            "segments": [asdict(s) for s in segments]}
    tmp = entry / "meta.json.tmp"
    tmp.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    tmp.replace(entry / "meta.json")  # atomic: a crash never leaves half a file
    (entry / "transcript.txt").write_text(to_text(segments), encoding="utf-8")
    (entry / "transcript.srt").write_text(to_srt(segments), encoding="utf-8")


def load(entry: Path) -> dict:
    """meta.json with segments turned back into Segment objects."""
    meta = json.loads((entry / "meta.json").read_text(encoding="utf-8"))
    meta["segments"] = [Segment(s["start"], s["end"], s["text"], [Word(**w) for w in s["words"]])
                        for s in meta["segments"]]
    return meta


def save_text(entry: Path, name: str, text: str):
    """Keep an edited transcript or AI-fixed text (name: "transcript.txt" or "corrected.txt")."""
    (entry / name).write_text(text, encoding="utf-8")


def entries() -> list[tuple[str, str]]:
    """(label, folder) for every entry, newest first."""
    out = []
    for entry in sorted(HISTORY_DIR.glob("*/meta.json"), reverse=True):
        try:
            meta = json.loads(entry.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        job, status = meta["job"], meta["status"]
        # "running" in a saved file means the app stopped mid-way (closed or crashed).
        mark = {"done": "✅", "stopped": "⏸", "running": "⏸"}[status]
        name = entry.parent.name  # "2026-10-22 09-01-05 title"
        when = f"{name[:10]} {name[11:16].replace('-', ':')}"
        out.append((f"{mark} {when} · {job['title']} · {job['model'].split(' - ')[0]}", str(entry.parent)))
    return out


def delete(entry: Path):
    if entry.resolve().parent == HISTORY_DIR.resolve():  # never outside the history folder
        shutil.rmtree(entry, ignore_errors=True)
