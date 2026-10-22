"""Folder watch (Batch tab): new audio/video files in a folder join the batch queue and are
transcribed into History, while the page is open."""
import json
import time
from pathlib import Path

import gradio as gr

from webui import settings
from webui.batch import _queue_ids, queue_views
from webui.common import AUDIO_EXT, VIDEO_EXT

SEEN = settings.FILE.parent / "watched.json"  # files already queued: "path|size|mtime"
SETTLE_SEC = 5  # a file still being copied keeps changing; wait until it has been quiet this long


def _seen() -> set[str]:
    try:
        return set(json.loads(SEEN.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        return set()


def _key(path: Path) -> str:
    st = path.stat()
    return f"{path}|{st.st_size}|{st.st_mtime_ns}"


def new_files(folder: Path) -> list[Path]:
    """Audio/video files in the folder (not its subfolders) not queued before, oldest first;
    a file is new again when it is replaced."""
    seen, now, out = _seen(), time.time(), []
    for path in folder.iterdir():
        if (path.is_file() and not path.name.startswith(".")
                and path.suffix.lower() in AUDIO_EXT | VIDEO_EXT):
            st = path.stat()
            if st.st_size and now - st.st_mtime > SETTLE_SEC and _key(path) not in seen:
                out.append(path)
    return sorted(out, key=lambda p: p.stat().st_mtime)


def mark_seen(paths: list[Path]):
    seen = _seen() | {_key(p) for p in paths}
    SEEN.parent.mkdir(parents=True, exist_ok=True)
    SEEN.write_text(json.dumps(sorted(seen), ensure_ascii=False, indent=0), encoding="utf-8")


def toggle(on, folder):
    """Watch on/off: check the folder, remember it, and start/stop the timer."""
    if not on:
        return gr.Timer(active=False), "Not watching.", gr.skip()
    path = Path(folder or "").expanduser()
    if not folder or not path.is_dir():
        if folder and path.parent.is_dir():
            path.mkdir()  # e.g. the suggested inbox folder
        else:
            gr.Warning("That folder doesn't exist.")
            return gr.Timer(active=False), "Not watching.", False
    settings.remember("watch_folder", str(path))
    return (gr.Timer(active=True),
            f"👀 Watching **{path}** — audio and video files put there are transcribed into "
            "🕘 History, one after another. Keep this page open.", gr.skip())


def tick(on, folder, queue, busy):
    """Every few seconds while watching: queue the new files; returns a start signal (a new
    value makes the page press ▶ Start) when files were added and nothing is running."""
    path = Path(folder or "").expanduser()
    if not on or busy or not path.is_dir():
        return (gr.skip(),) * 4
    files = [p for p in new_files(path) if str(p) not in {item["src"] for item in queue}]
    if not files:  # (not a restart either: a batch the user stopped stays stopped)
        return (gr.skip(),) * 4
    mark_seen(files)
    queue = queue + [{"id": next(_queue_ids), "kind": "file", "src": str(p), "name": p.name,
                      "status": "waiting"} for p in files]
    return (queue,) + queue_views(queue) + (str(time.monotonic_ns()),)
