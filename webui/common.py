"""Shared helpers: the temporary folder, RAM checks, progress bars, files and times."""
import atexit
import os
import shutil
import tempfile
import time
import types
from pathlib import Path

import gradio as gr

from stt.audio import browser_profile
from stt.transcribe import LANGUAGE_MODELS, LM_RAM_GB, RAM_NEEDED_GB
from stt.worker import WORKER


# All temporary files of this app live in a per-process folder, emptied per
# transcription and removed on exit. Folders left by crashed runs (their process
# is gone) are removed on start. Results worth keeping go to stt.history instead.
WORK_DIR = Path(tempfile.gettempdir()) / f"bangla-stt-{os.getpid()}"


for old in Path(tempfile.gettempdir()).glob("bangla-stt-*"):
    pid = old.name.removeprefix("bangla-stt-")
    if pid.isdigit() and not Path(f"/proc/{pid}").exists():
        shutil.rmtree(old, ignore_errors=True)


atexit.register(shutil.rmtree, WORK_DIR, ignore_errors=True)


IDLE_UNLOAD_SEC = 10 * 60  # free the models' RAM after this long without use


LLM_RAM_GB = 4.0


BROWSER = browser_profile()  # whose YouTube login to use when YouTube asks "are you a bot?"


VIDEO_EXT = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".flv", ".wmv", ".3gp"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".opus", ".flac", ".wma", ".amr"}


# Shared run-time state: transcriptions in progress (the idle unloader leaves the models
# alone while > 0) and when the models were last used.
runtime = types.SimpleNamespace(active=0, last_used=time.monotonic())

# History folder -> progress of a transcription running right now (any page, Batch or not), so a
# page that opens it can follow along: {"segments", "frac", "audio_sec", "loading"}.
LIVE: dict[str, dict] = {}


def mem_gb() -> tuple[float, float]:
    """(available, total) RAM in GB."""
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, value = line.split(":")
            info[key] = int(value.split()[0]) / 1024 / 1024
    return info["MemAvailable"], info["MemTotal"]


RAM_LEVELS = {"ok": 0, "low": 1, "critical": 2}


def ram_level(avail: float) -> str:
    return "ok" if avail >= 4 else "low" if avail >= 2 else "critical"


def ram_status() -> str:
    avail, total = mem_gb()
    icon = {"ok": "🟢", "low": "🟡", "critical": "🔴"}[ram_level(avail)]
    return f"{icon} Free RAM **{avail:.1f} GB** / {total:.0f} GB"


def ram_tick(prev_level: str):
    """Timer: refresh the RAM badge, and warn (with a sound when critical) as RAM runs out."""
    avail, _ = mem_gb()
    level, alert = ram_level(avail), ""
    if RAM_LEVELS[level] > RAM_LEVELS.get(prev_level, 0):
        if level == "critical":
            gr.Warning(f"RAM almost full: only {avail:.1f} GB free. Close other apps or browser tabs, "
                       "or press Stop, before the computer freezes.", duration=20)
            alert = f"beep {time.time()}"  # a new value each time; BEEP_JS plays it
        else:
            gr.Warning(f"RAM is getting low: {avail:.1f} GB free.", duration=8)
    return ram_status(), level, alert


def idle_unloader():
    """Background thread: unload the models after IDLE_UNLOAD_SEC without a transcription."""
    while True:
        time.sleep(60)
        if not runtime.active and time.monotonic() - runtime.last_used > IDLE_UNLOAD_SEC:
            WORKER.kill()  # ends the worker process: all of its memory is freed


def check_ram(model_name: str, quantize: bool, use_lm: bool):
    if WORKER.is_loaded(model_name, quantize):
        return
    avail, _ = mem_gb()
    need = RAM_NEEDED_GB[(model_name, quantize)]
    if avail < need:
        raise gr.Error(
            f"Not enough free RAM: this model needs about {need:.1f} GB, only {avail:.1f} GB is free. "
            "Close other apps/browser tabs, or pick the smaller model in Settings.")
    if use_lm and avail < need + LM_RAM_GB:
        gr.Warning("RAM is low: the Bangla language model may make transcription slow. "
                   "Turn it off in Settings if it crawls.")


def parse_time(value: str) -> float | None:
    """'90', '1:30' or '0:01:30' -> seconds; blank -> None."""
    value = (value or "").strip()
    if not value:
        return None
    try:
        sec = 0.0
        for part in value.split(":"):
            sec = sec * 60 + float(part)
        return sec
    except ValueError:
        raise gr.Error(f"Can't read time '{value}'. Use seconds (90) or mm:ss (1:30).")


def fmt_dur(sec: float) -> str:
    h, rest = divmod(int(round(sec)), 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02}:{s:02}" if h else f"{m}:{s:02}"


def safe_stem(title: str) -> str:
    return "".join(c if c.isalnum() or c in " -_" else "_" for c in title).strip() or "transcript"


def language_of(model_name: str) -> str:
    return "English" if model_name in LANGUAGE_MODELS["English"] else "Bangla"


def bar_html(text: str, fraction: float | None = None, done: bool = False) -> str:
    """A progress bar; fraction None = busy without a known amount."""
    cls = "pbar done" if done else "pbar" if fraction is not None else "pbar busy"
    width = 100 if fraction is None else round(fraction * 100)
    return f'<div class="{cls}"><div class="fill" style="width:{width}%"></div><span>{text}</span></div>'


def bar(text: str, fraction: float | None = None, done: bool = False) -> gr.HTML:
    return gr.HTML(bar_html(text, fraction, done), visible=True)


def time_left(first: tuple[float, float] | None, frac: float) -> str:
    """' · about 3:20 left', estimated from the progress since `first` (time, fraction)."""
    now = time.monotonic()
    if not first or not first[1] < frac < 1 or now - first[0] < 5:
        return ""
    return f" · about {fmt_dur((now - first[0]) / (frac - first[1]) * (1 - frac))} left"


def file_button(path: Path) -> gr.DownloadButton:
    return gr.DownloadButton(value=str(path), interactive=True)


def no_file() -> gr.DownloadButton:
    return gr.DownloadButton(value=None, interactive=False)


def out_file(title: str, suffix: str) -> Path:
    """Where a download for the current result is written (named after the audio)."""
    run_dir = WORK_DIR / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir / f"{safe_stem(title or 'transcript')}{suffix}"


def youtube_error(e: Exception) -> gr.Error:
    hint = ("" if BROWSER else
            " If YouTube asks you to sign in or flags you as a bot, upload a cookies.txt file "
            "exported from your browser (Settings).")
    return gr.Error(f"Download failed: {e}.{hint}")


def fresh_media_path(stem: str) -> Path:
    """A new file in the media folder, after removing what an earlier fetch left there."""
    media = WORK_DIR / "media"
    shutil.rmtree(media, ignore_errors=True)
    media.mkdir(parents=True)
    return media / f"{safe_stem(stem)}.mp3"


def cookie_args(cookies_file, use_browser):
    return cookies_file, (BROWSER if use_browser else None)
