"""Fetch audio from YouTube or a local file and prepare it for transcription."""
import os
import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np

SAMPLE_RATE = 16000

# yt-dlp needs a JS runtime (deno) to solve YouTube's challenges; the pip
# package installs it next to the venv's python, so make sure it is on PATH.
_VENV_BIN = str(Path(sys.executable).parent)
if _VENV_BIN not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = _VENV_BIN + os.pathsep + os.environ.get("PATH", "")


def _youtube_dl(opts: dict):
    """yt_dlp.YoutubeDL that skips its "Python 3.10 is deprecated" notice (shown on every
    download; the fix is a newer Python, which this system doesn't have yet)."""
    import yt_dlp

    class QuietYoutubeDL(yt_dlp.YoutubeDL):
        def deprecated_feature(self, message):
            if "Python version" not in message:
                super().deprecated_feature(message)

    return QuietYoutubeDL(opts)


# Where browsers keep their profile; yt-dlp reads YouTube login cookies from there.
_BROWSER_PROFILES = [
    ("brave", "~/snap/brave/current/.config/BraveSoftware/Brave-Browser"),
    ("brave", "~/.config/BraveSoftware/Brave-Browser"),
    ("chrome", "~/.config/google-chrome"),
    ("chromium", "~/snap/chromium/common/chromium"),
    ("chromium", "~/.config/chromium"),
    ("firefox", "~/snap/firefox/common/.mozilla/firefox"),
    ("firefox", "~/.mozilla/firefox"),
]


def browser_profile() -> tuple[str, str] | None:
    """(browser, profile folder) of the first browser found on this PC, for its cookies."""
    for name, folder in _BROWSER_PROFILES:
        path = Path(folder).expanduser()
        if path.is_dir():
            return name, str(path)
    return None


def _download(url: str, opts: dict, cookies: str | None, browser: tuple[str, str] | None) -> tuple[Path, dict]:
    """Run yt-dlp; when YouTube asks to sign in (bot check), retry with the browser's cookies."""
    if cookies:
        opts["cookiefile"] = cookies
    try:
        with _youtube_dl(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return Path(ydl.prepare_filename(info)), info
    except Exception as e:
        if not browser or cookies or "Sign in" not in str(e):
            raise
    opts["cookiesfrombrowser"] = (browser[0], browser[1], None, None)
    with _youtube_dl(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return Path(ydl.prepare_filename(info)), info


def download_youtube(url: str, out_dir: Path, cookies: str | None = None,
                     browser: tuple[str, str] | None = None) -> tuple[Path, str]:
    """Download the audio track of a YouTube video into out_dir. Returns (path, title)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "audio.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "noplaylist": True,
    }
    path, info = _download(url, opts, cookies, browser)
    return path, info.get("title") or "youtube"


def download_youtube_video(url: str, out_dir: Path, cookies: str | None = None,
                           browser: tuple[str, str] | None = None) -> tuple[Path, str]:
    """Download a YouTube video (up to 720p, mp4) into out_dir. Returns (path, title)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/bv*[height<=720]+ba/b",
        "merge_output_format": "mp4",
        "outtmpl": str(out_dir / "%(title).80s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "noplaylist": True,
    }
    path, info = _download(url, opts, cookies, browser)
    return path.with_suffix(".mp4"), info.get("title") or "youtube"


def load_audio(path: str | Path, start: float | None = None, end: float | None = None) -> np.ndarray:
    """Decode any audio/video file to 16 kHz mono float32 via ffmpeg, optionally only start..end seconds."""
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error"]
    if start:
        cmd += ["-ss", str(start)]
    if end:
        cmd += ["-to", str(end)]
    cmd += ["-i", str(path), "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        msg = proc.stderr.decode(errors="replace").strip().splitlines()
        raise RuntimeError(msg[-1] if msg else "ffmpeg failed")
    # A read-only view of ffmpeg's output, not a copy: 2 hours of audio is ~440 MB.
    return np.frombuffer(proc.stdout, dtype=np.float32)


def duration(path: str | Path) -> float | None:
    """Length of an audio/video file in seconds, or None if ffprobe can't tell."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True)
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return None


def cut_clip(path: str | Path, out: Path, start: float | None = None, end: float | None = None) -> Path:
    """Save start..end seconds of path as a small mp3 (for listening to the selected part)."""
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y"]
    if start:
        cmd += ["-ss", str(start)]
    if end:
        cmd += ["-to", str(end)]
    cmd += ["-i", str(path), "-vn", "-ac", "1", "-b:a", "64k", str(out)]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        msg = proc.stderr.decode(errors="replace").strip().splitlines()
        raise RuntimeError(msg[-1] if msg else "ffmpeg failed")
    return out


def split_on_silence(wav: np.ndarray, max_sec: float = 20.0, min_sec: float = 0.5,
                     top_db: float = 35.0) -> list[tuple[int, int]]:
    """Find the voiced spans of the audio, merging neighbours up to max_sec.

    A single voiced stretch longer than max_sec (e.g. speech over background music)
    is returned whole; the transcriber walks through it in windows. Spans shorter
    than min_sec (clicks, breaths) are dropped.
    """
    max_len, min_len = int(max_sec * SAMPLE_RATE), int(min_sec * SAMPLE_RATE)
    voiced = librosa.effects.split(wav, top_db=top_db, frame_length=2048, hop_length=512)

    # Greedily merge neighbouring pieces until a chunk would exceed max_len.
    chunks: list[tuple[int, int]] = []
    for s, e in voiced:
        if chunks and e - chunks[-1][0] <= max_len:
            chunks[-1] = (chunks[-1][0], e)
        else:
            chunks.append((s, e))
    return [(s, e) for s, e in chunks if e - s >= min_len]
