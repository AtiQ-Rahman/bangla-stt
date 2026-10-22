"""Getting the audio: YouTube, video/audio files, trimming, language detection."""
import os
import shutil
import time
from pathlib import Path

import gradio as gr

from stt.audio import cut_clip, download_youtube, download_youtube_video, duration
from stt.langid import detect
from webui.common import cookie_args, fmt_dur, fresh_media_path, parse_time, WORK_DIR, youtube_error


def get_youtube(url, cookies_file, use_browser, progress=gr.Progress()):
    """Download a YouTube video's audio into the Audio player, ready to listen to and trim."""
    url = (url or "").strip()
    if not url:
        raise gr.Error("Paste a YouTube link first.")
    progress(0, desc="Downloading audio from YouTube…")
    tmp = WORK_DIR / "download"
    try:
        path, title = download_youtube(url, tmp, *cookie_args(cookies_file, use_browser))
        progress(0.8, desc="Converting…")
        audio = cut_clip(path, fresh_media_path(title))
    except Exception as e:
        raise youtube_error(e)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return (str(audio), title, f"✅ Got the audio of “{title}”. Listen, trim if you like, then Transcribe.",
            None, url)


def get_youtube_video(url, cookies_file, use_browser, progress=gr.Progress()):
    """Download the YouTube video itself (to save), and put its sound into the Audio player."""
    url = (url or "").strip()
    if not url:
        raise gr.Error("Paste a YouTube link first.")
    progress(0, desc="Downloading the video (up to 720p)…")
    out = WORK_DIR / "video"
    shutil.rmtree(out, ignore_errors=True)  # only the latest video is kept
    try:
        path, title = download_youtube_video(url, out, *cookie_args(cookies_file, use_browser))
        progress(0.9, desc="Extracting the audio…")
        audio = cut_clip(path, fresh_media_path(title))
    except Exception as e:
        raise youtube_error(e)
    size = path.stat().st_size / 1024 / 1024
    save = gr.DownloadButton(f"💾 Save the video ({size:.0f} MB)", value=str(path), visible=True)
    return (str(audio), title, f"✅ Got the video “{title}”. Save it below; its audio is ready too.",
            save, str(path), url)


def get_file_audio(upload, is_video, progress=gr.Progress()):
    """Put the sound of an uploaded video or audio file into the Audio player (as mp3)."""
    if not upload:
        return (gr.skip(),) * 5
    progress(0, desc="Reading the file…")
    title = Path(upload).stem
    try:
        audio = cut_clip(upload, fresh_media_path(title))
    except RuntimeError as e:
        raise gr.Error(f"Could not read the sound of this file: {e}")
    return (str(audio), title, f"✅ Loaded “{title}”. Listen, trim if you like, then Transcribe.",
            upload if is_video else None, None)


def get_video_file(upload, progress=gr.Progress()):
    return get_file_audio(upload, True, progress)


def get_audio_file(upload, progress=gr.Progress()):
    return get_file_audio(upload, False, progress)


def selected_part(audio, start, end):
    """Cut the From..To part of the audio so it can be listened to before transcribing."""
    start, end = parse_time(start), parse_time(end)
    if not audio or not (start or end):
        return gr.Audio(value=None, visible=False)
    length = duration(audio)
    if length and start and start >= length:
        raise gr.Error(f"'From' is after the end of the audio ({fmt_dur(length)}).")
    if start and end and end <= start:
        raise gr.Error("'To' must be after 'From'.")
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    try:
        clip = cut_clip(audio, WORK_DIR / f"selected-{time.monotonic_ns()}.mp3", start, end)
    except RuntimeError as e:
        raise gr.Error(f"Could not cut the audio: {e}")
    for old in WORK_DIR.glob("selected-*.mp3"):  # keep only the newest preview
        if old != clip:
            old.unlink(missing_ok=True)
    label = f"Selected part: {fmt_dur(start or 0)} – {fmt_dur(end or length or 0)}"
    return gr.Audio(value=str(clip), visible=True, label=label)


def detect_language(path, current, text):
    """Set the Language choice to what the loaded audio sounds like. When a finished result was
    just opened (there is text), its own message stays and the language is already right."""
    if not path or (text or "").strip():
        return gr.skip(), gr.skip()
    try:
        language, confidence = detect(path)
    except Exception:
        return gr.skip(), gr.skip()  # detection is a convenience; never block on it
    if confidence < 0.8:
        return gr.skip(), gr.skip()
    note = (f"🌐 This sounds like **{language}** ({confidence:.0%} sure)"
            + ("." if language == current else f", so the language is now set to {language}.")
            + " Listen, trim if you like, then Transcribe.")
    return language, note


def show_length(path):
    """Put the audio's length into the 'To' box as a hint."""
    length = duration(path) if path else None
    return gr.Textbox(placeholder=f"end ({fmt_dur(length)})" if length else "end")


def audio_changed(audio, partial):
    """New or trimmed audio: forget a stopped run unless it is the same file (e.g. opened from
    History; Gradio hands over a copy, so the size is compared)."""
    same = (audio and partial and Path(partial["job"]["audio"]).exists()
            and os.path.getsize(audio) == os.path.getsize(partial["job"]["audio"]))
    return (gr.skip(),) * 3 if same else (None, gr.Button(visible=False), gr.Button(visible=False))
