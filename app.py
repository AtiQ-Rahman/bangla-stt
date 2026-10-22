"""Gradio web UI: paste a YouTube link or upload audio/video, get a Bangla or English transcript."""
import atexit
import html
import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
import traceback
from pathlib import Path

import gradio as gr

from stt import history
from stt.audio import (browser_profile, cut_clip, download_youtube, download_youtube_video, duration,
                       load_audio)
from stt.correct import DEFAULT_LLM, correct, list_models
from stt.correct import is_loaded as llm_is_loaded
from stt.export import bangla_digits, burn_subtitles, to_vtt, write_docx
from stt.langid import detect
from stt.retime import retime
from stt.transcribe import (DEFAULT_MODEL, LANGUAGE_MODELS, LM_RAM_GB, RAM_NEEDED_GB, Segment, Word,
                            is_loaded, to_srt, to_text, transcribe, unload)

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

# {lang} / {other} are filled in by the browser from the chosen language.
CLAUDE_TASKS = {
    "Fix spelling & punctuation": (
        "Below is a {lang} transcript produced by a speech-recognition model. It may have spelling "
        "mistakes, wrongly split or joined words, and missing punctuation. Rewrite it in correct "
        "standard {lang} with proper punctuation. Do not add, remove or summarise anything."),
    "Summarise (bullet points)": "Summarise this {lang} transcript in {lang} as short bullet points.",
    "Translate (Bangla ↔ English)": (
        "Translate this {lang} transcript into natural {other}. It came from speech recognition, "
        "so silently fix obvious recognition mistakes."),
    "Custom (write below)": "",
}

_last_used = time.monotonic()
_active = 0  # transcriptions in progress; the idle unloader leaves the models alone while > 0


# ---- RAM ------------------------------------------------------------------------------------

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
        if not _active and time.monotonic() - _last_used > IDLE_UNLOAD_SEC:
            unload()


threading.Thread(target=idle_unloader, daemon=True).start()


def check_ram(model_name: str, quantize: bool, use_lm: bool):
    if is_loaded(model_name, quantize):
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


# ---- small helpers --------------------------------------------------------------------------

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


# ---- getting the audio ----------------------------------------------------------------------

def fresh_media_path(stem: str) -> Path:
    """A new file in the media folder, after removing what an earlier fetch left there."""
    media = WORK_DIR / "media"
    shutil.rmtree(media, ignore_errors=True)
    media.mkdir(parents=True)
    return media / f"{safe_stem(stem)}.mp3"


def cookie_args(cookies_file, use_browser):
    return cookies_file, (BROWSER if use_browser else None)


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
            None)


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
            save, str(path))


def get_file_audio(upload, is_video, progress=gr.Progress()):
    """Put the sound of an uploaded video or audio file into the Audio player (as mp3)."""
    if not upload:
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()
    progress(0, desc="Reading the file…")
    title = Path(upload).stem
    try:
        audio = cut_clip(upload, fresh_media_path(title))
    except RuntimeError as e:
        raise gr.Error(f"Could not read the sound of this file: {e}")
    return (str(audio), title, f"✅ Loaded “{title}”. Listen, trim if you like, then Transcribe.",
            upload if is_video else None)


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


# ---- transcribing ---------------------------------------------------------------------------

def _work(job: dict, entry: Path, prior: list[Segment], at: float):
    """Transcribe job["audio"] from `at` seconds to job["end"], after the segments in prior,
    saving progress to the history entry (every 10 s, and when it ends or is stopped).

    Yields (segments, fraction done, resume point, length of the range, model was loaded):
    once before the model loads, then after every piece.
    """
    global _last_used, _active
    begin = job["start"] or 0.0
    # Resuming: hear 3 s before the last word too, as context; words only come after `at`.
    load_from = max(begin, at - 3.0) if prior else at
    try:
        wav = load_audio(job["audio"], load_from, job["end"])
    except RuntimeError as e:
        raise gr.Error(f"Could not read the audio: {e}")
    if not len(wav):
        raise gr.Error("No audio left in the selected range.")
    was_loaded = is_loaded(job["model"], job["quantize"])
    before, remaining = load_from - begin, len(wav) / 16000
    audio_sec = before + remaining
    segments, resume_at, finished, saved = list(prior), at, False, time.monotonic()
    digits = job.get("bn_digits") and language_of(job["model"]) == "Bangla"
    _active += 1
    try:
        yield segments, before / audio_sec, resume_at, audio_sec, was_loaded
        for done, total, seg in transcribe(wav, job["model"], job["quantize"], job["use_lm"],
                                           offset=load_from, hotwords=job["words"],
                                           keep_from_sec=at - load_from):
            if seg:
                if digits:
                    seg = Segment(seg.start, seg.end, bangla_digits(seg.text),
                                  [Word(w.start, w.end, bangla_digits(w.text)) for w in seg.words])
                segments.append(seg)
                resume_at = max(resume_at, seg.end)
            if time.monotonic() - saved > 10:
                history.save(entry, job, segments, resume_at, "running")
                saved = time.monotonic()
            yield segments, (before + done / total * remaining) / audio_sec, resume_at, audio_sec, was_loaded
        finished = True
    finally:
        _active -= 1
        _last_used = time.monotonic()
        history.save(entry, job, segments, resume_at, "done" if finished else "stopped")


# run() outputs: transcript, txt, srt, srt view, status, clip, progress bar, partial, segs, entry
def stage(message: str) -> tuple:
    """A run() update that only changes the status line and the progress bar."""
    return (gr.skip(),) * 4 + (message, gr.skip(), bar(message)) + (gr.skip(),) * 3


def make_job(audio, title, model_name, quantize, use_lm, hotwords, start, end, digits):
    start, end = parse_time(start), parse_time(end)
    if start and end and end <= start:
        raise gr.Error("End time must be after start time.")
    words = [w.strip() for w in (hotwords or "").replace("\n", ",").split(",") if w.strip()]
    use_lm = use_lm and language_of(model_name) == "Bangla"  # Whisper has its own
    return {"audio": audio, "title": title or Path(audio).stem, "model": model_name,
            "quantize": quantize, "use_lm": use_lm, "words": words, "start": start, "end": end,
            "bn_digits": bool(digits)}


def run(audio, title, model_name, quantize, use_lm, hotwords, start, end, digits, video):
    if not audio:
        raise gr.Error("Add some audio first: get it from YouTube, upload a video or audio file, or record.")
    job = make_job(audio, title, model_name, quantize, use_lm, hotwords, start, end, digits)
    if video:
        job["video"] = video  # so "video with subtitles" also works when reopened from History
    check_ram(job["model"], job["quantize"], job["use_lm"])

    # Clear the previous result straight away.
    yield ("", no_file(), no_file(), "", "⏳ Starting…", gr.Audio(), bar("⏳ Checking the language…"),
           None, None, None)
    chosen = language_of(model_name)
    try:
        heard, confidence = detect(audio, job["start"], job["end"])
    except Exception:
        heard, confidence = chosen, 0.0
    if heard != chosen and confidence >= 0.8:
        raise gr.Error(f"This sounds like {heard} ({confidence:.0%} sure), but {chosen} is selected. "
                       f"Choose {heard} under 'Language of the speech' and press Transcribe again.")
    shutil.rmtree(WORK_DIR / "run", ignore_errors=True)
    entry = history.create(job)  # copies the audio in; job["audio"] now points at that copy
    yield from _job(job, entry, [], job["start"] or 0.0)


def resume(partial):
    """Continue a stopped transcription from its last word, with the same audio and settings."""
    if not partial:
        raise gr.Error("Nothing to resume: press Transcribe.")
    job = partial["job"]
    check_ram(job["model"], job["quantize"], job["use_lm"])
    yield stage(f"⏳ Resuming at {fmt_dur(partial['at'])}…")
    yield from _job(job, Path(partial["entry"]), partial["segments"], partial["at"])


def _job(job, entry, prior, at):
    """run()/resume() progress for the UI around _work()."""
    txt_path, srt_path = out_file(job["title"], ".txt"), out_file(job["title"], ".srt")
    yield stage("⏳ Reading the audio…")
    clip = gr.skip()
    if not prior and (job["start"] or job["end"]):  # let the user hear what is transcribed
        try:
            clip = gr.Audio(value=str(cut_clip(job["audio"], out_file(job["title"], " part.mp3"),
                                               job["start"], job["end"])),
                            visible=True, label="Transcribed part")
        except RuntimeError:
            pass
    t0 = first = None
    segments, audio_sec, was_loaded = list(prior), 0.0, True
    for segments, frac, resume_at, audio_sec, was_loaded in _work(job, entry, prior, at):
        if t0 is None:  # before the model loads
            t0 = time.monotonic()
            if not was_loaded:
                yield stage("⏳ Loading the model into memory — the first time this takes up to a minute…")
            continue
        first = first or (time.monotonic(), frac)
        txt_path.write_text(to_text(segments), encoding="utf-8")
        srt = to_srt(segments)
        srt_path.write_text(srt, encoding="utf-8")
        progress_text = (f"⏳ Transcribing… {frac:.0%} · {fmt_dur(frac * audio_sec)} of "
                         f"{fmt_dur(audio_sec)}{time_left(first, frac)}")
        partial = {"job": job, "entry": str(entry), "segments": segments, "at": resume_at}
        yield (to_text(segments), file_button(txt_path), file_button(srt_path), srt,
               "⏳ Transcribing — press Stop any time; you can Resume later", clip,
               bar(progress_text, frac), partial, segments, str(entry))
    took = time.monotonic() - (t0 or time.monotonic())
    done_text = (f"✅ Done: {fmt_dur(audio_sec)} of audio"
                 + (f" (this part took {fmt_dur(took)})" if prior else f" in {fmt_dur(took)}")
                 + ("" if was_loaded else ", incl. loading the model") + " · saved in History")
    txt_path.write_text(to_text(segments), encoding="utf-8")
    srt_path.write_text(to_srt(segments), encoding="utf-8")
    yield (to_text(segments), file_button(txt_path), file_button(srt_path), to_srt(segments),
           done_text, clip, bar(done_text, 1, done=True), None, segments, str(entry))


# ---- edits: downloads follow what is in the text boxes --------------------------------------

def edited_text(text, title, entry, name):
    """The user typed in a text box: its .txt download (and History copy) follow."""
    path = out_file(title, " (fixed).txt" if name == "corrected.txt" else ".txt")
    path.write_text(text or "", encoding="utf-8")
    if entry:
        history.save_text(Path(entry), name, text or "")
    return file_button(path)


def edited_subtitles(text, segs, title, suffix):
    """When the user leaves a text box: re-time its text onto the transcript for new subtitles."""
    if not segs or not (text or "").strip():
        return gr.skip(), gr.skip()
    srt = to_srt(retime(text, segs))
    path = out_file(title, suffix)
    path.write_text(srt, encoding="utf-8")
    return file_button(path), srt


# ---- local AI fix ---------------------------------------------------------------------------

def run_correct(text, llm, language, segs, title, entry, fix_partial, resuming=False):
    """Fix the transcript with the local LLM, piece by piece; Stop keeps the finished pieces."""
    if not (text or "").strip():
        raise gr.Error("Transcribe something first.")
    if not list_models():
        raise gr.Error("Ollama is not running. Start it with `docker start ollama` (see README).")
    if _active:
        raise gr.Error("Wait for the transcription to finish (or press Stop) first.")
    if resuming and fix_partial:
        text, llm, language = fix_partial["text"], fix_partial["llm"], fix_partial["language"]
        done, reverted = fix_partial["done"], fix_partial["reverted"]
    else:
        done, reverted = [], 0
    unload()  # the speech models and the LLM together don't fit in 16 GB RAM
    avail, _ = mem_gb()
    if avail < LLM_RAM_GB and not llm_is_loaded(llm):  # e.g. resuming: it is still loaded
        raise gr.Error(f"Not enough free RAM for the LLM ({avail:.1f} GB free, ~{LLM_RAM_GB:.0f} GB needed).")
    yield gr.skip(), no_file(), no_file(), bar("⏳ Loading the AI model…"), gr.skip()
    fixed, first = " ".join(done), None
    try:
        for fixed, reverted, done, total in correct(text, llm, language, done=done, reverted=reverted):
            frac = len(done) / total
            first = first or (time.monotonic(), frac)
            state = {"text": text, "llm": llm, "language": language, "done": done, "reverted": reverted}
            yield (fixed, gr.skip(), gr.skip(),
                   bar(f"⏳ Fixing… part {min(len(done) + 1, total)} of {total}{time_left(first, frac)}", frac),
                   state)
    except Exception as e:
        raise gr.Error(f"LLM failed: {e}")
    txt = out_file(title, " (fixed).txt")
    txt.write_text(fixed, encoding="utf-8")
    srt = no_file()
    if segs:
        srt_path = out_file(title, " (fixed).srt")
        srt_path.write_text(to_srt(retime(fixed, segs)), encoding="utf-8")
        srt = file_button(srt_path)
    if entry:
        history.save_text(Path(entry), "corrected.txt", fixed)
    note = (f" · undid {reverted} change(s) that added, dropped or reworded words" if reverted else "")
    yield fixed, file_button(txt), srt, bar(f"✅ Fixed{note}", 1, done=True), None


def resume_correct(text, llm, language, segs, title, entry, fix_partial):
    yield from run_correct(text, llm, language, segs, title, entry, fix_partial, resuming=True)


def refresh_llms():
    models = list_models()
    value = DEFAULT_LLM if DEFAULT_LLM in models else (models[0] if models else None)
    return gr.Dropdown(choices=models, value=value)


# ---- export ---------------------------------------------------------------------------------

def export_files(which, transcript_text, fixed_text, segs, title):
    text = fixed_text if which.startswith("AI") else transcript_text
    if not (text or "").strip():
        raise gr.Error("There is no text to export yet.")
    suffix = " (fixed)" if which.startswith("AI") else ""
    files = [str(write_docx(text, out_file(title, f"{suffix}.docx"), title or "Transcript"))]
    if segs:
        vtt = out_file(title, f"{suffix}.vtt")
        vtt.write_text(to_vtt(retime(text, segs)), encoding="utf-8")
        files.insert(0, str(vtt))
    return gr.File(value=files, visible=True)


def export_video(which, transcript_text, fixed_text, segs, title, video_src):
    """Burn the subtitles into the video the audio came from."""
    if not video_src or not Path(video_src).exists():
        raise gr.Error("There is no video for this audio: use 🎬 Video + audio (YouTube) or "
                       "🎬 Video file first.")
    text = fixed_text if which.startswith("AI") else transcript_text
    if not segs or not (text or "").strip():
        raise gr.Error("Transcribe the video first.")
    work = WORK_DIR / "burn"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    video = work / ("video" + Path(video_src).suffix)
    video.symlink_to(Path(video_src).resolve())
    srt = work / "subs.srt"
    srt.write_text(to_srt(retime(text, segs)), encoding="utf-8")
    out = out_file(title, " (subtitled).mp4")
    first = None
    try:
        for frac in burn_subtitles(video, srt, out):
            first = first or (time.monotonic(), frac)
            yield bar(f"⏳ Making the video… {frac:.0%}{time_left(first, frac)}", frac), gr.skip()
    except RuntimeError as e:
        raise gr.Error(f"Could not make the video: {e}")
    size = out.stat().st_size / 1024 / 1024
    yield (bar("✅ Video ready", 1, done=True),
           gr.DownloadButton(f"💾 Save video with subtitles ({size:.0f} MB)", value=str(out), visible=True))


# ---- batch ----------------------------------------------------------------------------------

_queue_ids = iter(range(1, 10**9))


def queue_rows(queue: list[dict]) -> list[list]:
    """The batch queue as table rows; the last column says a row can be opened."""
    return [[n, f"{'🔗' if item['kind'] == 'link' else '🎞️'} {item['name']}", item.get("language", ""),
             item["status"], "📂 Open"] for n, item in enumerate(queue, 1)]


def queue_views(queue):
    """The table, plus the item picker for Remove (only items not yet done make sense there)."""
    choices = [(f"{n}. {item['name'][:60]}", item["id"]) for n, item in enumerate(queue, 1)]
    return queue_rows(queue), gr.Dropdown(choices=choices, value=[])


def add_links(text, queue):
    """Add every link in the box (one or many, any separators) and empty the box."""
    links = [w for w in (text or "").replace(",", " ").split() if w.startswith(("http://", "https://"))]
    if not links:
        raise gr.Error("Paste a YouTube link (starting with https://) first.")
    queue = queue + [{"id": next(_queue_ids), "kind": "link", "src": u, "name": u, "status": "waiting"}
                     for u in links]
    return ("", queue) + queue_views(queue)


def add_files(files, queue):
    queue = queue + [{"id": next(_queue_ids), "kind": "file", "src": f, "name": Path(f).name,
                      "status": "waiting"} for f in files or []]
    return (queue,) + queue_views(queue)


def remove_items(ids, queue):
    queue = [item for item in queue if item["id"] not in set(ids or [])]
    return (queue,) + queue_views(queue)


def clear_queue():
    return ([],) + queue_views([]) + (gr.HTML(visible=False),)


def run_batch(queue, lang_choice, quantize, use_lm, digits, cookies_file, use_browser):
    """Transcribe the queued items that aren't done yet, one after another; each goes to History."""
    queue = [dict(item) for item in queue]  # this run's copy; yielded back to the page
    todo = [item for item in queue if item["status"] != "✅ done"]
    if not todo:
        raise gr.Error("Nothing to do: add YouTube links or files to the queue first.")
    work = WORK_DIR / "batch"
    first = None
    for i, item in enumerate(todo):
        def show(status, frac=0.0):
            item["status"] = status
            overall = (i + frac) / len(todo)
            return (queue, queue_rows(queue),
                    bar(f"⏳ Item {i + 1} of {len(todo)}{time_left(first, overall)}", overall))

        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        try:
            yield show("getting the audio…")
            if item["kind"] == "link":
                path, title = download_youtube(item["src"], work / "download",
                                               *cookie_args(cookies_file, use_browser))
            else:
                path, title = Path(item["src"]), Path(item["src"]).stem
            item["name"] = title
            audio = cut_clip(path, work / "audio.mp3")
            language = lang_choice if lang_choice != "Auto" else detect(audio)[0]
            item["language"] = language
            model = LANGUAGE_MODELS[language][0]
            job = make_job(str(audio), title, model, quantize, use_lm, "", "", "", digits)
            if item["kind"] == "file" and Path(item["src"]).suffix.lower() in VIDEO_EXT:
                job["video"] = item["src"]
            check_ram(model, quantize, job["use_lm"])
            entry = history.create(job)
            item["entry"] = str(entry)
            yield show("loading the model…")
            for _, frac, _, _, _ in _work(job, entry, [], 0.0):
                first = first or (time.monotonic(), (i + frac) / len(todo))
                yield show(f"transcribing {frac:.0%}", frac)
            item["status"] = "✅ done"
        except Exception as e:  # one bad item shouldn't stop the rest
            item["status"] = f"❌ {e}"[:120]
        yield queue, queue_rows(queue), bar(f"⏳ Item {i + 1} of {len(todo)}", (i + 1) / len(todo))
    shutil.rmtree(work, ignore_errors=True)
    ok = sum(item["status"] == "✅ done" for item in todo)
    yield (queue, queue_rows(queue),
           bar(f"✅ Finished: {ok} of {len(todo)} done · open them in 🕘 History", 1, done=True))


# ---- history --------------------------------------------------------------------------------

def history_choices():
    return gr.Dropdown(choices=history.entries(), value=None)


def history_preview(folder):
    if not folder:
        return ""
    meta = history.load(Path(folder))
    job, segs = meta["job"], meta["segments"]
    text = (Path(folder) / "transcript.txt").read_text(encoding="utf-8")
    state = {"done": "✅ finished", "stopped": f"⏸ stopped at {fmt_dur(meta['at'])}",
             "running": f"⏸ interrupted at {fmt_dur(meta['at'])}"}[meta["status"]]
    fixed = " · has AI-fixed text" if (Path(folder) / "corrected.txt").exists() else ""
    return (f"**{job['title']}** · {job['model']} · {state} · {len(segs)} pieces{fixed}\n\n"
            f"> {text[:400]}{'…' if len(text) > 400 else ''}")


def history_open(folder, still_running=False):
    """Load a saved run into the Transcribe tab; an unfinished one can be Resumed (unless the
    batch is still transcribing it). Returns the OPEN_OUTPUTS values."""
    if not folder:
        raise gr.Error("Pick an entry first.")
    entry = Path(folder)
    meta = history.load(entry)
    job, segs = meta["job"], meta["segments"]
    if not Path(job["audio"]).exists():
        raise gr.Error("This entry's audio file is missing.")
    text = (entry / "transcript.txt").read_text(encoding="utf-8")
    fixed_path = entry / "corrected.txt"
    fixed = fixed_path.read_text(encoding="utf-8") if fixed_path.exists() else ""
    txt, srt_path = out_file(job["title"], ".txt"), out_file(job["title"], ".srt")
    txt.write_text(text, encoding="utf-8")
    srt = to_srt(retime(text, segs)) if segs else ""
    srt_path.write_text(srt, encoding="utf-8")
    partial = None
    note = f"📂 Opened “{job['title']}”."
    if still_running:
        note = (f"📂 “{job['title']}” is still being transcribed by the batch — this is the text so "
                "far. Open it again when it's done for the full result.")
    elif meta["status"] != "done":
        partial = {"job": job, "entry": str(entry), "segments": segs, "at": meta["at"]}
        note = f"📂 Opened “{job['title']}” — stopped at {fmt_dur(meta['at'])}: press ⏯ Resume to continue."
    return (gr.Tabs(selected="main"), job["audio"], job["title"], language_of(job["model"]), text,
            file_button(txt), file_button(srt_path), srt, fixed, note, partial, segs, str(entry),
            gr.Button(visible=bool(partial)), gr.Button(visible=bool(partial)), job.get("video"),
            gr.skip())


def open_queue_item(queue, busy, evt: gr.SelectData):
    """A click on a Batch queue row: open that item in the Transcribe dashboard."""
    item = queue[evt.index[0]]
    if item.get("entry") and Path(item["entry"], "meta.json").exists():
        running = busy == "batch" and not item["status"].startswith(("✅", "❌"))
        return history_open(item["entry"], still_running=running)
    skip = gr.skip()
    if item["kind"] == "link":  # not transcribed yet: ready to fetch and run on its own
        note = (f"🔗 Not transcribed yet — the link is in the YouTube box: press **Audio only** "
                "(or **Video + audio**), then Transcribe.")
        return (gr.Tabs(selected="main"),) + (skip,) * 8 + (note,) + (skip,) * 6 + (item["src"],)
    audio, title, note, video = get_file_audio(item["src"], Path(item["src"]).suffix.lower() in VIDEO_EXT)
    return ((gr.Tabs(selected="main"), audio, title) + (skip,) * 6
            + ("🎞️ Not transcribed yet — the file is loaded: press Transcribe.",)
            + (skip,) * 5 + (video, skip))


def history_delete(folder):
    if folder:
        history.delete(Path(folder))
    return history_choices(), ""


# ---- buttons: what can be pressed when ------------------------------------------------------

def busy_buttons(kind):
    """While transcribing, fixing, exporting or batching: only Stop can be pressed."""
    def busy():
        off = gr.Button(interactive=False)
        return (gr.Button("⏳ Working…", interactive=False), gr.Button(interactive=True),
                gr.Button(visible=kind == "transcribe"), off, off, off, off,
                gr.Button(visible=False), gr.Button(visible=False), off, gr.Button(interactive=True),
                *(gr.update(interactive=False),) * 4,
                gr.Button(visible=False), off, kind)
    return busy


def idle_buttons(audio, text, partial, fix_partial):
    """Nothing running: Transcribe needs audio, Fix needs a transcript, Resume a stopped run."""
    on = gr.Button(interactive=True)
    return (gr.Button("▶ Transcribe", interactive=bool(audio)), gr.Button(interactive=False),
            gr.Button(visible=False), gr.Button(interactive=bool((text or "").strip())), on, on, on,
            gr.Button(visible=bool(partial)), gr.Button(visible=bool(partial)), on,
            gr.Button(interactive=False), *(gr.update(interactive=True),) * 4,
            gr.Button(visible=bool(fix_partial)), on, "")


def stopped(audio, text, partial, fix_partial, kind):
    message = "⏹ Stopped."
    pbar_update, fix_update = gr.skip(), gr.skip()
    if kind == "transcribe":
        message = "⏹ Stopped — the text so far is kept (and saved in History)."
        if partial:
            message = (f"⏹ Stopped at {fmt_dur(partial['at'])} — press ⏯ Resume to continue from "
                       "there, or Transcribe to start over.")
        pbar_update = bar(message, done=True)
    elif kind == "fix":
        message = "⏹ AI fix stopped — press ⏯ Resume fix to continue."
        fix_update = bar(message, done=True)
    return (message, pbar_update, fix_update) + idle_buttons(audio, text, partial, fix_partial)


def enable_if(value, busy):
    """Enable a button when value is non-empty, unless this page is busy."""
    return gr.skip() if busy else gr.Button(interactive=bool(value and str(value).strip()))


def text_choices(fixed):
    """Offer "AI-fixed text" (to send or export) only once there is some."""
    has_fixed = bool((fixed or "").strip())
    choices = ["Transcript", "AI-fixed text"] if has_fixed else ["Transcript"]
    radio = gr.Radio(choices=choices) if has_fixed else gr.Radio(choices=choices, value="Transcript")
    return radio, radio


def pick_language(language):
    """Show the models and options that fit the chosen language."""
    bangla = language == "Bangla"
    models = LANGUAGE_MODELS[language]
    return (gr.Radio(choices=models, value=models[0]), gr.Checkbox(visible=bangla),
            gr.Checkbox(visible=bangla), gr.Checkbox(visible=bangla),
            gr.Textbox(placeholder="e.g. রাজশাহী, তারেক রহমান, প্রথম আলো" if bangla
                       else "e.g. Stanford, Reed College, Steve Jobs"))


def clear_results():
    """Empty the results and the From/To range; the audio and any downloaded video stay."""
    return ("", "", "", no_file(), no_file(), "", "", no_file(), no_file(), "Cleared.",
            gr.Audio(value=None, visible=False), gr.HTML(visible=False), gr.Button(visible=False), "",
            None, gr.Button(visible=False), gr.Button(visible=False), None, None, gr.HTML(visible=False),
            None, gr.Button(visible=False))


# ---- Send to Claude -------------------------------------------------------------------------

# Runs in the browser: copy prompt + transcript, then open claude.ai with it prefilled
# when it fits in a URL (otherwise the user pastes it).
SEND_TO_CLAUDE_JS = """
async (transcript, fixed, which, task, custom, language) => {
    const tasks = %s;
    transcript = transcript || ""; fixed = fixed || ""; custom = custom || "";  // empty boxes are null
    const text = which.startsWith("AI") ? fixed : transcript;
    if (!text.trim() || (task.startsWith("Custom") && !custom.trim())) {
        return [text, fixed, which, task, custom, language, false];  // the server explains
    }
    const other = language === "English" ? "Bangla" : "English";
    const instruction = task.startsWith("Custom") ? custom
        : tasks[task].replaceAll("{lang}", language).replaceAll("{other}", other);
    const full = (instruction ? instruction + "\\n\\n" : "") + "Transcript:\\n" + text;
    // Open the tab first, while the click still counts as a user action (popup blockers).
    const q = encodeURIComponent(full);
    window.open("https://claude.ai/new" + (q.length < 14000 ? "?q=" + q : ""), "_blank");
    let copied = false;
    try { await navigator.clipboard.writeText(full); copied = true; } catch (e) {
        const area = document.createElement("textarea");  // fallback for blocked clipboard API
        area.value = full; document.body.appendChild(area); area.select();
        copied = document.execCommand("copy"); area.remove();
    }
    return [text, fixed, which, task, custom, language, copied];
}
""" % json.dumps(CLAUDE_TASKS)


def claude_sent(text, _fixed, which, task, custom, language, copied):
    # Shown under the button rather than raised: the button's error popup was easy to miss.
    if not (text or "").strip():
        return ("⚠️ There is no AI-fixed text yet: use the **AI-fixed** tab first."
                if which.startswith("AI") else "⚠️ Transcribe something first.")
    if task.startswith("Custom") and not (custom or "").strip():
        return "⚠️ Write what Claude should do in the **Your instruction** box."
    if copied:
        return ("📋 **Copied for Claude.** In the Claude tab that just opened, click the message box, "
                "press **Ctrl+V**, then Enter. (Short texts are filled in already.)")
    return ("⚠️ Your browser blocked copying. Use the copy button on the text box, then paste it "
            "into Claude with Ctrl+V.")


# ---- look -----------------------------------------------------------------------------------

# Browsers only allow sound after the user has interacted with the page, so the first click
# anywhere unlocks an AudioContext; sttBeep() then plays a two-tone alert through it.
BEEP_JS = """
<script>
(() => {
  let ctx = null;
  const unlock = () => {
    ctx = ctx || new (window.AudioContext || window.webkitAudioContext)();
    if (ctx.state === "suspended") ctx.resume();
  };
  document.addEventListener("pointerdown", unlock, { capture: true });
  document.addEventListener("keydown", unlock, { capture: true });
  window.sttBeep = () => {
    unlock();
    [[880, 0], [660, 0.28], [880, 0.56], [660, 0.84]].forEach(([freq, at]) => {
      const osc = ctx.createOscillator(), gain = ctx.createGain();
      osc.frequency.value = freq;
      gain.gain.setValueAtTime(0.0001, ctx.currentTime + at);
      gain.gain.exponentialRampToValueAtTime(0.3, ctx.currentTime + at + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + at + 0.22);
      osc.connect(gain).connect(ctx.destination);
      osc.start(ctx.currentTime + at);
      osc.stop(ctx.currentTime + at + 0.25);
    });
  };
})();
</script>
"""

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.emerald,
    secondary_hue=gr.themes.colors.teal,
    neutral_hue=gr.themes.colors.slate,
    # Hind Siliguri is designed for Bangla and has matching Latin letters.
    font=[gr.themes.GoogleFont("Hind Siliguri"), "Noto Sans Bengali", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="*neutral_50",
    body_background_fill_dark="*neutral_950",
    block_radius="14px",
    block_shadow="0 1px 3px rgb(15 23 42 / 0.06)",
    block_label_background_fill="transparent",
    block_label_background_fill_dark="transparent",
    block_label_text_color="*neutral_500",
    block_label_text_color_dark="*neutral_400",
    block_title_text_color="*neutral_600",
    block_title_text_color_dark="*neutral_300",
    button_large_radius="10px",
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
    button_primary_background_fill_dark="*primary_600",
    button_primary_text_color="white",
)

CSS = """
.gradio-container { max-width: none !important; }
.main.fill_width { max-width: 1280px !important; margin: 0 auto !important; width: 100% !important; }
#header { align-items: center; border-bottom: 1px solid var(--border-color-primary); padding-bottom: 10px; }
#header h1 { font-size: 1.75rem; margin: 0; }
#header p { margin: 2px 0 0; color: var(--body-text-color-subdued); }
#ram { text-align: right; font-size: 0.95rem; }
.step-title { margin: 8px 0 0; font-size: 1.05rem; font-weight: 600; color: var(--color-accent); }
#claude-btn { background: #c96442; color: white; border: none; }
#claude-btn:hover { background: #b4573a; }
.stop-btn { background: #fef2f2 !important; color: #b91c1c !important; border: 1px solid #fecaca !important; }
.stop-btn:hover { background: #fee2e2 !important; }
.dark .stop-btn { background: #450a0a !important; color: #fecaca !important; border-color: #7f1d1d !important; }
.transcript textarea { font-size: 1.08rem !important; line-height: 1.8 !important; }
.srt textarea { font-family: var(--font-mono) !important; font-size: 0.85rem !important; line-height: 1.6 !important; }
.downloads button { min-height: 44px; }
.source { background: var(--block-background-fill); border: 1px solid var(--border-color-primary);
          border-radius: 14px; padding: 4px 12px 12px; }
.source .tabitem { padding: 10px 0 0; gap: 10px; }
.source button.lg, .source button.md { min-height: 44px; }
.pbar { position: relative; height: 36px; border-radius: 10px; overflow: hidden;
        background: var(--neutral-200); margin-bottom: 4px; }
.dark .pbar { background: var(--neutral-800); }
.pbar .fill { position: absolute; top: 0; bottom: 0; left: 0; background: var(--primary-500);
              opacity: 0.35; transition: width 0.4s ease; }
.pbar.busy .fill { opacity: 0.2; background-size: 36px 36px; animation: pbar-move 1s linear infinite;
                   background-image: repeating-linear-gradient(45deg, var(--primary-500) 0 9px,
                                                               transparent 9px 18px); }
.pbar.done .fill { opacity: 0.25; }
.pbar span { position: relative; line-height: 36px; padding: 0 14px; font-weight: 600; }
@keyframes pbar-move { from { background-position: 0 0; } to { background-position: 36px 0; } }
#status p { margin: 0; padding: 10px 14px; border-radius: 10px;
            background: var(--block-background-fill); border: 1px solid var(--border-color-primary); }
#status p:empty { display: none; }
#alert { display: none; }
#batch-table tbody tr { cursor: pointer; }
#batch-table, #batch-table * { font-family: var(--font) !important; }
#batch-table tbody tr:hover td { background: var(--color-accent-soft); }
.hint p { margin: 0; color: var(--body-text-color-subdued); }
"""


def step(title: str) -> gr.HTML:
    return gr.HTML(f'<div class="step-title">{title}</div>', padding=False)


# ---- layout ---------------------------------------------------------------------------------

with gr.Blocks(title="Bangla STT", delete_cache=(3600, 3600), fill_width=True) as demo:
    title = gr.State("")
    ram_state = gr.State("ok")
    busy = gr.State("")  # what this page is doing: "", "transcribe", "fix", "export" or "batch"
    partial = gr.State(None)  # a stopped transcription that Resume can continue
    fix_partial = gr.State(None)  # a stopped AI fix that Resume fix can continue
    segs = gr.State(None)  # the timed pieces of the current transcript, for subtitles
    entry = gr.State(None)  # the History folder of the current transcript
    video_src = gr.State(None)  # the video the audio came from, for subtitled video export
    batch_queue = gr.State([])  # the Batch tab's items: {id, kind, src, name, status, language}

    with gr.Row(elem_id="header"):
        gr.Markdown("# 🎙️ Bangla & English Speech-to-Text\n"
                    "Runs entirely on this PC — no audio is sent anywhere.")
        ram = gr.Markdown(ram_status(), elem_id="ram")
        alert = gr.HTML("", elem_id="alert", padding=False)

    with gr.Tabs() as pages:
        with gr.Tab("🎙️ Transcribe", id="main"):
            with gr.Row():
                # ---- Left: input ------------------------------------------------------------
                with gr.Column(scale=2):
                    step("① Get the audio")
                    with gr.Tabs(elem_classes="source"):
                        with gr.Tab("▶ YouTube"):
                            youtube_url = gr.Textbox(show_label=False, container=False,
                                                     placeholder="Paste a YouTube link here…")
                            with gr.Row(equal_height=True):
                                get_yt = gr.Button("⬇ Audio only", variant="primary")
                                get_yt_video = gr.Button("🎬 Video + audio")
                            save_video = gr.DownloadButton("💾 Save the video", visible=False)
                        with gr.Tab("💻 From your computer"):
                            with gr.Row(equal_height=True):
                                video_btn = gr.UploadButton("🎬 Video file", file_types=["video"],
                                                            type="filepath")
                                audio_btn = gr.UploadButton("🎵 Audio file", file_types=["audio"],
                                                            type="filepath")
                    audio = gr.Audio(label="Audio — listen, ✂ trim, or 🎤 record", type="filepath",
                                     sources=["microphone"], buttons=["download"])
                    with gr.Row():
                        start = gr.Textbox(label="From (optional)", placeholder="0:00")
                        end = gr.Textbox(label="To (optional)", placeholder="end")
                    clip = gr.Audio(label="Selected part", type="filepath", interactive=False,
                                    visible=False, buttons=["download"])
                    language = gr.Radio(["Bangla", "English"], value="Bangla", label="Language of the speech")
                    with gr.Accordion("⚙️ Settings", open=False):
                        model_name = gr.Radio(LANGUAGE_MODELS["Bangla"], value=DEFAULT_MODEL, label="Model")
                        quantize = gr.Checkbox(value=True, label="Fast mode (int8): ~1.7x faster, less RAM, "
                                                                 "same accuracy")
                        use_lm = gr.Checkbox(value=True, label="Bangla spell-fixing language model "
                                                               "(3.5 GB download on first use)")
                        digits = gr.Checkbox(value=False, label="Write numbers in Bangla digits (২০, not 20)")
                        hotwords = gr.Textbox(
                            label="Names or special words in the audio (optional, comma separated)",
                            placeholder="e.g. রাজশাহী, তারেক রহমান, প্রথম আলো",
                            info="Helps the right spelling of names win when the audio sounds close to them.")
                        use_browser = gr.Checkbox(
                            value=bool(BROWSER), visible=bool(BROWSER),
                            label=f"If YouTube says “confirm you're not a bot”, use my "
                                  f"{(BROWSER or ('',))[0].title()} login (cookies stay on this PC)")
                        cookies_file = gr.File(
                            label="YouTube cookies.txt (only if YouTube still blocks the download)",
                            file_types=[".txt"], type="filepath")
                    with gr.Row():
                        btn = gr.Button("▶ Transcribe", variant="primary", size="lg", scale=3,
                                        interactive=False)  # until there is audio
                        resume_btn = gr.Button("⏯ Resume", size="lg", scale=2, visible=False)
                        stop = gr.Button("■ Stop", size="lg", scale=1, min_width=90,
                                         elem_classes="stop-btn", interactive=False)
                    status = gr.Markdown("", elem_id="status")

                # ---- Right: result ----------------------------------------------------------
                with gr.Column(scale=3):
                    step("② Result")
                    with gr.Row(equal_height=True):
                        pbar = gr.HTML("", padding=False, visible=False, scale=1)  # while transcribing
                        stop_top = gr.Button("■ Stop", visible=False, scale=0, min_width=100,
                                             elem_classes="stop-btn")
                        resume_top = gr.Button("⏯ Resume", visible=False, scale=0, min_width=120,
                                               variant="primary")
                    with gr.Tabs():
                        with gr.Tab("Transcript"):
                            transcript = gr.Textbox(
                                show_label=False, lines=16, max_lines=16, interactive=True,
                                buttons=["copy"], elem_classes="transcript",
                                placeholder="The transcript appears here. You can edit it; "
                                            "the downloads follow your edits.")
                        with gr.Tab("Subtitles (.srt)"):
                            srt_view = gr.Textbox(show_label=False, lines=16, max_lines=16,
                                                  buttons=["copy"], elem_classes="srt", autoscroll=False)
                        with gr.Tab("AI-fixed (local)"):
                            with gr.Row(equal_height=True):
                                llm = gr.Dropdown([], label="Ollama model", allow_custom_value=True, scale=3)
                                refresh = gr.Button("↻", scale=0, min_width=44)
                                fix_btn = gr.Button("✨ Fix transcript", variant="primary", scale=2,
                                                    interactive=False)  # until there is a transcript
                                fix_resume_btn = gr.Button("⏯ Resume fix", scale=2, visible=False)
                            fix_bar = gr.HTML("", padding=False, visible=False)
                            corrected = gr.Textbox(
                                show_label=False, lines=12, max_lines=12, interactive=True,
                                buttons=["copy"], elem_classes="transcript",
                                placeholder="Press 'Fix transcript' to add punctuation and fix spelling "
                                            "with a free AI on this PC (Ollama). Slower and less accurate "
                                            "than Claude; check the result.")
                            with gr.Row(elem_classes="downloads"):
                                fixed_txt_btn = gr.DownloadButton("⬇ Fixed text (.txt)", interactive=False)
                                fixed_srt_btn = gr.DownloadButton("⬇ Fixed subtitles (.srt)",
                                                                  interactive=False)
                        with gr.Tab("⬇ Export"):
                            export_which = gr.Radio(["Transcript"], value="Transcript",
                                                    label="Export which text")
                            export_btn = gr.Button("📄 Make .vtt subtitles and a Word (.docx) file")
                            export_out = gr.File(label="Files", file_count="multiple", visible=False)
                            gr.Markdown("**Video with subtitles** — for audio that came from a video "
                                        "(🎬 Video + audio, or 🎬 Video file).")
                            video_btn_burn = gr.Button("🎬 Make the video with subtitles")
                            export_bar = gr.HTML("", padding=False, visible=False)
                            save_subtitled = gr.DownloadButton("💾 Save video with subtitles", visible=False)
                    with gr.Row(elem_classes="downloads"):
                        txt_btn = gr.DownloadButton("⬇ Transcript (.txt)", interactive=False)
                        srt_btn = gr.DownloadButton("⬇ Subtitles (.srt)", interactive=False)
                        clear = gr.Button("✕ Clear results", min_width=120)

                    step("③ Send to Claude")
                    with gr.Group():
                        with gr.Row(equal_height=True):
                            claude_text = gr.Radio(["Transcript"], value="Transcript", label="Text to send")
                            claude_task = gr.Dropdown(list(CLAUDE_TASKS), value=next(iter(CLAUDE_TASKS)),
                                                      label="Ask Claude to…")
                        claude_custom = gr.Textbox(label="Your instruction (only for Custom)", lines=1,
                                                   placeholder="e.g. Turn this into a news article")
                    claude_btn = gr.Button("Send to Claude ↗", size="lg", elem_id="claude-btn")
                    claude_status = gr.Markdown("", elem_id="claude-status")

        with gr.Tab("📚 Batch", id="batch"):
            gr.Markdown("Queue up YouTube links and audio/video files, then start — they are "
                        "transcribed one after another and saved in **🕘 History**. Uses the "
                        "Settings of the Transcribe tab.")
            with gr.Row():
                with gr.Column(scale=2):
                    step("① Add to the queue")
                    with gr.Group():
                        with gr.Row(equal_height=True):
                            batch_link = gr.Textbox(show_label=False, container=False, scale=4,
                                                    placeholder="Paste YouTube link(s), then Add or Enter")
                            batch_add = gr.Button("➕ Add", variant="primary", scale=1, min_width=90)
                        batch_upload = gr.UploadButton("🎬🎵 Add audio / video files (pick many at once)",
                                                       file_types=["audio", "video"],
                                                       file_count="multiple", type="filepath")
                    batch_lang = gr.Radio(["Auto", "Bangla", "English"], value="Auto",
                                          label="Language (Auto detects each one)")
                    with gr.Row():
                        batch_btn = gr.Button("▶ Start", variant="primary", size="lg", scale=3)
                        batch_stop = gr.Button("■ Stop", size="lg", scale=1, elem_classes="stop-btn",
                                               interactive=False)
                with gr.Column(scale=3):
                    step("② Queue")
                    batch_bar = gr.HTML("", padding=False, visible=False)
                    gr.Markdown("Click a row to open it in the 🎙️ Transcribe dashboard — transcript, "
                                "subtitles, AI fix, export, Claude, Resume.", elem_classes="hint")
                    batch_table = gr.Dataframe(
                        headers=["#", "Item", "Language", "Status", ""], value=[],
                        datatype=["number", "str", "str", "str", "str"], interactive=False, wrap=True,
                        column_widths=["48px", "45%", "90px", "25%", "90px"], elem_id="batch-table")
                    with gr.Row(equal_height=True):
                        batch_pick = gr.Dropdown([], multiselect=True, label="Remove from the queue",
                                                 scale=4)
                        batch_remove = gr.Button("🗑 Remove", scale=1, min_width=100)
                        batch_clear = gr.Button("✕ Empty queue", scale=1, min_width=150)

        with gr.Tab("🕘 History", id="history"):
            gr.Markdown("Every transcription is kept here (in the `history` folder of the app), even "
                        "after the app is closed. Open one to read, download or **Resume** it.")
            with gr.Row(equal_height=True):
                history_pick = gr.Dropdown(history.entries(), label="Saved transcriptions", scale=4)
                history_refresh = gr.Button("↻", scale=0, min_width=44)
            history_info = gr.Markdown("")
            with gr.Row():
                history_open_btn = gr.Button("📂 Open", variant="primary")
                history_delete_btn = gr.Button("🗑 Delete", elem_classes="stop-btn")

    # ---- events ---------------------------------------------------------------------------
    # Buttons switched together by busy_buttons()/idle_buttons(), then the page's busy state.
    controls = [btn, stop, stop_top, fix_btn, clear, get_yt, get_yt_video, resume_btn, resume_top,
                batch_btn, batch_stop, batch_add, batch_upload, batch_remove, batch_clear,
                fix_resume_btn, video_btn_burn, busy]
    run_outputs = [transcript, txt_btn, srt_btn, srt_view, status, clip, pbar, partial, segs, entry]
    idle_inputs = [audio, transcript, partial, fix_partial]

    transcribe_event = btn.click(busy_buttons("transcribe"), None, controls).then(
        run, [audio, title, model_name, quantize, use_lm, hotwords, start, end, digits, video_src],
        run_outputs,
        show_progress="hidden")  # our own progress bar above the transcript shows it
    resume_event = resume_btn.click(busy_buttons("transcribe"), None, controls).then(
        resume, partial, run_outputs, show_progress="hidden")
    resume_top_event = resume_top.click(busy_buttons("transcribe"), None, controls).then(
        resume, partial, run_outputs, show_progress="hidden")
    fix_inputs = [transcript, llm, language, segs, title, entry, fix_partial]
    fix_outputs = [corrected, fixed_txt_btn, fixed_srt_btn, fix_bar, fix_partial]
    correct_event = fix_btn.click(busy_buttons("fix"), None, controls).then(
        run_correct, fix_inputs, fix_outputs, show_progress="hidden")
    fix_resume_event = fix_resume_btn.click(busy_buttons("fix"), None, controls).then(
        resume_correct, fix_inputs, fix_outputs, show_progress="hidden")
    export_event = video_btn_burn.click(busy_buttons("export"), None, controls).then(
        export_video, [export_which, transcript, corrected, segs, title, video_src],
        [export_bar, save_subtitled], show_progress="hidden")
    batch_event = batch_btn.click(busy_buttons("batch"), None, controls).then(
        run_batch, [batch_queue, batch_lang, quantize, use_lm, digits, cookies_file, use_browser],
        [batch_queue, batch_table, batch_bar], show_progress="hidden")
    for trigger in (batch_add.click, batch_link.submit):
        trigger(add_links, [batch_link, batch_queue], [batch_link, batch_queue, batch_table, batch_pick])
    batch_upload.upload(add_files, [batch_upload, batch_queue], [batch_queue, batch_table, batch_pick])
    batch_remove.click(remove_items, [batch_pick, batch_queue], [batch_queue, batch_table, batch_pick])
    batch_clear.click(clear_queue, None, [batch_queue, batch_table, batch_pick, batch_bar])
    running = [transcribe_event, resume_event, resume_top_event, correct_event, fix_resume_event,
               export_event, batch_event]
    for event in running:  # .then runs after errors too; Stop restores the buttons itself
        event.then(idle_buttons, idle_inputs, controls)
    stopping = bar_html("⏹ Stopping — finishing the current step…", done=True)
    for button in (stop, stop_top, batch_stop):
        # Instant feedback from the browser: while a model loads the server can take a few
        # seconds to answer.
        button.click(None, None, [status, pbar], js=f"() => ['⏹ Stopping…', {json.dumps(stopping)}]")
        # queue=False: answer at once instead of waiting behind the busy queue.
        button.click(stopped, idle_inputs + [busy], [status, pbar, fix_bar] + controls,
                     cancels=running, queue=False)
    clear.click(clear_results, None,
                [start, end, transcript, txt_btn, srt_btn, srt_view, corrected, fixed_txt_btn,
                 fixed_srt_btn, status, clip, pbar, stop_top, claude_status, partial, resume_btn,
                 resume_top, segs, entry, fix_bar, fix_partial, fix_resume_btn], cancels=running)

    # Getting audio
    get_yt.click(get_youtube, [youtube_url, cookies_file, use_browser], [audio, title, status, video_src],
                 show_progress_on=[audio])
    youtube_url.submit(get_youtube, [youtube_url, cookies_file, use_browser],
                       [audio, title, status, video_src], show_progress_on=[audio])
    get_yt_video.click(get_youtube_video, [youtube_url, cookies_file, use_browser],
                       [audio, title, status, save_video, video_src], show_progress_on=[audio])
    video_btn.upload(get_video_file, video_btn, [audio, title, status, video_src], show_progress_on=[audio])
    audio_btn.upload(get_audio_file, audio_btn, [audio, title, status, video_src], show_progress_on=[audio])
    audio.stop_recording(lambda: ("recording", None), None, [title, video_src])
    # New or trimmed audio: length hint, From..To preview, language, buttons, stale Resume.
    audio.change(show_length, audio, end).then(selected_part, [audio, start, end], clip).then(
        detect_language, [audio, language, transcript], [language, status])
    audio.change(enable_if, [audio, busy], btn)
    audio.change(audio_changed, [audio, partial], [partial, resume_btn, resume_top])
    demo.load(enable_if, [audio, busy], btn)  # for audio that is already there when the page opens
    for trigger in (start.blur, start.submit, end.blur, end.submit):
        trigger(selected_part, [audio, start, end], clip)
    language.change(pick_language, language, [model_name, quantize, use_lm, digits, hotwords])

    # Edits: the downloads follow the text boxes
    transcript.change(enable_if, [transcript, busy], fix_btn)
    transcript.input(lambda t, ti, en: edited_text(t, ti, en, "transcript.txt"),
                     [transcript, title, entry], txt_btn, trigger_mode="always_last")
    transcript.blur(lambda t, s, ti: edited_subtitles(t, s, ti, ".srt"), [transcript, segs, title],
                    [srt_btn, srt_view])
    corrected.input(lambda t, ti, en: edited_text(t, ti, en, "corrected.txt"),
                    [corrected, title, entry], fixed_txt_btn, trigger_mode="always_last")
    corrected.blur(lambda t, s, ti: edited_subtitles(t, s, ti, " (fixed).srt")[0],
                   [corrected, segs, title], fixed_srt_btn)

    corrected.change(text_choices, corrected, [claude_text, export_which])

    # Export
    export_btn.click(export_files, [export_which, transcript, corrected, segs, title], export_out)

    # History
    history_refresh.click(history_choices, None, history_pick)
    pages.select(history_choices, None, history_pick)  # fresh list whenever a page is opened
    history_pick.change(history_preview, history_pick, history_info)
    open_outputs = [pages, audio, title, language, transcript, txt_btn, srt_btn, srt_view, corrected,
                    status, partial, segs, entry, resume_btn, resume_top, video_src, youtube_url]
    history_open_btn.click(history_open, history_pick, open_outputs)
    batch_table.select(open_queue_item, [batch_queue, busy], open_outputs)
    history_delete_btn.click(history_delete, history_pick, [history_pick, history_info])

    # Claude, LLM list, RAM
    copied = gr.Checkbox(False, visible=False)  # filled by the browser (a State can't be)
    claude_btn.click(claude_sent, [transcript, corrected, claude_text, claude_task, claude_custom,
                                   language, copied], claude_status, js=SEND_TO_CLAUDE_JS)
    refresh.click(refresh_llms, None, llm)
    demo.load(refresh_llms, None, llm)
    gr.Timer(5).tick(ram_tick, ram_state, [ram, ram_state, alert], show_progress="hidden").then(
        None, alert, None, js="(signal) => { if (signal) window.sttBeep && window.sttBeep(); }")


def open_browser(url: str):
    """Open the app in the default browser without its startup chatter in this terminal
    (a browser that is already running prints e.g. "zygote … Broken pipe" while handing over)."""
    opener = shutil.which("xdg-open")
    if opener:
        subprocess.Popen([opener, url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
    else:
        import webbrowser
        webbrowser.open(url)


class _HideCancelRace(logging.Filter):
    """Gradio 6.30 may get two cancel requests for one event and fail the second with a
    KeyError in cancel_event (the event is already cancelled). Harmless; don't print it."""

    def filter(self, record):
        exc = record.exc_info and record.exc_info[1]
        return not (isinstance(exc, KeyError) and "cancel_event" in "".join(
            traceback.format_tb(record.exc_info[2])))


if __name__ == "__main__":
    logging.getLogger("uvicorn.error").addFilter(_HideCancelRace())
    history.HISTORY_DIR.mkdir(exist_ok=True)
    # No footer links: the API docs, settings and run history are developer tools; History
    # above keeps the results. allowed_paths: History audio is played and downloaded from there.
    demo.queue().launch(theme=THEME, css=CSS, head=BEEP_JS, footer_links=[], run_history=False,
                        allowed_paths=[str(history.HISTORY_DIR)], prevent_thread_lock=True, quiet=True)
    print(f"Bangla STT is running at {demo.local_url}  —  press Ctrl+C here to stop it.", flush=True)
    open_browser(demo.local_url)
    demo.block_thread()
