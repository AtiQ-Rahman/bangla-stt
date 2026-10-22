"""Gradio web UI: paste a YouTube link or upload audio, get a Bangla transcript."""
import atexit
import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path

import gradio as gr

from stt.audio import cut_clip, download_youtube, download_youtube_video, duration, load_audio
from stt.correct import DEFAULT_LLM, correct, list_models
from stt.langid import detect
from stt.transcribe import (DEFAULT_MODEL, LANGUAGE_MODELS, LM_RAM_GB, RAM_NEEDED_GB, is_loaded,
                            to_srt, to_text, transcribe, unload)

# All temporary files of this app live in a per-process folder, emptied per
# transcription and removed on exit. Folders left by crashed runs (their process
# is gone) are removed on start.
WORK_DIR = Path(tempfile.gettempdir()) / f"bangla-stt-{os.getpid()}"
for old in Path(tempfile.gettempdir()).glob("bangla-stt-*"):
    pid = old.name.removeprefix("bangla-stt-")
    if pid.isdigit() and not Path(f"/proc/{pid}").exists():
        shutil.rmtree(old, ignore_errors=True)
atexit.register(shutil.rmtree, WORK_DIR, ignore_errors=True)

IDLE_UNLOAD_SEC = 10 * 60  # free the models' RAM after this long without use
LLM_RAM_GB = 4.0

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
    m, s = divmod(int(round(sec)), 60)
    return f"{m}:{s:02}"


def safe_stem(title: str) -> str:
    return "".join(c if c.isalnum() or c in " -_" else "_" for c in title).strip() or "transcript"


def fresh_media_path(stem: str) -> Path:
    """A new file in the media folder, after removing what an earlier fetch left there."""
    media = WORK_DIR / "media"
    shutil.rmtree(media, ignore_errors=True)
    media.mkdir(parents=True)
    return media / f"{safe_stem(stem)}.mp3"


def get_youtube(url, cookies_file, progress=gr.Progress()):
    """Download a YouTube video's audio into the Audio player, ready to listen to and trim."""
    url = (url or "").strip()
    if not url:
        raise gr.Error("Paste a YouTube link first.")
    progress(0, desc="Downloading audio from YouTube…")
    tmp = WORK_DIR / "download"
    try:
        path, title = download_youtube(url, tmp, cookies_file)
        progress(0.8, desc="Converting…")
        audio = cut_clip(path, fresh_media_path(title))
    except Exception as e:
        raise gr.Error(
            f"Download failed: {e}. If YouTube asks you to sign in or flags you as a bot, "
            "upload a cookies.txt file exported from your browser (Settings).")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return str(audio), title, f"✅ Got the audio of “{title}”. Listen, trim if you like, then Transcribe."


def get_youtube_video(url, cookies_file, progress=gr.Progress()):
    """Download the YouTube video itself (to save), and put its sound into the Audio player."""
    url = (url or "").strip()
    if not url:
        raise gr.Error("Paste a YouTube link first.")
    progress(0, desc="Downloading the video (up to 720p)…")
    out = WORK_DIR / "video"
    shutil.rmtree(out, ignore_errors=True)  # only the latest video is kept
    try:
        path, title = download_youtube_video(url, out, cookies_file)
        progress(0.9, desc="Extracting the audio…")
        audio = cut_clip(path, fresh_media_path(title))
    except Exception as e:
        raise gr.Error(
            f"Download failed: {e}. If YouTube asks you to sign in or flags you as a bot, "
            "upload a cookies.txt file exported from your browser (Settings).")
    size = path.stat().st_size / 1024 / 1024
    save = gr.DownloadButton(f"💾 Save the video ({size:.0f} MB)", value=str(path), visible=True)
    return str(audio), title, f"✅ Got the video “{title}”. Save it below; its audio is ready too.", save


def get_file_audio(upload, progress=gr.Progress()):
    """Put the sound of an uploaded video or audio file into the Audio player (as mp3)."""
    if not upload:
        return gr.skip(), gr.skip(), gr.skip()
    progress(0, desc="Reading the file…")
    title = Path(upload).stem
    try:
        audio = cut_clip(upload, fresh_media_path(title))
    except RuntimeError as e:
        raise gr.Error(f"Could not read the sound of this file: {e}")
    return str(audio), title, f"✅ Loaded “{title}”. Listen, trim if you like, then Transcribe."


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


def detect_language(path, current):
    """Set the Language choice to what the loaded audio sounds like."""
    if not path:
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


def bar_html(text: str, fraction: float | None = None, done: bool = False) -> str:
    """Progress bar shown above the transcript; fraction None = busy without a known amount."""
    cls = "pbar done" if done else "pbar" if fraction is not None else "pbar busy"
    width = 100 if fraction is None else round(fraction * 100)
    return f'<div class="{cls}"><div class="fill" style="width:{width}%"></div><span>{text}</span></div>'


def bar(text: str, fraction: float | None = None, done: bool = False) -> gr.HTML:
    return gr.HTML(bar_html(text, fraction, done), visible=True)


def stage(message: str) -> tuple:
    """A run() update that only changes the status line and the progress bar."""
    return gr.skip(), gr.skip(), gr.skip(), gr.skip(), message, gr.skip(), bar(message), gr.skip()


def file_button(path: Path) -> gr.DownloadButton:
    return gr.DownloadButton(value=str(path), interactive=True)


def no_file() -> gr.DownloadButton:
    return gr.DownloadButton(value=None, interactive=False)


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


def run(audio, title, model_name, quantize, use_lm, hotwords, start, end):
    if not audio:
        raise gr.Error("Add some audio first: get it from YouTube, upload a video or audio file, or record.")
    start, end = parse_time(start), parse_time(end)
    if start and end and end <= start:
        raise gr.Error("End time must be after start time.")
    use_lm = use_lm and model_name not in LANGUAGE_MODELS["English"]  # Whisper has its own
    check_ram(model_name, quantize, use_lm)
    words = [w.strip() for w in (hotwords or "").replace("\n", ",").split(",") if w.strip()]
    job = {"audio": audio, "title": title or Path(audio).stem, "model": model_name,
           "quantize": quantize, "use_lm": use_lm, "words": words, "start": start, "end": end}

    # Clear the previous result straight away.
    yield ("", no_file(), no_file(), "", "⏳ Starting…", gr.Audio(), bar("⏳ Checking the language…"),
           None)

    chosen = "English" if model_name in LANGUAGE_MODELS["English"] else "Bangla"
    try:
        heard, confidence = detect(audio, start, end)
    except Exception:
        heard, confidence = chosen, 0.0
    if heard != chosen and confidence >= 0.8:
        raise gr.Error(f"This sounds like {heard} ({confidence:.0%} sure), but {chosen} is selected. "
                       f"Choose {heard} under 'Language of the speech' and press Transcribe again.")
    shutil.rmtree(WORK_DIR / "run", ignore_errors=True)
    yield from _job(job, [], start or 0.0)


def resume(partial):
    """Continue a stopped transcription from its last word, with the same audio and settings."""
    if not partial:
        raise gr.Error("Nothing to resume: press Transcribe.")
    job = partial["job"]
    check_ram(job["model"], job["quantize"], job["use_lm"])
    yield stage(f"⏳ Resuming at {fmt_dur(partial['at'])}…")
    yield from _job(job, partial["segments"], partial["at"])


def _job(job, prior, at):
    """Transcribe job["audio"] from `at` seconds to job["end"], after the segments in prior.

    Yields the run outputs; the last one is a "partial" record ({job, segments, at}) that
    Resume continues from, or None once the whole range is done.
    """
    global _last_used, _active
    path, title = Path(job["audio"]), job["title"]
    begin = job["start"] or 0.0
    run_dir = WORK_DIR / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    txt_path, srt_path = run_dir / f"{safe_stem(title)}.txt", run_dir / f"{safe_stem(title)}.srt"

    _active += 1
    try:
        yield stage("⏳ Reading the audio…")
        clip = gr.skip()
        # Resuming: hear 3 s before the last word too, as context; words only come after `at`.
        load_from = max(begin, at - 3.0) if prior else at
        try:
            wav = load_audio(path, load_from, job["end"])
            if not prior and (job["start"] or job["end"]):  # let the user hear what is transcribed
                clip = gr.Audio(value=str(cut_clip(path, run_dir / "selected.mp3", job["start"],
                                                   job["end"])), visible=True, label="Transcribed part")
        except RuntimeError as e:
            raise gr.Error(f"Could not read the audio: {e}")
        if not len(wav):
            raise gr.Error("No audio left in the selected range.")

        model_name, quantize = job["model"], job["quantize"]
        was_loaded = is_loaded(model_name, quantize)
        if not was_loaded:
            yield stage("⏳ Loading the model into memory — the first time this takes up to a minute…")
        before, remaining = load_from - begin, len(wav) / 16000
        audio_sec = before + remaining  # the whole selected range, for progress
        segments, t0 = list(prior), time.monotonic()
        first = None  # (time, fraction) of the first piece, to estimate the time left after loading
        for done, total, seg in transcribe(wav, model_name, quantize, job["use_lm"], offset=load_from,
                                           hotwords=job["words"], keep_from_sec=at - load_from):
            frac = (before + done / total * remaining) / audio_sec
            now = time.monotonic()
            first = first or (now, frac)
            left = ""
            if first[1] < frac < 1 and now - first[0] > 5:
                secs = (now - first[0]) / (frac - first[1]) * (1 - frac)
                left = f" · about {fmt_dur(secs)} left"
            if seg:
                segments.append(seg)
            # Rewrite the files every piece so a stopped run still leaves a partial transcript.
            txt_path.write_text(to_text(segments), encoding="utf-8")
            srt = to_srt(segments)
            srt_path.write_text(srt, encoding="utf-8")
            status = "⏳ Transcribing — press Stop any time; you can Resume later"
            progress_text = (f"⏳ Transcribing… {frac:.0%} · {fmt_dur(frac * audio_sec)} of "
                             f"{fmt_dur(audio_sec)}{left}")
            resume_at = segments[-1].end if segments and segments[-1].end > at else at
            yield (to_text(segments), file_button(txt_path), file_button(srt_path), srt, status, clip,
                   bar(progress_text, frac), {"job": job, "segments": segments, "at": resume_at})
    finally:
        _active -= 1
        _last_used = time.monotonic()
    took = time.monotonic() - t0
    done_text = (f"✅ Done: {fmt_dur(audio_sec)} of audio"
                 + (f" (this part took {fmt_dur(took)})" if prior else f" in {fmt_dur(took)}")
                 + ("" if was_loaded else ", incl. loading the model"))
    yield (to_text(segments), file_button(txt_path), file_button(srt_path), to_srt(segments),
           done_text, clip, bar(done_text, 1, done=True), None)


def run_correct(text, llm, language):
    if not (text or "").strip():
        raise gr.Error("Transcribe something first.")
    if not list_models():
        raise gr.Error("Ollama is not running. Start it with `docker start ollama` (see README).")
    if _active:
        raise gr.Error("Wait for the transcription to finish (or press Stop) first.")
    unload()  # the speech models and the LLM together don't fit in 16 GB RAM
    avail, _ = mem_gb()
    if avail < LLM_RAM_GB:
        raise gr.Error(f"Not enough free RAM for the LLM ({avail:.1f} GB free, ~{LLM_RAM_GB:.0f} GB needed).")
    out = WORK_DIR / "corrected.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    fixed, rejected = "", 0
    try:
        for fixed, rejected in correct(text, llm, language):
            yield fixed, no_file()
    except Exception as e:
        raise gr.Error(f"LLM failed: {e}")
    out.write_text(fixed, encoding="utf-8")
    if rejected:
        gr.Info(f"Undid {rejected} AI change(s) that added, dropped or reworded words, "
                f"or used a non-{language} script. The rest of the fixes were kept.")
    yield fixed, file_button(out)


def refresh_llms():
    models = list_models()
    value = DEFAULT_LLM if DEFAULT_LLM in models else (models[0] if models else None)
    return gr.Dropdown(choices=models, value=value)


def clear_results():
    """Empty the results and the From/To range; the audio and any downloaded video stay."""
    return ("", "", "", no_file(), no_file(), "", "", no_file(), "Cleared.",
            gr.Audio(value=None, visible=False), gr.HTML(visible=False), gr.Button(visible=False), "",
            None, gr.Button(visible=False))


def busy_buttons():
    """While transcribing or fixing: only Stop can be pressed."""
    off = gr.Button(interactive=False)
    return (gr.Button("⏳ Working…", interactive=False), gr.Button(interactive=True),
            gr.Button(visible=True), off, off, off, off, gr.Button(visible=False), True)


def idle_buttons(audio, text, partial):
    """Nothing running: Transcribe needs audio, Fix needs a transcript, Resume a stopped run."""
    on = gr.Button(interactive=True)
    return (gr.Button("▶ Transcribe", interactive=bool(audio)), gr.Button(interactive=False),
            gr.Button(visible=False), gr.Button(interactive=bool((text or "").strip())), on, on, on,
            gr.Button(visible=bool(partial)), False)


def stopped(audio, text, partial):
    message = "⏹ Stopped — the text so far is kept."
    if partial:
        message = (f"⏹ Stopped at {fmt_dur(partial['at'])} — press ⏯ Resume to continue from there, "
                   "or Transcribe to start over.")
    return (message, bar(message, done=True)) + idle_buttons(audio, text, partial)


def enable_if(value, busy):
    """Enable a button when value is non-empty, unless this page is transcribing or fixing."""
    return gr.skip() if busy else gr.Button(interactive=bool(value and str(value).strip()))


def pick_language(language):
    """Show the models and options that fit the chosen language."""
    bangla = language == "Bangla"
    models = LANGUAGE_MODELS[language]
    return (gr.Radio(choices=models, value=models[0]), gr.Checkbox(visible=bangla),
            gr.Checkbox(visible=bangla),
            gr.Textbox(placeholder="e.g. রাজশাহী, তারেক রহমান, প্রথম আলো" if bangla
                       else "e.g. Stanford, Reed College, Steve Jobs"))


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
    button_cancel_background_fill="*red_50",
    button_cancel_background_fill_hover="*red_100",
    button_cancel_background_fill_dark="*red_950",
    button_cancel_background_fill_hover_dark="*red_900",
    button_cancel_text_color="*red_700",
    button_cancel_text_color_dark="*red_200",
    button_cancel_border_color="*red_200",
    button_cancel_border_color_dark="*red_800",
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
"""

def step(title: str) -> gr.HTML:
    return gr.HTML(f'<div class="step-title">{title}</div>', padding=False)


with gr.Blocks(title="Bangla STT", delete_cache=(3600, 3600), fill_width=True) as demo:
    title = gr.State("")
    ram_state = gr.State("ok")

    with gr.Row(elem_id="header"):
        gr.Markdown("# 🎙️ Bangla & English Speech-to-Text\n"
                    "Runs entirely on this PC — no audio is sent anywhere.")
        ram = gr.Markdown(ram_status(), elem_id="ram")
        alert = gr.HTML("", elem_id="alert", padding=False)

    with gr.Row():
        # ---- Left: input --------------------------------------------------------
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
                        video = gr.UploadButton("🎬 Video file", file_types=["video"], type="filepath")
                        audio_file = gr.UploadButton("🎵 Audio file", file_types=["audio"],
                                                     type="filepath")
            audio = gr.Audio(label="Audio — listen, ✂ trim, or 🎤 record", type="filepath",
                             sources=["microphone"], buttons=["download"])
            with gr.Row():
                start = gr.Textbox(label="From (optional)", placeholder="0:00")
                end = gr.Textbox(label="To (optional)", placeholder="end")
            clip = gr.Audio(label="Selected part", type="filepath", interactive=False, visible=False,
                            buttons=["download"])
            language = gr.Radio(["Bangla", "English"], value="Bangla", label="Language of the speech")
            with gr.Accordion("⚙️ Settings", open=False):
                model_name = gr.Radio(LANGUAGE_MODELS["Bangla"], value=DEFAULT_MODEL, label="Model")
                quantize = gr.Checkbox(value=True, label="Fast mode (int8): ~1.7x faster, less RAM, same accuracy")
                use_lm = gr.Checkbox(value=True, label="Bangla spell-fixing language model "
                                                       "(3.5 GB download on first use)")
                hotwords = gr.Textbox(
                    label="Names or special words in the audio (optional, comma separated)",
                    placeholder="e.g. রাজশাহী, তারেক রহমান, প্রথম আলো",
                    info="Helps the right spelling of names win when the audio sounds close to them.")
                cookies_file = gr.File(
                    label="YouTube cookies.txt (only if YouTube blocks the download)",
                    file_types=[".txt"], type="filepath",
                )
            with gr.Row():
                btn = gr.Button("▶ Transcribe", variant="primary", size="lg", scale=3,
                                interactive=False)  # until there is audio
                resume_btn = gr.Button("⏯ Resume", size="lg", scale=2, visible=False)
                stop = gr.Button("■ Stop", size="lg", scale=1, min_width=90, elem_classes="stop-btn",
                                 interactive=False)
            status = gr.Markdown("", elem_id="status")

        # ---- Right: result ------------------------------------------------------
        with gr.Column(scale=3):
            step("② Result")
            with gr.Row(equal_height=True):
                pbar = gr.HTML("", padding=False, visible=False, scale=1)  # shown while transcribing
                stop_top = gr.Button("■ Stop", visible=False, scale=0, min_width=100,
                                     elem_classes="stop-btn")
            with gr.Tabs():
                with gr.Tab("Transcript"):
                    transcript = gr.Textbox(show_label=False, lines=16, max_lines=16, interactive=True,
                                            buttons=["copy"], elem_classes="transcript",
                                            placeholder="The transcript appears here. You can edit it.")
                with gr.Tab("Subtitles (.srt)"):
                    srt_view = gr.Textbox(show_label=False, lines=16, max_lines=16, buttons=["copy"],
                                          elem_classes="srt", autoscroll=False)
                with gr.Tab("AI-fixed (local)"):
                    with gr.Row(equal_height=True):
                        llm = gr.Dropdown([], label="Ollama model", allow_custom_value=True, scale=3)
                        refresh = gr.Button("↻", scale=0, min_width=44)
                        fix_btn = gr.Button("✨ Fix transcript", variant="primary", scale=2,
                                            interactive=False)  # until there is a transcript
                    corrected = gr.Textbox(show_label=False, lines=12, max_lines=12, interactive=True,
                                           buttons=["copy"], elem_classes="transcript",
                                           placeholder="Press 'Fix transcript' to add punctuation and fix "
                                                       "spelling with a free AI on this PC (Ollama). "
                                                       "Slower and less accurate than Claude; check the result.")
                    corrected_btn = gr.DownloadButton("⬇ Download fixed text (.txt)", interactive=False)
            with gr.Row(elem_classes="downloads"):
                txt_btn = gr.DownloadButton("⬇ Transcript (.txt)", interactive=False)
                srt_btn = gr.DownloadButton("⬇ Subtitles (.srt)", interactive=False)
                clear = gr.Button("✕ Clear results", min_width=120)

            step("③ Send to Claude")
            with gr.Group():
                with gr.Row(equal_height=True):
                    claude_text = gr.Radio(["Transcript", "AI-fixed text"], value="Transcript",
                                           label="Text to send")
                    claude_task = gr.Dropdown(list(CLAUDE_TASKS), value=next(iter(CLAUDE_TASKS)),
                                              label="Ask Claude to…")
                claude_custom = gr.Textbox(label="Your instruction (only for Custom)", lines=1,
                                           placeholder="e.g. Turn this into a news article")
            claude_btn = gr.Button("Send to Claude ↗", size="lg", elem_id="claude-btn")
            claude_status = gr.Markdown("", elem_id="claude-status")

    # [Transcribe, Stop, top Stop, Fix, Clear, Get audio, Get video] + this page's busy flag
    busy = gr.State(False)
    partial = gr.State(None)  # a stopped transcription that Resume can continue
    controls = [btn, stop, stop_top, fix_btn, clear, get_yt, get_yt_video, resume_btn, busy]
    transcribe_event = btn.click(busy_buttons, None, controls).then(
        run, [audio, title, model_name, quantize, use_lm, hotwords, start, end],
        [transcript, txt_btn, srt_btn, srt_view, status, clip, pbar, partial],
        show_progress="hidden",  # our own progress bar above the transcript shows it
    )
    resume_event = resume_btn.click(busy_buttons, None, controls).then(
        resume, partial, [transcript, txt_btn, srt_btn, srt_view, status, clip, pbar, partial],
        show_progress="hidden")
    resume_event.then(idle_buttons, [audio, transcript, partial], controls)
    language.change(pick_language, language, [model_name, quantize, use_lm, hotwords])
    get_yt.click(get_youtube, [youtube_url, cookies_file], [audio, title, status], show_progress_on=[audio])
    get_yt_video.click(get_youtube_video, [youtube_url, cookies_file], [audio, title, status, save_video],
                       show_progress_on=[audio])
    youtube_url.submit(get_youtube, [youtube_url, cookies_file], [audio, title, status],
                       show_progress_on=[audio])
    for button in (video, audio_file):
        button.upload(get_file_audio, button, [audio, title, status], show_progress_on=[audio])
    audio.stop_recording(lambda: "recording", None, title)
    # A new or trimmed audio: refresh the length hint and the From..To preview.
    audio.change(show_length, audio, end).then(selected_part, [audio, start, end], clip).then(
        detect_language, [audio, language], [language, status])
    # Re-cut the preview when the range changes (blur/submit: not on every keystroke).
    for trigger in (start.blur, start.submit, end.blur, end.submit):
        trigger(selected_part, [audio, start, end], clip)
    correct_event = fix_btn.click(busy_buttons, None, controls).then(
        run_correct, [transcript, llm, language], [corrected, corrected_btn], show_progress_on=[corrected])
    correct_event.then(idle_buttons, [audio, transcript, partial], controls)
    transcribe_event.then(idle_buttons, [audio, transcript, partial], controls)  # after errors too
    audio.change(enable_if, [audio, busy], btn)
    demo.load(enable_if, [audio, busy], btn)  # for audio that is already there when the page opens
    transcript.change(enable_if, [transcript, busy], fix_btn)
    for button in (stop, stop_top):
        # queue=False: answer at once instead of waiting behind the busy queue.
        button.click(stopped, [audio, transcript, partial], [status, pbar] + controls,
                     cancels=[transcribe_event, resume_event, correct_event], queue=False)
    clear.click(clear_results, None, [start, end, transcript, txt_btn, srt_btn, srt_view, corrected,
                                      corrected_btn, status, clip, pbar, stop_top, claude_status,
                                      partial, resume_btn],
                cancels=[transcribe_event, resume_event, correct_event])
    # New or trimmed audio: an old stopped run no longer matches it.
    audio.change(lambda: (None, gr.Button(visible=False)), None, [partial, resume_btn])
    copied = gr.Checkbox(False, visible=False)  # filled by the browser (a State can't be)
    claude_btn.click(claude_sent, [transcript, corrected, claude_text, claude_task, claude_custom,
                                   language, copied], claude_status, js=SEND_TO_CLAUDE_JS)
    refresh.click(refresh_llms, None, llm)
    demo.load(refresh_llms, None, llm)
    gr.Timer(5).tick(ram_tick, ram_state, [ram, ram_state, alert], show_progress="hidden").then(
        None, alert, None, js="(signal) => { if (signal) window.sttBeep && window.sttBeep(); }")


if __name__ == "__main__":
    # No footer links: the API docs, settings and run history are developer tools, and old
    # runs can't be reloaded anyway once their temporary files are cleaned up.
    demo.queue().launch(inbrowser=True, theme=THEME, css=CSS, head=BEEP_JS,
                        footer_links=[], run_history=False)
