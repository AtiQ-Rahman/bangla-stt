"""History, and opening results in the dashboard (following running ones live)."""
import html
from pathlib import Path

import gradio as gr

from stt import history
from stt.retime import retime
from stt.transcribe import to_srt, to_text
from webui.common import bar, file_button, fmt_dur, language_of, LIVE, out_file, VIDEO_EXT
from webui.sources import get_file_audio


def history_choices():
    return gr.Dropdown(choices=history.entries(), value=None)


def source_link(job: dict, markdown: bool = True) -> str:
    """' · ▶ YouTube ↗' linking to where the audio came from, if it was a link."""
    url = job.get("source")
    if not url:
        return ""
    if markdown:
        return f" · [▶ source ↗]({url})"
    return f' · <a href="{html.escape(url, quote=True)}" target="_blank" rel="noopener">▶ source ↗</a>'


def history_preview(folder):
    if not folder:
        return ""
    meta = history.load(Path(folder))
    job, segs = meta["job"], meta["segments"]
    text = (Path(folder) / "transcript.txt").read_text(encoding="utf-8")
    state = {"done": "✅ finished", "stopped": f"⏸ stopped at {fmt_dur(meta['at'])}",
             "running": f"⏸ interrupted at {fmt_dur(meta['at'])}"}[meta["status"]]
    if folder in LIVE:
        state = "⏳ being transcribed now"
    fixed = " · has AI-fixed text" if (Path(folder) / "corrected.txt").exists() else ""
    return (f"**{job['title']}** · {job['model']} · {state} · {len(segs)} pieces{fixed}"
            f"{source_link(job)}\n\n> {text[:400]}{'…' if len(text) > 400 else ''}")


# What opening something in the dashboard can change; see opened().
OPEN_FIELDS = ("pages", "audio", "title", "language", "transcript", "txt_btn", "srt_btn", "srt_view",
               "corrected", "status", "partial", "segs", "entry", "resume_btn", "resume_top",
               "video_src", "youtube_url", "source", "pbar", "follow", "follow_timer")


def opened(**values) -> tuple:
    """Values for the OPEN_FIELDS outputs; fields not given stay as they are."""
    assert set(values) <= set(OPEN_FIELDS), set(values) - set(OPEN_FIELDS)
    return tuple(values.get(name, gr.skip()) for name in OPEN_FIELDS)


def history_open(folder, switch_page=True):
    """Load a saved run into the Transcribe dashboard. An unfinished one can be Resumed; one that
    is being transcribed right now (e.g. by the Batch) is followed live instead."""
    if not folder:
        raise gr.Error("Pick an entry first.")
    entry = Path(folder)
    meta = history.load(entry)
    job, segs = meta["job"], meta["segments"]
    if not Path(job["audio"]).exists():
        raise gr.Error("This entry's audio file is missing.")
    running = folder in LIVE
    text = (entry / "transcript.txt").read_text(encoding="utf-8")
    if running:
        segs = list(LIVE[folder]["segments"])
        text = to_text(segs)
    fixed_path = entry / "corrected.txt"
    fixed = fixed_path.read_text(encoding="utf-8") if fixed_path.exists() else ""
    txt, srt_path = out_file(job["title"], ".txt"), out_file(job["title"], ".srt")
    txt.write_text(text, encoding="utf-8")
    srt = to_srt(retime(text, segs)) if segs else ""
    srt_path.write_text(srt, encoding="utf-8")
    partial = None
    title = job["title"]
    if running:
        note = f"📂 “{title}” is being transcribed right now — following it live{source_link(job)}"
        progress = live_bar(folder)
    elif meta["status"] == "done":
        note = f"📂 Opened “{title}”{source_link(job)}"
        progress = bar(f"✅ “{html.escape(title)}” — finished{source_link(job, False)}", 1, done=True)
    else:
        partial = {"job": job, "entry": folder, "segments": segs, "at": meta["at"]}
        note = (f"📂 Opened “{title}” — stopped at {fmt_dur(meta['at'])}: press ⏯ Resume to "
                f"continue{source_link(job)}")
        progress = bar(f"⏸ “{html.escape(title)}” — stopped at {fmt_dur(meta['at'])}, ⏯ Resume continues"
                       f"{source_link(job, False)}", done=True)
    return opened(
        pages=gr.Tabs(selected="main") if switch_page else gr.skip(),
        audio=job["audio"] if switch_page else gr.skip(), title=title,
        language=language_of(job["model"]), transcript=text, txt_btn=file_button(txt),
        srt_btn=file_button(srt_path), srt_view=srt, corrected=fixed, status=note, partial=partial,
        segs=segs, entry=folder, resume_btn=gr.Button(visible=bool(partial)),
        resume_top=gr.Button(visible=bool(partial)), video_src=job.get("video"),
        source=job.get("source"), pbar=progress, follow=folder if running else None,
        follow_timer=gr.Timer(active=running))


def live_bar(folder):
    live = LIVE[folder]
    if live.get("loading"):
        return bar("⏳ Loading the model to transcribe this — the first time takes up to a minute…")
    frac, sec = live["frac"], live["audio_sec"]
    return bar(f"⏳ Transcribing in the background… {frac:.0%} · {fmt_dur(frac * sec)} of {fmt_dur(sec)}",
               frac)


def follow_tick(follow):
    """Timer: refresh a dashboard that follows a running transcription; load the saved result
    when it ends."""
    if not follow:
        return opened(follow_timer=gr.Timer(active=False))
    if follow not in LIVE:  # finished or stopped
        return history_open(follow, switch_page=False)
    segs = list(LIVE[follow]["segments"])
    return opened(transcript=to_text(segs), srt_view=to_srt(segs), segs=segs, pbar=live_bar(follow))


def stop_following():
    return None, gr.Timer(active=False)


def open_queue_item(queue, evt: gr.SelectData):
    """A click on a Batch queue row: open that item in the Transcribe dashboard."""
    item = queue[evt.index[0]]
    if item.get("entry") and Path(item["entry"], "meta.json").exists():
        return history_open(item["entry"])
    if item["kind"] == "link":  # not transcribed yet: ready to fetch and run on its own
        return opened(pages=gr.Tabs(selected="main"), youtube_url=item["src"], follow=None,
                      follow_timer=gr.Timer(active=False),
                      status=("🔗 Not transcribed yet — the link is in the YouTube box: press "
                              "**Audio only** (or **Video + audio**), then Transcribe."))
    audio, title, _, video, source = get_file_audio(item["src"],
                                                    Path(item["src"]).suffix.lower() in VIDEO_EXT)
    return opened(pages=gr.Tabs(selected="main"), audio=audio, title=title, video_src=video,
                  source=source, follow=None, follow_timer=gr.Timer(active=False),
                  status="🎞️ Not transcribed yet — the file is loaded: press Transcribe.")


def history_delete(folder):
    if folder:
        history.delete(Path(folder))
    return history_choices(), ""
