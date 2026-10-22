"""Export: .vtt, .docx, and the video with subtitles burned in."""
import shutil
import time
from pathlib import Path

import gradio as gr

from stt.export import burn_subtitles, to_vtt, write_docx
from stt.retime import retime
from stt.transcribe import to_srt
from webui.common import bar, out_file, time_left, WORK_DIR


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
