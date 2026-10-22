"""The Chapters tab: chapter list and summaries by the local AI, or a prompt for Claude."""
import time
from pathlib import Path

import gradio as gr

from stt import history
from stt.chapters import chapters, CLAUDE_PROMPT, timed_text, to_text
from stt.correct import is_loaded as llm_is_loaded, list_models
from stt.retime import retime
from stt.worker import WORKER
from webui.common import bar, file_button, LLM_RAM_GB, mem_gb, no_file, out_file, runtime, time_left

FILE = "chapters.txt"  # in the History entry


def _timed(text, segs):
    """The transcript's segments, with the user's edits of the text."""
    if not segs:
        raise gr.Error("Transcribe something first (chapters need the word times).")
    return retime(text, segs) if (text or "").strip() else segs


def make_chapters(text, segs, llm, language, title, entry):
    """Chapters by the local LLM, part by part; Stop keeps the parts done so far."""
    timed = _timed(text, segs)
    if not list_models():
        raise gr.Error("Ollama is not running. Start it with `docker start ollama` (see README).")
    if runtime.active:
        raise gr.Error("Wait for the transcription to finish (or press Stop) first.")
    WORKER.kill()  # the speech models and the LLM together don't fit in 16 GB RAM
    avail, _ = mem_gb()
    if avail < LLM_RAM_GB and not llm_is_loaded(llm):
        raise gr.Error(f"Not enough free RAM for the LLM ({avail:.1f} GB free, ~{LLM_RAM_GB:.0f} GB needed).")
    yield gr.skip(), no_file(), bar("⏳ Loading the AI model…")
    out, first = "", None
    try:
        for found, total in chapters(timed, llm, language):
            frac = len(found) / total
            first = first or (time.monotonic(), frac)
            out = to_text(found)
            if entry:  # kept as it grows, so a stopped run's chapters are there when reopened
                history.save_text(Path(entry), FILE, out)
            yield out, gr.skip(), bar(f"⏳ Chapter {len(found)} of {total}{time_left(first, frac)}", frac)
    except Exception as e:
        raise gr.Error(f"LLM failed: {e}")
    path = out_file(title, " chapters.txt")
    path.write_text(out, encoding="utf-8")
    yield out, file_button(path), bar("✅ Chapters ready — the list at the top can go in a YouTube "
                                      "description", 1, done=True)


def saved_chapters(entry):
    """Opening the tab: the chapters made earlier for this transcript (if any)."""
    path = Path(entry, FILE) if entry else None
    return path.read_text(encoding="utf-8") if path and path.exists() else ""


def claude_prompt(text, segs, language):
    return gr.Textbox(value=CLAUDE_PROMPT.format(lang=language, text=timed_text(_timed(text, segs))),
                      visible=True)
