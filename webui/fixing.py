"""The local AI fix (Ollama), piece by piece, with Stop/Resume."""
import time
from pathlib import Path

import gradio as gr

from stt import history
from stt.correct import correct, DEFAULT_LLM, is_loaded as llm_is_loaded, list_models
from stt.retime import retime
from stt.transcribe import to_srt, unload
from webui.common import bar, file_button, LLM_RAM_GB, mem_gb, no_file, out_file, runtime, time_left


def run_correct(text, llm, language, segs, title, entry, fix_partial, resuming=False):
    """Fix the transcript with the local LLM, piece by piece; Stop keeps the finished pieces."""
    if not (text or "").strip():
        raise gr.Error("Transcribe something first.")
    if not list_models():
        raise gr.Error("Ollama is not running. Start it with `docker start ollama` (see README).")
    if runtime.active:
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
