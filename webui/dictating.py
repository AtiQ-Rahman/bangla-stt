"""The Live tab: dictation from the microphone (stt.dictation), transcribed by the worker; each
recording is saved in History like any other transcription."""
import threading
import time
from datetime import datetime

import gradio as gr
import numpy as np
import soundfile as sf

from stt import history
from stt.audio import cut_clip, SAMPLE_RATE
from stt.dictation import Dictation
from stt.export import bangla_digits
from stt.transcribe import Segment, Word
from stt.worker import WORKER
from webui.common import check_ram, file_button, language_of, no_file, out_file, runtime, WORK_DIR
from webui.transcribing import make_job

SESSIONS: dict[str, dict] = {}  # browser session -> {"dictation", "job", "before"}
MIC = "3. Microphone — press Record, Stop when done"


def _mic(ready: bool):
    """The microphone, on or off (off while the model loads: then it is an empty box, so say why)."""
    return gr.Audio(interactive=ready, label=MIC if ready else "3. Microphone — ⏳ waits for the model…")


def _transcriber(job: dict):
    """transcribe(wav, offset) for Dictation: one phrase through the worker."""
    digits = job["bn_digits"] and language_of(job["model"]) == "Bangla"

    def fix(text):
        return bangla_digits(text) if digits else text

    def transcribe(wav: np.ndarray, offset: float) -> list[Segment]:
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        path = WORK_DIR / f"live-{time.monotonic_ns()}.wav"
        sf.write(path, wav, SAMPLE_RATE)
        try:
            out = []
            for kind, value in WORKER.transcribe(path, 0.0, None, job["model"], job["quantize"],
                                                 job["use_lm"], job["words"], 0.0):
                if kind == "piece" and value[2]:
                    s = value[2]
                    out.append(Segment(s.start + offset, s.end + offset, fix(s.text),
                                       [Word(w.start + offset, w.end + offset, fix(w.text), w.speaker)
                                        for w in s.words]))
            return out
        finally:
            path.unlink(missing_ok=True)
    return transcribe


def _shown(session: dict) -> str:
    d = session["dictation"]
    text = "\n\n".join(t for t in (session["before"], d.text()) if t)
    return text + (" …" if d.pending else "")


def _in_memory() -> str:
    loaded = WORKER.loaded if WORKER.alive() else None
    return f"In memory: **{loaded[0]}**." if loaded else "No model in memory yet."


def live_selected(model_name, quantize):
    """Language or model picked: Record is on only if that model is the one in memory.
    Returns (status, mic)."""
    if WORKER.is_loaded(model_name, quantize):
        return f"✅ **{model_name}** is ready — press Record and speak.", _mic(True)
    return (f"⚪ **{model_name}** is selected but not loaded — press **▶ Load model**. "
            f"{_in_memory()}"), _mic(False)


def live_ready(model_name, quantize, use_lm):
    """Load the selected model (▶ Load model, or the Live tab opened); Record is off until it
    is ready, so the text appears as the user speaks. The language and model can't be changed
    while it loads (the status would name the wrong one). Returns (status, mic, language, model)."""
    free = gr.Radio(interactive=True), gr.Radio(interactive=True)
    if runtime.active:  # a recording is being finished, or a transcription runs
        yield ("⏳ Busy: the last recording is still being written (or a transcription runs) — "
               f"press ▶ Load model again in a moment. {_in_memory()}"), _mic(False), *free
        return
    if not WORKER.is_loaded(model_name, quantize):
        yield (f"⏳ Loading **{model_name}** (about a minute; the first time it may download)… "
               "Record turns on when it is ready."), _mic(False), gr.Radio(interactive=False), \
            gr.Radio(interactive=False)
        try:
            job = make_job("live", "", model_name, quantize, use_lm, "", "", "", False, use_learned=False)
            check_ram(job["model"], job["quantize"], job["use_lm"])
            _transcriber(job)(np.zeros(SAMPLE_RATE // 2, np.float32), 0.0)
        except Exception as e:
            yield f"⚠️ **{model_name}** could not be loaded: {e}", _mic(False), *free
            return
        runtime.last_used = time.monotonic()
    yield *live_selected(model_name, quantize), *free


def live_start(text, model_name, quantize, use_lm, hotwords, digits, use_learned, request: gr.Request):
    """Recording started: get the model ready while the user starts speaking."""
    if runtime.active:
        raise gr.Error("Wait for the transcription to finish (or press Stop) first.")
    job = make_job("live", "", model_name, quantize, use_lm, hotwords, "", "", digits,
                   use_learned=use_learned)
    check_ram(job["model"], job["quantize"], job["use_lm"])
    runtime.active += 1  # the idle unloader must not take the model away mid-sentence
    transcribe = _transcriber(job)
    if not WORKER.is_loaded(job["model"], job["quantize"]):  # a phrase spoken meanwhile waits for it
        threading.Thread(target=transcribe, args=(np.zeros(SAMPLE_RATE // 2, np.float32), 0.0),
                         daemon=True).start()
    SESSIONS[request.session_hash] = {"dictation": Dictation(transcribe), "job": job,
                                      "before": (text or "").strip()}
    return _listening(job)


def _listening(job: dict) -> str:
    model = job["model"].split(" - ")[0]
    if not WORKER.is_loaded(job["model"], job["quantize"]):
        return f"🔴 Recording with **{model}** — it is still loading (about a minute), the text will catch up."
    return f"🔴 Recording with **{model}** — the text appears a moment after each pause."


def live_chunk(chunk, request: gr.Request):
    session = SESSIONS.get(request.session_hash)
    if not session or chunk is None:
        return gr.skip(), gr.skip()
    sr, data = chunk
    session["dictation"].feed(sr, data)
    return _shown(session), _listening(session["job"])


def _end(session_hash: str):
    """Finish a recording: write the rest, and keep the recording and its text in History.
    Returns (session, History entry or None)."""
    session = SESSIONS.pop(session_hash, None)
    if not session:
        return None, None
    d, job = session["dictation"], session["job"]
    try:
        d.finish()
    finally:
        runtime.active -= 1
        runtime.last_used = time.monotonic()
    if not d.segments:
        return session, None
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    wav = WORK_DIR / f"dictation-{session_hash}.wav"
    sf.write(wav, d.audio, SAMPLE_RATE)
    job.update(audio=str(cut_clip(wav, wav.with_suffix(".mp3"))),
               title=f"Dictation {datetime.now():%Y-%m-%d %H-%M}")
    wav.unlink(missing_ok=True)
    entry = history.create(job)
    history.save(entry, job, d.segments, len(d.audio) / SAMPLE_RATE, "done")
    return session, entry


def live_stop(request: gr.Request):
    session, entry = _end(request.session_hash)
    empty = gr.Audio(value=None)  # the Record button again, not a player of the recording
    if not session:
        return (gr.skip(),) * 5 + (empty,)
    text = _shown(session)
    if session["dictation"].error:
        gr.Warning(f"Some speech could not be transcribed: {session['dictation'].error}")
    if not entry:
        return text, "🔇 Nothing was heard — is the right microphone selected?", gr.skip(), gr.skip(), gr.skip(), empty
    path = out_file(session["job"]["title"], ".txt")
    path.write_text(text, encoding="utf-8")
    return (text, "✅ Done — saved in 🕘 History. Press Record to go on, or 📂 Open in Transcribe.",
            file_button(path), str(entry), gr.Button(visible=True), empty)


def live_abandoned(request: gr.Request):
    """The page was closed while recording: still keep what was said."""
    _end(request.session_hash)


def live_stopping():
    return "⏳ Stopped — writing your last words (longer with Whisper)…"


def live_clear():
    return "", "", no_file(), None, gr.Button(visible=False)
