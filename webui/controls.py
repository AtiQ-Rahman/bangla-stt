"""Which buttons can be pressed when, and clearing the results."""
import gradio as gr

from stt.transcribe import LANGUAGE_MODELS
from stt.worker import WORKER
from webui.common import bar, fmt_dur, no_file


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
    if kind in ("transcribe", "batch"):
        WORKER.cancel()  # the worker ends the job after the current piece
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
