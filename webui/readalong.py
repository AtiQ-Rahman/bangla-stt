"""Read along: the transcript next to the audio; click a word to hear it, the word playing is lit,
and the text can be corrected right there while listening.

The page script in webui.look (READALONG_JS) does the clicking, highlighting, playback controls
and editing in the browser; saving sends the edited text to save_readalong().
"""
import html
import urllib.parse
from pathlib import Path

import gradio as gr

from stt import history
from stt.retime import retime
from stt.transcribe import to_text
from webui.common import fmt_dur

EMPTY = '<div class="ra-empty">Transcribe something, or open it from History, to read along here.</div>'


def readalong(text, segs, entry):
    """The read-along view: an audio player and the (edited) transcript with word times."""
    if not segs or not entry or not Path(entry, "meta.json").exists():
        return EMPTY
    audio = history.load(Path(entry))["job"]["audio"]
    timed = retime(text, segs) if (text or "").strip() else segs
    paragraphs = []
    speaker = None
    for seg in timed:
        parts = []
        for w in seg.words:
            if w.speaker is not None and w.speaker != speaker:  # a new speaker's turn
                parts.append(f'<span class="spk">Speaker {w.speaker}</span>')
                speaker = w.speaker
            parts.append(f'<span class="w" data-s="{w.start:.2f}">{html.escape(w.text.strip())}</span>')
        words = " ".join(parts) or html.escape(seg.text)
        paragraphs.append(f'<p><span class="ts" data-s="{seg.start:.2f}">{fmt_dur(seg.start)}</span>'
                          f"{words}</p>")
    src = "/gradio_api/file=" + urllib.parse.quote(audio)
    return (f'<div class="readalong" data-follow="1"><audio controls preload="metadata" src="{src}"></audio>'
            f"{TOOLBAR}"
            '<div class="ra-hint"><span class="read-only">Click any word to play from there.</span>'
            '<span class="edit-only">Editing: type to fix words. <b>Alt+click</b> a word to hear it, '
            '<b>Ctrl+Space</b> plays/pauses from the cursor, <b>Ctrl+Enter</b> saves, <b>Esc</b> '
            "cancels. Timings are kept.</span></div>"
            f'<div class="ra-text">{"".join(paragraphs)}</div></div>')


TOOLBAR = """<div class="ra-bar">
  <button data-act="back" title="Back 5 seconds">⏪ 5s</button>
  <button data-act="play" title="Play / pause">⏯</button>
  <button data-act="fwd" title="Forward 5 seconds">5s ⏩</button>
  <label title="Playback speed">Speed <select data-act="speed">
    <option>0.75</option><option selected>1</option><option>1.25</option><option>1.5</option>
    <option>2</option></select></label>
  <label title="Keep the word being played in view"><input type="checkbox" data-act="follow" checked> Follow</label>
  <span class="ra-spacer"></span>
  <button data-act="edit" class="read-only">✏️ Edit text</button>
  <button data-act="save" class="edit-only ra-primary">💾 Save</button>
  <button data-act="cancel" class="edit-only">✕ Cancel</button>
</div>"""


def save_readalong(edited, segs, title, entry):
    """Text edited in Read along becomes the transcript: the box, downloads, subtitles, view."""
    from webui.transcribing import edited_subtitles, edited_text

    text = (edited or "").strip()
    if not text:
        return (gr.skip(),) * 5
    if segs:  # the usual layout (paragraphs, speaker turns), with the user's words
        text = to_text(retime(text, segs))
    srt_button, srt = edited_subtitles(text, segs, title, ".srt")
    return text, edited_text(text, title, entry, "transcript.txt"), srt_button, srt, readalong(text, segs, entry)
