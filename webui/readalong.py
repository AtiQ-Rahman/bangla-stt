"""Read along: the transcript next to the audio; click a word to hear it, the word playing is lit.

The page script in webui.look (READALONG_JS) does the clicking and highlighting in the browser.
"""
import html
import urllib.parse
from pathlib import Path

from stt import history
from stt.retime import retime
from webui.common import fmt_dur

EMPTY = '<div class="ra-empty">Transcribe something, or open it from History, to read along here.</div>'


def readalong(text, segs, entry):
    """The read-along view: an audio player and the (edited) transcript with word times."""
    if not segs or not entry or not Path(entry, "meta.json").exists():
        return EMPTY
    audio = history.load(Path(entry))["job"]["audio"]
    timed = retime(text, segs) if (text or "").strip() else segs
    paragraphs = []
    for seg in timed:
        words = " ".join(f'<span class="w" data-s="{w.start:.2f}">{html.escape(w.text.strip())}</span>'
                         for w in seg.words) or html.escape(seg.text)
        paragraphs.append(f'<p><span class="ts" data-s="{seg.start:.2f}">{fmt_dur(seg.start)}</span>'
                          f"{words}</p>")
    src = "/gradio_api/file=" + urllib.parse.quote(audio)
    return (f'<div class="readalong"><audio controls preload="metadata" src="{src}"></audio>'
            f'<div class="ra-hint">Click any word to play from there.</div>'
            f'<div class="ra-text">{"".join(paragraphs)}</div></div>')
