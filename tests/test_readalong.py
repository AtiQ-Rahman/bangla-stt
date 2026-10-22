"""The read-along view: words carry their start time, edits keep the timing, text is escaped."""
from stt import history
from webui.readalong import EMPTY, readalong


def test_words_have_times_and_audio(tmp_path, monkeypatch, segments):
    monkeypatch.setattr(history, "HISTORY_DIR", tmp_path / "h")
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"x")
    job = {"audio": str(audio), "title": "t", "model": "m", "quantize": True, "use_lm": True,
           "words": [], "start": None, "end": None}
    entry = history.create(job)
    view = readalong("আমি ভাত খাই। <b>জন</b> প্রশাসন মন্ত্রণালয়", segments, str(entry))
    assert view.count('class="w"') == 6
    assert 'data-s="2.00">খাই।<' in view  # an edited word keeps its time
    text_part = view[view.index('class="ra-text"'):]
    assert "&lt;b&gt;" in text_part and "<b>" not in text_part  # typed text can't inject HTML
    assert "/gradio_api/file=" in view


def test_nothing_to_show():
    assert readalong("", None, None) == EMPTY
