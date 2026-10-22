"""Chapters: parts cut at pauses, titles/summaries parsed from the LLM, YouTube times."""
from stt import chapters
from stt.chapters import Chapter, parts, stamp, to_text, to_youtube
from stt.transcribe import Segment


def talk(minutes: int, pause_every: int = 0) -> list[Segment]:
    """10 s segments with 1 s gaps, and a 5 s pause every `pause_every` segments."""
    out, t = [], 0.0
    for i in range(minutes * 6):
        out.append(Segment(t, t + 10, f"কথা {i}", []))
        t += 11 + (5 if pause_every and i % pause_every == pause_every - 1 else 0)
    return out


def test_parts_cut_at_the_long_pauses():
    segs = talk(40, pause_every=25)  # a long pause every ~5 min
    pieces = parts(segs, target=8)
    assert 6 <= len(pieces) <= 10
    assert sum(len(p) for p in pieces) == len(segs)
    for p in pieces[:-1]:  # each ends just before a long pause
        nxt = segs[segs.index(p[-1]) + 1]
        assert nxt.start - p[-1].end == 6
    assert parts(talk(1)) == [talk(1)]  # short: one part


def test_stamps_and_text():
    assert stamp(0) == "00:00" and stamp(312.7) == "05:12" and stamp(3912) == "1:05:12"
    chs = [Chapter(0, "শুরু", "পরিচয়।"), Chapter(312, "রাজনীতি", "")]
    assert to_youtube(chs) == "00:00 শুরু\n05:12 রাজনীতি"
    assert to_text(chs).startswith("00:00 শুরু\n05:12 রাজনীতি\n\n— Summaries —\n\n00:00 শুরু\nপরিচয়।")


def test_chapters_from_llm_replies(monkeypatch):
    replies = iter(["TITLE: **শুরুর কথা**\nSUMMARY: বক্তা নিজের\nপরিচয় দেন।", ""])
    monkeypatch.setattr(chapters, "_generate", lambda prompt, model: iter([next(replies)]))
    segs = talk(2, pause_every=6)
    *_, (found, total) = chapters.chapters(segs, "m")
    assert total == len(found) == 2
    assert found[0] == Chapter(0.0, "শুরুর কথা", "বক্তা নিজের পরিচয় দেন।")
    assert found[1].title.endswith("…") and found[1].start > 0


def test_parse_any_labels():
    assert chapters._parse("শিরোনাম: কমিশনারের বদলি\nসারসংক্ষেপ: তাঁকে সরানো হয়।", "x") == (
        "কমিশনারের বদলি", "তাঁকে সরানো হয়।")
    assert chapters._parse("**Title:** Intro\n\n**Summary:** We meet: Bob.", "x") == ("Intro", "We meet: Bob.")
    assert chapters._parse("", "fallback …") == ("fallback …", "")
