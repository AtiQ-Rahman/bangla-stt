"""Chapters for a long recording: the transcript cut into parts at pauses, each given a title and
a short summary by the local LLM. The "MM:SS Title" list works as YouTube chapters."""
import re
from typing import Iterator, NamedTuple

from stt.correct import _generate
from stt.transcribe import Segment

PROMPT = """Below is part {i} of {n} of the transcript of a {lang} recording (from speech \
recognition, so it may have mistakes). Reply in {lang} with exactly two lines and nothing else:
TITLE: a short chapter title for this part (at most 6 words)
SUMMARY: one or two sentences saying what this part is about
Use only what is said in this part: no names, dates or facts that are not in it.

Transcript:
{text}"""

CLAUDE_PROMPT = """Below is a {lang} transcript with times, from speech recognition (it may have \
mistakes). Split it into chapters by topic (at least 3, the first at 00:00) and reply in {lang} \
with:
1. A list for a YouTube description, one chapter per line: "MM:SS Title" (H:MM:SS past an hour).
2. Under it, each chapter's title with a one or two sentence summary.

Transcript:
{text}"""


class Chapter(NamedTuple):
    start: float
    title: str
    summary: str


def stamp(sec: float) -> str:
    """YouTube's chapter time: 05:12, or 1:05:12 past an hour."""
    s = int(sec)
    return f"{s // 3600}:{s % 3600 // 60:02}:{s % 60:02}" if s >= 3600 else f"{s // 60:02}:{s % 60:02}"


def parts(segments: list[Segment], target: int = 8, min_sec: float = 60.0) -> list[list[Segment]]:
    """About `target` parts (each at least min_sec), each cut at the longest pause near where it
    should end."""
    segments = [s for s in segments if s.text.strip()]
    if not segments:
        return []
    length = max(min_sec, (segments[-1].end - segments[0].start) / target)
    out, cur = [], []
    for i, seg in enumerate(segments):
        cur.append(seg)
        span = seg.end - cur[0].start
        if i + 1 < len(segments) and span >= 0.75 * length:
            pause = segments[i + 1].start - seg.end
            # the longest pause from here until the part would be 1.25x too long
            later = [segments[j + 1].start - segments[j].end for j in range(i + 1, len(segments) - 1)
                     if segments[j].end - cur[0].start <= 1.25 * length]
            if pause >= max(later, default=0.0) or span >= 1.25 * length:
                out.append(cur)
                cur = []
    if cur:
        if out and cur[-1].end - cur[0].start < min_sec / 2:  # a short tail joins the last part
            out[-1] += cur
        else:
            out.append(cur)
    return out


_LABEL = re.compile(r"^[\W_]*[^\W\d_][^:：\n]{0,20}[:：]\s*")  # "TITLE:", "শিরোনাম:", "**Summary**:"


def _parse(reply: str, fallback: str) -> tuple[str, str]:
    """(title, summary) from the reply: its first line and the rest, without labels (the model
    may translate "TITLE:" too)."""
    lines = [_LABEL.sub("", line).strip() for line in reply.splitlines()]
    lines = [line.strip("*\"'“” ") for line in lines if line.strip("*\"'“” ")]
    if not lines:
        return fallback, ""
    title, summary = lines[0], " ".join(lines[1:])
    if not summary and len(title.split()) > 10:  # only a summary came back
        title, summary = fallback, title
    return title.rstrip("।.")[:80], summary


def chapters(segments: list[Segment], model: str, language: str = "Bangla",
             done: list[Chapter] | None = None) -> Iterator[tuple[list[Chapter], int]]:
    """Yields (chapters so far, number of parts) after each part; `done` resumes."""
    pieces = parts(segments)
    out = list(done or [])
    for i, piece in enumerate(pieces[len(out):], len(out) + 1):
        text = " ".join(s.text.strip() for s in piece)
        words = text.split()
        reply = "".join(_generate(PROMPT.format(i=i, n=len(pieces), lang=language,
                                                text=" ".join(words[:1500])), model))
        title, summary = _parse(reply, " ".join(words[:5]) + " …")
        out.append(Chapter(0.0 if i == 1 else piece[0].start, title, summary))
        yield out, len(pieces)


def to_youtube(chapters: list[Chapter]) -> str:
    return "\n".join(f"{stamp(c.start)} {c.title}" for c in chapters)


def to_text(chapters: list[Chapter]) -> str:
    """The YouTube list, then each chapter with its summary."""
    details = "\n\n".join(f"{stamp(c.start)} {c.title}\n{c.summary}".strip() for c in chapters)
    return f"{to_youtube(chapters)}\n\n— Summaries —\n\n{details}" if chapters else ""


def timed_text(segments: list[Segment], every: float = 30.0) -> str:
    """The transcript with a [MM:SS] mark about every `every` seconds (for Claude's prompt)."""
    lines, last = [], None
    for seg in segments:
        if not seg.text.strip():
            continue
        if last is None or seg.start - last >= every:
            lines.append(f"[{stamp(seg.start)}] {seg.text.strip()}")
            last = seg.start
        else:
            lines[-1] += " " + seg.text.strip()
    return "\n".join(lines)
