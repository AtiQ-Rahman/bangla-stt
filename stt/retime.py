"""Give edited or AI-fixed text the timing of the transcript it came from."""
import difflib
import re

from stt.transcribe import Segment, Word

_PUNCT = re.compile(r"^[\W_]+|[\W_]+$")


def _key(word: str) -> str:
    return _PUNCT.sub("", word).lower()


def retime(text: str, segments: list[Segment]) -> list[Segment]:
    """Segments carrying `text`, timed from the original words they line up with.

    Words are aligned with difflib: unchanged words keep their own time, a run of replaced
    (re-spelled, joined, split) words shares the time span of the words it replaced, and added words
    sit at the end of the word before them. Words are grouped back into the original
    segments, so subtitle cues keep their places.
    """
    timed: list[tuple[int, Word]] = []  # (segment index, original word)
    for i, seg in enumerate(segments):
        if seg.words:
            timed += [(i, w) for w in seg.words]
        else:  # no word times: spread the segment's words evenly over it
            parts = seg.text.split() or [seg.text]
            step = (seg.end - seg.start) / len(parts)
            timed += [(i, Word(seg.start + k * step, seg.start + (k + 1) * step, p))
                      for k, p in enumerate(parts)]
    if not timed:
        return []
    new = text.split()
    placed: list[tuple[int, Word]] = []
    ops = difflib.SequenceMatcher(None, [_key(w.text) for _, w in timed], [_key(w) for w in new],
                                  autojunk=False).get_opcodes()
    for op, i1, i2, j1, j2 in ops:
        if op == "equal":  # same words: each keeps its own time
            placed += [(timed[i1 + k][0], Word(timed[i1 + k][1].start, timed[i1 + k][1].end, w))
                       for k, w in enumerate(new[j1:j2])]
        elif op == "replace" and i2 > i1:
            seg_i, first = timed[i1]
            start, end = first.start, timed[i2 - 1][1].end
            step = (end - start) / (j2 - j1) if j2 > j1 else 0
            for k, w in enumerate(new[j1:j2]):
                placed.append((seg_i, Word(start + k * step, start + (k + 1) * step, w)))
        elif op == "insert":
            prev_seg, prev = placed[-1] if placed else timed[0]
            at = prev.end if placed else prev.start
            placed += [(prev_seg, Word(at, at, w)) for w in new[j1:j2]]
    out: list[Segment] = []
    for seg_i, word in placed:
        if out and out[-1].start == segments[seg_i].start:
            out[-1].words.append(word)
        else:
            out.append(Segment(segments[seg_i].start, segments[seg_i].end, "", [word]))
    for seg in out:
        seg.text = " ".join(w.text for w in seg.words)
        seg.start, seg.end = seg.words[0].start, max(w.end for w in seg.words)
    return out
