"""Tell Bangla from English speech with Whisper tiny (75 MB, ~0.3 s per check on CPU)."""
from functools import lru_cache
from pathlib import Path

from stt import cached_first
from stt.audio import duration, load_audio

CODES = {"bn": "Bangla", "en": "English"}


@lru_cache(maxsize=1)
def _model():
    from faster_whisper import WhisperModel

    return cached_first(lambda **c: WhisperModel("tiny", device="cpu", compute_type="int8", **c))


def detect(path: str | Path, start: float | None = None, end: float | None = None) -> tuple[str, float]:
    """("Bangla" or "English", confidence 0..1) for the speech in path (start..end if given).

    Listens to 30 s, skipping a little of the beginning, where intros and music often are.
    """
    length = (end or duration(path) or 0) - (start or 0)
    skip = min(20.0, max(0.0, length - 30) / 2)
    begin = (start or 0) + skip
    wav = load_audio(path, begin, begin + 30)
    _, _, probs = _model().detect_language(wav)
    probs = dict(probs)
    bn, en = probs.get("bn", 0.0), probs.get("en", 0.0)
    best = "bn" if bn >= en else "en"
    return CODES[best], max(bn, en) / ((bn + en) or 1)
