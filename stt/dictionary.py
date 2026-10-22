"""The user's own words, learned from their corrections, used as hotwords in later transcriptions.

When a transcript is edited, words the user changed (e.g. a misheard name fixed to "রাজশাহী") are
remembered per language in ~/.config/bangla-stt/dictionary.json (STT_CONFIG_DIR to move it),
counted once per transcript, most-corrected first.
"""
import difflib
import json
import os
import re
import unicodedata
from pathlib import Path

from stt.transcribe import Segment

FILE = Path(os.environ.get("STT_CONFIG_DIR", Path.home() / ".config" / "bangla-stt")) / "dictionary.json"
LIMIT = 50  # hotwords used per run; many slow the decoder down
_SPEAKER = re.compile(r"(?m)^\s*Speaker \d+:\s*")


def _clean(word: str) -> str:
    """The word without punctuation around it (not with \\W: Bangla vowel signs count as \\W)."""
    word = unicodedata.normalize("NFC", word)
    edge = lambda ch: unicodedata.category(ch)[0] in "PSZ"
    while word and edge(word[0]):
        word = word[1:]
    while word and edge(word[-1]):
        word = word[:-1]
    return word


def _load() -> dict:
    """{"words": {language: {word: count}}, "seen": ["<transcript>|<word>", ...]}"""
    try:
        data = json.loads(FILE.read_text(encoding="utf-8"))
        return {"words": {lang: dict(ws) for lang, ws in data.get("words", {}).items()},
                "seen": list(data.get("seen", []))}
    except (OSError, ValueError, TypeError, AttributeError):
        return {"words": {}, "seen": []}


def _save(data: dict):
    FILE.parent.mkdir(parents=True, exist_ok=True)
    FILE.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def language_of(word: str) -> str:
    return "Bangla" if any("ঀ" <= ch <= "৿" for ch in word) else "English"


def corrections(segments: list[Segment], edited: str) -> list[str]:
    """Words of `edited` that replace different words of the transcript (not just punctuation or
    capitals, and not numbers): what the user corrected."""
    original = [_clean(w.text) for seg in segments for w in seg.words]
    new = [_clean(w) for w in _SPEAKER.sub("", edited or "").split()]
    out = []
    ops = difflib.SequenceMatcher(None, [w.lower() for w in original], [w.lower() for w in new],
                                  autojunk=False).get_opcodes()
    for op, i1, i2, j1, j2 in ops:
        if op != "replace":
            continue
        replaced = {w.lower() for w in original[i1:i2]}
        for word in new[j1:j2]:
            if len(word) >= 2 and not word.isdigit() and word.lower() not in replaced and word not in out:
                out.append(word)
    return out


def learn(segments: list[Segment], edited: str, key: str = "") -> list[str]:
    """Remember the corrections in an edit of the transcript `key` (e.g. its History folder);
    returns the words newly learned from it. Saving the same edit again counts nothing twice."""
    found = corrections(segments, edited)
    if not found:
        return []
    data = _load()
    seen = set(data["seen"])
    new = [w for w in found if f"{key}|{w}" not in seen]
    for word in new:
        bucket = data["words"].setdefault(language_of(word), {})
        bucket[word] = bucket.get(word, 0) + 1
        data["seen"].append(f"{key}|{word}")
    if new:
        data["seen"] = data["seen"][-5000:]
        _save(data)
    return new


def words(language: str, limit: int = LIMIT) -> list[str]:
    """The language's learned words, most corrected first."""
    bucket = _load()["words"].get(language, {})
    return sorted(bucket, key=lambda w: -bucket[w])[:limit]


def set_words(language: str, text: str):
    """Replace the language's words with a comma/line separated list (keeping known counts)."""
    data = _load()
    old = data["words"].get(language, {})
    listed = [_clean(w) for w in re.split(r"[,\n]", text or "") if _clean(w)]
    data["words"][language] = {w: old.get(w, 1) for w in dict.fromkeys(listed)}
    _save(data)
