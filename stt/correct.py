"""Clean up a raw transcript with a local LLM served by Ollama (https://ollama.com)."""
import difflib
import json
import os
import re
import urllib.request
from typing import Iterator

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
DEFAULT_LLM = "gemma3:4b"

PROMPTS = {}
PROMPTS["Bangla"] = """You fix Bangla speech-to-text output. Correct spelling mistakes and wrongly split \
or joined words, and add punctuation (। , ?). Keep every word the speaker said, in the same order. \
Never add, drop, summarise or translate words, and never use any script other than Bangla.
Reply with the corrected text only.

Example input:
আজকে ঢাকার আবহাওযা খুব গরম ছিলো বিকেলে হালকা বৃস্টি হতে পারে বলে জানিযেছে আবহাওয়া অধি দপ্তর
Example output:
আজকে ঢাকার আবহাওয়া খুব গরম ছিল। বিকেলে হালকা বৃষ্টি হতে পারে বলে জানিয়েছে আবহাওয়া অধিদপ্তর।

Input:
{text}
Output:
"""

PROMPTS["English"] = """You fix English speech-to-text output. Correct spelling, capital letters and \
punctuation, and join sentences that were broken in the middle. Keep every word the speaker said, \
in the same order. Never add, drop, summarise or translate words. Reply with the corrected text only.

Example input:
so we went to the the market in dhaka and it was really crowded we bought some mangos
Example output:
So we went to the the market in Dhaka, and it was really crowded. We bought some mangoes.

Input:
{text}
Output:
"""

PUNCT = "।॥,?!;:.'\"‘’“”-"


def list_models() -> list[str]:
    """Names of models installed in the local Ollama, or [] if Ollama is not running."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as r:
            return sorted(m["name"] for m in json.load(r)["models"])
    except OSError:
        return []


def _generate(prompt: str, model: str) -> Iterator[str]:
    # keep_alive: unload the model soon after we are done; RAM is tight next to the STT models.
    body = json.dumps({"model": model, "prompt": prompt, "stream": True, "keep_alive": "30s",
                       "options": {"temperature": 0}}).encode()
    req = urllib.request.Request(f"{OLLAMA_URL}/api/generate", body,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            msg = json.loads(line)
            if "error" in msg:
                raise RuntimeError(msg["error"])
            yield msg.get("response", "")


def _in_script(word: str, language: str) -> bool:
    """True if every letter in word belongs to the language's script (digits/punctuation are fine)."""
    if language == "Bangla":
        return all("\u0980" <= ch <= "\u09FF" for ch in word if ch.isalpha())
    return all(ch < "\u0250" for ch in word if ch.isalpha())  # Latin, incl. accented letters


def _similar(a: str, b: str) -> float:
    """How alike two strings read, ignoring capitalisation (a fix may capitalise)."""
    return difflib.SequenceMatcher(None, a.lower(), b.lower(), autojunk=False).ratio()


def _merge(original: str, fixed: str, language: str = "Bangla") -> tuple[str, int]:
    """Keep the LLM's spelling/punctuation fixes but never its added, dropped or reworded words.

    Words are aligned; a changed stretch is accepted only when it is in the language's script
    and reads close to
    the original (same words re-spelled, or split/joined differently). Everything else falls
    back to the original words. Returns (merged text, number of stretches reverted).
    """
    orig, new = original.split(), fixed.split()
    bare_orig, bare_new = [w.strip(PUNCT) for w in orig], [w.strip(PUNCT) for w in new]
    out, reverted = [], 0
    ops = difflib.SequenceMatcher(None, bare_orig, bare_new, autojunk=False).get_opcodes()
    for op, i1, i2, j1, j2 in ops:
        if op == "equal":
            out += new[j1:j2]
        elif op == "replace" and all(_in_script(w, language) for w in new[j1:j2]) and (
                _similar("".join(bare_orig[i1:i2]), "".join(bare_new[j1:j2])) >= 0.7):
            out += new[j1:j2]
        elif op == "insert" and not "".join(bare_new[j1:j2]):  # punctuation only
            out += new[j1:j2]
        else:
            out += orig[i1:i2]
            reverted += op != "insert" or bool("".join(bare_new[j1:j2]))
    return " ".join(out), reverted


def _chunks(words: list[str], size: int) -> Iterator[str]:
    """Pieces of about size words, stretched (by up to half) to end at a sentence end if any."""
    i = 0
    while i < len(words):
        j = min(i + size, len(words))
        for k in range(j, min(i + size * 3 // 2, len(words))):
            if words[k - 1].endswith((".", "?", "!", "।")):
                j = k
                break
        yield " ".join(words[i:j])
        i = j


def correct(text: str, model: str = DEFAULT_LLM, language: str = "Bangla",
            chunk_words: int = 80) -> Iterator[tuple[str, int]]:
    """Correct the transcript in pieces of about chunk_words words.

    Yields (text so far, changes reverted). Transcript lines are cut at arbitrary points
    mid-sentence, so they are joined first; the LLM places the sentence breaks. Each piece
    goes through _merge(), so the LLM can fix spelling but cannot change what was said.
    """
    words = text.split()
    done: list[str] = []
    reverted = 0
    for original in _chunks(words, chunk_words):
        cur = ""
        for piece in _generate(PROMPTS[language].format(text=original), model):
            cur += piece
            yield " ".join(done + [cur.strip()]), reverted
        merged, n = _merge(original, cur, language)
        done.append(merged)
        reverted += n
        yield " ".join(done), reverted
