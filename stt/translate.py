"""Translate subtitle lines between Bangla and English with the local LLM (Ollama)."""
import re
from typing import Iterator

from stt.correct import _generate

PROMPT = """Translate these numbered {source} subtitle lines into natural {target}. They come from \
speech recognition, so silently fix obvious recognition mistakes. A sentence may continue on the \
next line: keep each line's part on its own line. Reply with exactly the same numbers, one line \
each, and nothing else.

{lines}"""

ONE = """Translate this {source} subtitle line into natural {target}. Reply with the translation only.

{line}"""


def _parse(reply: str, count: int) -> list[str] | None:
    """The numbered lines 1..count of a reply, or None if any is missing."""
    found = {}
    for line in reply.splitlines():
        m = re.match(r"\s*(\d+)\s*[.):-]\s*(.*)", line)
        if m and 1 <= int(m.group(1)) <= count:
            found.setdefault(int(m.group(1)), m.group(2).strip())
    return [found[i] for i in range(1, count + 1)] if len(found) == count else None


def translate_lines(lines: list[str], source: str, target: str, model: str,
                    batch: int = 12) -> Iterator[list[str]]:
    """Translate lines in batches, yielding all translations so far after each batch.

    A batch whose reply doesn't number every line is redone line by line, so the result always
    has exactly one translation per line (and subtitle timing stays right).
    """
    out: list[str] = []
    for i in range(0, len(lines), batch):
        chunk = lines[i:i + batch]
        numbered = "\n".join(f"{n}. {line}" for n, line in enumerate(chunk, 1))
        reply = "".join(_generate(PROMPT.format(source=source, target=target, lines=numbered), model))
        parsed = _parse(reply, len(chunk))
        if parsed is None:
            parsed = ["".join(_generate(ONE.format(source=source, target=target, line=line), model)).strip()
                      for line in chunk]
        out += parsed
        yield out
