"""Shared test helpers. Tests that need the speech models are marked `models` and use real
speech from STT_SAMPLE_BN / STT_SAMPLE_EN (audio files on your PC); they are skipped otherwise."""
import os
import subprocess
from pathlib import Path

import pytest

from stt.transcribe import Segment, Word


def pytest_configure(config):
    config.addinivalue_line("markers", "models: loads the speech models (slow, needs RAM)")


def make_audio(path: Path, spec: str) -> Path:
    """An audio file from an ffmpeg lavfi graph, e.g. a tone, silence, then a tone."""
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi", "-i", spec,
                    "-ac", "1", "-ar", "16000", str(path)], check=True)
    return path


@pytest.fixture
def tone_gap_tone(tmp_path) -> Path:
    """3 s tone, 1.5 s silence, 2 s tone (6.5 s)."""
    return make_audio(tmp_path / "tones.wav",
                      "sine=f=440:d=3[a];anullsrc=r=16000:cl=mono:d=1.5[s];sine=f=440:d=2[b];"
                      "[a][s][b]concat=n=3:v=0:a=1")


@pytest.fixture
def segments() -> list[Segment]:
    return [
        Segment(0.0, 3.0, "আমি ভাত খাই", [Word(0, 1, "আমি"), Word(1, 2, "ভাত"), Word(2, 3, "খাই")]),
        Segment(5.0, 8.0, "জন প্রশাসন মন্ত্রণালয়",
                [Word(5, 6, "জন"), Word(6, 7, "প্রশাসন"), Word(7, 8, "মন্ত্রণালয়")]),
    ]


def sample(lang: str) -> Path:
    path = os.environ.get(f"STT_SAMPLE_{lang.upper()}")
    if not path or not Path(path).exists():
        pytest.skip(f"set STT_SAMPLE_{lang.upper()} to a {lang} speech file to run this test")
    return Path(path)
