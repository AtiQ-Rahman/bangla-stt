"""Saved defaults: kept in a file, applied on page open, per-language models, reset."""
import pytest

from stt.transcribe import LANGUAGE_MODELS
from webui import settings


@pytest.fixture(autouse=True)
def tmp_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "FILE", tmp_path / "settings.json")


def test_built_in_until_saved():
    assert settings.load() == settings.BUILT_IN
    assert settings.model_for("Bangla") == LANGUAGE_MODELS["Bangla"][0]


def test_save_keeps_each_languages_model():
    bn_whisper = LANGUAGE_MODELS["Bangla"][-1]
    settings.save_defaults("Bangla", bn_whisper, False, False, True, True, True, False, "Translate (Bangla ↔ English)",
                           "gemma3:4b", "English")
    settings.save_defaults("English", "Whisper small - faster", False, False, True, True, False, False,
                           "Summarise (bullet points)", "gemma3:4b", "English")
    assert settings.model_for("Bangla") == bn_whisper  # not lost by saving the English choice
    assert settings.model_for("English") == "Whisper small - faster"
    values = settings.apply_defaults()
    assert values[0] == "English" and values[1].value == "Whisper small - faster" and values[4] is True


def test_unknown_model_falls_back_and_reset(tmp_path):
    settings.FILE.write_text('{"bangla_model": "gone", "language": "Bangla", "junk": 1}')
    assert settings.model_for("Bangla") == LANGUAGE_MODELS["Bangla"][0]
    assert "junk" not in settings.load()
    settings.reset_defaults()
    assert not settings.FILE.exists()
