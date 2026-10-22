"""The user's own defaults for the settings, kept in ~/.config/bangla-stt/settings.json
(STT_CONFIG_DIR to move it, e.g. for a test server)."""
import json
import os
from pathlib import Path

import gradio as gr

from stt.correct import DEFAULT_LLM, list_models
from stt.transcribe import LANGUAGE_MODELS, WHISPER_MODELS

FILE = Path(os.environ.get("STT_CONFIG_DIR", Path.home() / ".config" / "bangla-stt")) / "settings.json"
BUILT_IN = {"language": "Bangla", "bangla_model": LANGUAGE_MODELS["Bangla"][0],
            "english_model": LANGUAGE_MODELS["English"][0], "quantize": True, "use_lm": True,
            "digits": False, "speakers": False, "use_learned": True, "use_browser": True,
            "claude_task": "Fix spelling & punctuation", "llm": DEFAULT_LLM, "batch_lang": "Auto",
            "watch_folder": str(Path.home() / "bangla-stt-inbox"),
            "live_language": "Bangla", "live_bangla_model": LANGUAGE_MODELS["Bangla"][0],
            "live_english_model": "Whisper small - faster"}
# Live tab models: not the Bangla Whisper, which is slower than speech on a CPU; for English
# the small Whisper first (large-v3-turbo falls behind the speaker)
LIVE_MODELS = {"Bangla": [m for m in LANGUAGE_MODELS["Bangla"] if m not in WHISPER_MODELS],
               "English": sorted(LANGUAGE_MODELS["English"], key=lambda m: "small" not in m)}


def load() -> dict:
    """The saved defaults over the built-in ones (unknown or broken entries are ignored)."""
    try:
        saved = json.loads(FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        saved = {}
    return {**BUILT_IN, **{k: v for k, v in saved.items() if k in BUILT_IN}}


def model_for(language: str) -> str:
    """The default model for a language: the saved one if it still exists."""
    choice = load()[f"{language.lower()}_model"]
    return choice if choice in LANGUAGE_MODELS[language] else LANGUAGE_MODELS[language][0]


def save_defaults(language, model, quantize, use_lm, digits, speakers, use_learned, use_browser,
                  claude_task, llm, batch_lang):
    values = load()
    values.update({"language": language, f"{language.lower()}_model": model, "quantize": quantize,
                   "use_lm": use_lm, "digits": digits, "speakers": speakers, "use_learned": use_learned,
                   "use_browser": use_browser,
                   "claude_task": claude_task, "llm": llm or DEFAULT_LLM, "batch_lang": batch_lang})
    FILE.parent.mkdir(parents=True, exist_ok=True)
    FILE.write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8")
    return f"✅ Saved as your defaults ({language}: {model}). They apply every time the app opens."


def apply_defaults():
    """Page open: put the defaults into the settings (DEFAULT_OUTPUTS order)."""
    v = load()
    models = LANGUAGE_MODELS[v["language"]]
    return (v["language"], gr.Radio(choices=models, value=model_for(v["language"])), v["quantize"],
            v["use_lm"], v["digits"], v["speakers"], v["use_learned"], v["use_browser"], v["claude_task"], v["batch_lang"])


def live_model_for(language: str) -> str:
    """The Live tab's default model for a language: the saved one if it is still offered."""
    choice = load()[f"live_{language.lower()}_model"]
    return choice if choice in LIVE_MODELS[language] else LIVE_MODELS[language][0]


def live_models(language):
    """The Live language changed: its models, with the saved one picked."""
    return gr.Radio(choices=LIVE_MODELS[language], value=live_model_for(language))


def apply_live_defaults():
    """Page open: the Live tab's own language and model."""
    language = load()["live_language"]
    return language, live_models(language)


def save_live_defaults(language, model):
    values = load()
    values.update({"live_language": language, f"live_{language.lower()}_model": model})
    FILE.parent.mkdir(parents=True, exist_ok=True)
    FILE.write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8")
    return f"✅ Saved as the Live default ({language}: {model})."


def remember(key: str, value):
    """Keep one setting (e.g. the watched folder) without touching the others."""
    values = load()
    values[key] = value
    FILE.parent.mkdir(parents=True, exist_ok=True)
    FILE.write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8")


def reset_defaults():
    FILE.unlink(missing_ok=True)
    return apply_defaults() + ("↺ Back to the built-in defaults.",)


def default_llm():
    """The local AI model to preselect: the saved one if Ollama has it."""
    models = list_models()
    saved = load()["llm"]
    value = saved if saved in models else (DEFAULT_LLM if DEFAULT_LLM in models else (models[0] if models else None))
    return gr.Dropdown(choices=models, value=value)
