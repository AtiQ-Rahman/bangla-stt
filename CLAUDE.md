# Bangla STT — notes for working on this repo

Local Bangla/English speech-to-text web app (Gradio 6, CPU only). README.md has the user-facing
features; this file is about the code and the lessons learned building it.

## Layout
- `app.py` — page layout and event wiring only; `main()` parses `--phone/--password/--port`.
- `webui/` — handlers by area: `common` (temp dir, RAM, helpers, `LIVE`, `runtime`), `sources`
  (YouTube/files/trim/language), `transcribing` (`_work` core loop, run/resume, edit sync),
  `fixing` (Ollama fix), `exporting` (vtt/docx/burned video/translation), `batch`, `opening`
  (History, dashboard open, live follow via `OPEN_FIELDS`/`opened()`), `controls` (button states),
  `readalong` + `look` (Read along HTML + page JS/CSS), `settings` (saved defaults), `claude`, `phone`,
  `chaptering` (Chapters tab), `watch` (Batch folder watch), `dictating` (Live tab; per-browser
  sessions keyed by `request.session_hash`).
- `stt/` — no UI: `transcribe` (MMS/XLS-R CTC windows + Whisper, NFC output), `worker` (models run
  in a spawned process), `audio` (ffmpeg, yt-dlp, browser cookies), `history`, `retime`, `correct`
  (guarded LLM fix), `translate`, `speakers` (sherpa-onnx diarization), `export`, `langid`,
  `dictionary` (words learned from edits → hotwords), `chapters` (parts at pauses + LLM titles),
  `dictation` (mic chunks → phrases at pauses → background transcription).

## Commands
- Run: `./run.sh` (port 7860). Tests: `.venv/bin/python -m pytest -q tests` (~16 s);
  `STT_SAMPLE_BN=<bangla audio>` enables the model tests. Lint: `.venv/bin/python -m pyflakes app.py webui stt tests`.
- The user's own app usually runs on 7860 — test servers go on 7861 and must not touch
  `~/.config/bangla-stt` (start them with `STT_CONFIG_DIR=<temp dir>`) or kill the user's
  processes; xdg-open is skipped by patching `app.open_browser` in a small launcher.

## Gotchas (each cost real debugging time)
- Handlers return positional tuples matching output lists; `tests/test_app.py` checks the counts —
  update `busy_buttons`/`idle_buttons`/`clear_results`/`OPEN_FIELDS` together with the layout.
- Bangla vowel signs (ী, া …) are not `\w` in Python regex: never strip "punctuation" with `\W`;
  use `unicodedata.category`. Whisper chunk words may lack the leading space: join with `_join`.
- gemma3:4b answers in Bangla with translated labels ("শিরোনাম:" for "TITLE:") and invents
  names when not told to stick to the text: parse loosely, forbid outside facts in prompts.
- The Live tab is browser-tested with headless Chrome's fake mic: `--use-fake-ui-for-media-stream
  --use-fake-device-for-media-stream --use-file-for-fake-audio-capture=<48 kHz wav>%noloop`.
  Click only visible buttons (`b.offsetParent`): every tab's Audio has a Record button.
- Never `pkill -f`/`pgrep -f` a pattern that also appears in the same shell command (it kills
  that shell); keep test servers' PIDs in a file.
- Values computed in browser JS cannot reach the server through `gr.State`; use a hidden
  component (`gr.Checkbox`/`gr.Textbox` with `visible=False` or a `.ra-hidden` class).
- Gradio cancel of a generator only lands when the current `next()` returns, so Stop goes through
  `WORKER.cancel()` (+ kill after a grace period); `stopped()` uses `queue=False` and a JS-only
  instant "Stopping…" update.
- `multiprocessing` spawn re-runs the main script; `Worker._start` hides `__main__` while starting.
  The worker exits itself when its parent dies (orphans held GBs of RAM).
- Bangla `য়` has two Unicode encodings: all text is NFC (transcribe output, History search, CER).
- MMS drops words at window starts/ends: windows overlap (3 s lead, 1 s margin). The Bengali.AI
  Whisper must run on ~14 s chunks (long-form mode drops passages), with word timestamps on.
  It invents words at each chunk's end: trailing zero-length or <0.8-probability words are
  dropped. Shorter chunks (8–10 s) and beam 5 are much worse; it also writes ো as া sometimes.
- RAM is tight (16 GB): int8 quantize *in place*, KenLM is memory-mapped, the worker is killed
  before Ollama runs, `check_ram` refuses loads that won't fit.
- API tests (`gradio_client`) don't fire UI change events (e.g. call `/pick_language` first so a
  model radio accepts the model; `/text_choices` before exporting "AI-fixed text").
- `pkill -f <pattern>` in a shell whose own command line contains the pattern kills that shell —
  use a bracket trick (`[s]erve.py`) or PID files.

## Measured (test clip: 102 s Bangla news)
CER: MMS+LM int8 4.0% (~35 s), Bengali.AI Whisper chunked 3.9% (~4 min; better names),
XLS-R 9%. Noise filters, Demucs, LM weight tuning and gemma3:12b fixes did not help.
