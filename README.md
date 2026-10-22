# Bangla STT

Local Bangla and English speech-to-text: paste a YouTube link or upload audio/video, get a
transcript (`.txt`) and subtitles (`.srt`). Runs on CPU; no audio leaves the PC.

## Run

```bash
./run.sh            # creates .venv and installs requirements on first run / when they change
```

Opens http://127.0.0.1:7860. Needs `ffmpeg` on the system and a C++ compiler +
cmake the first time (to build `kenlm`).

## Using it

1. **Get the audio**: in the *YouTube* tab paste a link and press *Audio only*, or
   *Video + audio* to also get the video file to save. In *From your computer* pick a
   video or audio file. Or record with the microphone. The audio lands in the player:
   listen, cut it with ✂, or type *From* / *To* (`90`, `1:30`, `0:01:30`) — the selected
   part appears below. The language (Bangla or English) is detected and set for you.
2. **Transcribe**. The text appears piece by piece under a progress bar. **Stop** keeps
   what is done so far, and **⏯ Resume** later continues from the last word (same audio
   and settings). **Clear results** empties the results but keeps the audio. The transcript
   box is editable.
3. **Use the text**: download `.txt` / `.srt`, or **Send to Claude** — picks a task
   (fix spelling, summarise, translate, or your own), copies prompt + transcript to the
   clipboard and opens claude.ai with it (paste with Ctrl+V if it isn't filled in).
   With the Claude in Chrome extension you can also just ask it about the transcript on
   this page.

In *Settings*, list names or special words in the audio (comma separated) so the decoder
prefers their correct spelling.

| Setting | Effect |
|---|---|
| Language: English | uses Whisper (large-v3-turbo, the default: best, ~3x faster than real time; small: ~8x faster, less accurate) |
| Model: MMS 1B | best accuracy (~0.35x real time on an i7-1165G7 with the defaults) |
| Model: XLS-R 300M | several times faster, noticeably more mistakes |
| Fast mode (int8) | ~1.7x faster and much less RAM, same accuracy |
| Bangla language model | fixes many spellings; 3.5 GB download on first use |

## RAM

The header shows free RAM. Before loading a model the app checks there is enough
(MMS ~2.5 GB, XLS-R ~1 GB with Fast mode) and refuses with a hint otherwise. The models
are unloaded after 10 idle minutes, and before the local AI fix runs.

## Fix the transcript with a local LLM (optional)

Adds punctuation and fixes spelling with a free model served by [Ollama](https://ollama.com):

```bash
docker run -d --name ollama -v ollama:/root/.ollama -p 127.0.0.1:11434:11434 ollama/ollama
docker exec ollama ollama pull gemma3:4b
# after a reboot: docker start ollama
```

Then open the "AI-fixed (local)" tab, press ↻ to list models and *Fix transcript*.
The fix is word-guarded: the AI may re-spell words and add punctuation, but changes that
add, drop or reword words, or use another script, are undone. Bigger models fix more
(`ollama pull gemma3:12b`, ~8 GB, slower); Claude is still clearly better.
The speech models are unloaded before the LLM runs, because both do not fit in 16 GB RAM.
