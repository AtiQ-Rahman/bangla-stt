# Bangla STT

Local Bangla and English speech-to-text: paste a YouTube link or upload audio/video, get a
transcript, subtitles and more. Runs on CPU; no audio leaves the PC.

## Run

```bash
./run.sh                 # creates .venv and installs requirements on first run / when they change
./install-shortcut.sh    # optional: "Bangla STT" in the app menu and on the desktop
```

Opens http://127.0.0.1:7860. Needs `ffmpeg` on the system, and a C++ compiler + cmake the
first time (to build `kenlm`). Stop the app with **Ctrl+C** (or by closing the shortcut's
terminal window) — Ctrl+Z only pauses it and keeps its memory.

## Transcribe

1. **Get the audio**: in the *YouTube* tab paste a link and press *Audio only*, or
   *Video + audio* to also get the video file. In *From your computer* pick a video or audio
   file. Or record with the microphone. Listen, cut with ✂, or type *From* / *To*
   (`90`, `1:30`, `0:01:30`). The language (Bangla or English) is detected and set for you.
2. **Transcribe**. A progress bar shows how far it is and the time left. **Stop** keeps what
   is done; **⏯ Resume** continues from the last word. The transcript box is editable and the
   downloads (.txt, .srt) follow your edits.
3. **Use the text**:
   - *AI-fixed (local)*: punctuation and spelling with a free local model (below); it can be
     stopped and resumed too, and gives fixed .txt and .srt.
   - *Export*: `.vtt` subtitles, a Word `.docx`, and the **video with subtitles burned in**
     (for audio that came from a video).
   - **Send to Claude** with a task (fix, summarise, translate, or your own): the prompt and
     text are copied and claude.ai opens — paste with Ctrl+V. The Claude in Chrome extension
     can also read the transcript on this page directly.

| Setting | Effect |
|---|---|
| Model: MMS 1B (Bangla) | best Bangla accuracy, ~2.5x faster than real time |
| Model: XLS-R 300M (Bangla) | faster, noticeably more mistakes |
| Model: Whisper large-v3-turbo (English) | best, ~3x faster than real time |
| Model: Whisper small (English) | ~8x faster, less accurate |
| Fast mode (int8) | ~1.7x faster and much less RAM, same accuracy |
| Bangla language model | fixes many spellings; 3.5 GB download on first use |
| Bangla digits | writes ২০ instead of 20 |
| Names or special words | the decoder prefers their spelling |
| Use my browser login | if YouTube says "confirm you're not a bot", yt-dlp retries with your browser's YouTube cookies (they stay on this PC) |

## Batch and History

- **📚 Batch**: build a queue — paste YouTube links (one or many) and press *Add*, or pick
  many audio/video files at once — then *Start*. Items run one after another, the language of
  each is detected, and each lands in History. Pressing *Start* again later only runs the
  items that aren't done yet; remove items or empty the queue below it.
- **🕘 History**: every transcription (audio, text, subtitles, AI-fixed text, progress) is kept
  in the `history` folder. Open one to read, download or **Resume** it — even after the app or
  the PC was restarted. Delete entries you no longer need (the audio copies take space).

## RAM

The header shows free RAM; it warns, with a sound, when RAM runs low. Before loading a model
the app checks there is enough and refuses with a hint otherwise. Models are unloaded after 10
idle minutes, and before the local AI fix runs. A 2-hour file needs ~3.5 GB while transcribing.

## Local AI fix (optional)

Adds punctuation and fixes spelling with a free model served by [Ollama](https://ollama.com):

```bash
docker run -d --name ollama -v ollama:/root/.ollama -p 127.0.0.1:11434:11434 ollama/ollama
docker exec ollama ollama pull gemma3:4b
# after a reboot: docker start ollama
```

The fix is word-guarded: the AI may re-spell words and add punctuation, but changes that add,
drop or reword words, or use another script, are undone. Claude is still clearly better.
