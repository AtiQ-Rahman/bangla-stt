"""Transcribing (and resuming) with live progress; downloads follow text edits."""
import shutil
import time
import unicodedata
from pathlib import Path

import gradio as gr

from stt import dictionary, history
from stt.audio import cut_clip
from stt.export import bangla_digits, parse_srt, srt_from_cues
from stt.langid import detect
from stt.retime import retime
from stt.transcribe import Segment, to_srt, to_text, Word, WHISPER_MODELS
from stt.speakers import label
from stt.worker import WORKER, WorkerDied
from webui.common import (bar, check_ram, file_button, fmt_dur, language_of, LIVE, no_file,
                          out_file, parse_time, runtime, time_left, WORK_DIR)


def _work(job: dict, entry: Path, prior: list[Segment], at: float):
    """Transcribe job["audio"] from `at` seconds to job["end"], after the segments in prior,
    saving progress to the history entry (every 10 s, and when it ends or is stopped).

    Yields (segments, fraction done, resume point, length of the range, model was loaded, phase):
    once before the model loads, then after every piece ("transcribing"), and once more before
    the speakers are found if job["speakers"] ("speakers"). The work itself runs in the worker
    process (stt.worker).
    """
    begin = job["start"] or 0.0
    # Resuming: hear 3 s before the last word too, as context; words only come after `at`.
    load_from = max(begin, at - 3.0) if prior else at
    was_loaded = WORKER.is_loaded(job["model"], job["quantize"])
    pieces = WORKER.transcribe(job["audio"], load_from, job["end"], job["model"], job["quantize"],
                               job["use_lm"], job["words"], at - load_from, job.get("whisper"))
    try:
        _, samples = next(pieces)  # ("start", samples) once the worker has read the audio
    except (WorkerDied, RuntimeError) as e:
        raise gr.Error(f"Could not read the audio: {e}")
    if not samples:
        pieces.close()
        raise gr.Error("No audio left in the selected range.")
    before, remaining = load_from - begin, samples / 16000
    audio_sec = before + remaining
    segments, resume_at, finished, saved = list(prior), at, False, time.monotonic()
    digits = job.get("bn_digits") and language_of(job["model"]) == "Bangla"
    live = LIVE[str(entry)] = {"segments": segments, "frac": before / audio_sec, "audio_sec": audio_sec,
                               "loading": True}
    runtime.active += 1
    try:
        yield segments, before / audio_sec, resume_at, audio_sec, was_loaded, "loading"
        for kind, piece in pieces:
            if kind == "cancelled":  # Stop: the worker ended the job early
                break
            done, total, seg = piece
            if seg:
                if digits:
                    seg = Segment(seg.start, seg.end, bangla_digits(seg.text),
                                  [Word(w.start, w.end, bangla_digits(w.text)) for w in seg.words])
                segments.append(seg)
                resume_at = max(resume_at, seg.end)
            live["frac"], live["loading"] = (before + done / total * remaining) / audio_sec, False
            if time.monotonic() - saved > 10:
                history.save(entry, job, segments, resume_at, "running")
                saved = time.monotonic()
            yield (segments, (before + done / total * remaining) / audio_sec, resume_at, audio_sec,
                   was_loaded, "transcribing")
        else:
            finished = True
        if finished and job.get("speakers"):  # over the whole range, so numbers stay the same
            live["phase"], live["speakers_frac"] = "speakers", 0.0
            yield segments, 0.0, resume_at, audio_sec, was_loaded, "speakers"
            turns = []
            for kind, value in WORKER.speaker_steps(job["audio"], begin, job["end"]):
                if kind == "progress":
                    live["speakers_frac"] = value
                    yield segments, value, resume_at, audio_sec, was_loaded, "speakers"
                else:
                    turns = value
            if turns is None:  # Stop
                finished = False
            else:
                segments[:] = label(segments, turns)
    except WorkerDied as e:
        raise gr.Error(f"Transcription failed: {e}. The text so far is kept; ⏯ Resume continues.")
    except RuntimeError as e:
        raise gr.Error(f"Transcription failed: {e}")
    finally:
        pieces.close()
        runtime.active -= 1
        runtime.last_used = time.monotonic()
        history.save(entry, job, segments, resume_at, "done" if finished else "stopped")
        LIVE.pop(str(entry), None)  # after saving: a follower then loads the saved result


# run() outputs: transcript, txt, srt, srt view, status, clip, progress bar, partial, segs, entry
def stage(message: str) -> tuple:
    """A run() update that only changes the status line and the progress bar."""
    return (gr.skip(),) * 4 + (message, gr.skip(), bar(message)) + (gr.skip(),) * 3


def speakers_stage(frac: float, first: tuple[float, float] | None = None) -> tuple:
    message = f"⏳ Finding who speaks when (speaker labels)… {frac:.0%}{time_left(first, frac)}"
    return (gr.skip(),) * 4 + (message, gr.skip(), bar(message, frac)) + (gr.skip(),) * 3


def make_job(audio, title, model_name, quantize, use_lm, hotwords, start, end, digits, speakers=False,
             use_learned=True, whisper=None):
    start, end = parse_time(start), parse_time(end)
    if start and end and end <= start:
        raise gr.Error("End time must be after start time.")
    words = [w.strip() for w in (hotwords or "").replace("\n", ",").split(",") if w.strip()]
    if use_learned:  # the words the user corrected before; fewer for Whisper (a short prompt)
        learned = dictionary.words(language_of(model_name), 15 if model_name in WHISPER_MODELS else 50)
        words += [w for w in learned if w not in words]
    use_lm = use_lm and model_name not in WHISPER_MODELS  # Whisper has its own
    return {"audio": audio, "title": title or Path(audio).stem, "model": model_name,
            "quantize": quantize, "use_lm": use_lm, "words": words, "start": start, "end": end,
            "bn_digits": bool(digits), "speakers": bool(speakers), "whisper": whisper or {}}


def whisper_options(prompt, vad, silence_guard, carry_text) -> dict:
    return {"prompt": (prompt or "").strip(), "vad": bool(vad), "silence_guard": bool(silence_guard),
            "carry_text": bool(carry_text)}


def run(audio, title, model_name, quantize, use_lm, hotwords, start, end, digits, speakers,
        use_learned, video, source, prompt="", vad=True, silence_guard=True, carry_text=True):
    if not audio:
        raise gr.Error("Add some audio first: get it from YouTube, upload a video or audio file, or record.")
    job = make_job(audio, title, model_name, quantize, use_lm, hotwords, start, end, digits, speakers,
                   use_learned, whisper_options(prompt, vad, silence_guard, carry_text))
    if video:
        job["video"] = video  # so "video with subtitles" also works when reopened from History
    if source:
        job["source"] = source  # the YouTube link, shown when reopened
    check_ram(job["model"], job["quantize"], job["use_lm"])

    # Clear the previous result straight away.
    yield ("", no_file(), no_file(), "", "⏳ Starting…", gr.Audio(), bar("⏳ Checking the language…"),
           None, None, None)
    chosen = language_of(model_name)
    try:
        heard, confidence = detect(audio, job["start"], job["end"])
    except Exception:
        heard, confidence = chosen, 0.0
    if heard != chosen and confidence >= 0.8:
        raise gr.Error(f"This sounds like {heard} ({confidence:.0%} sure), but {chosen} is selected. "
                       f"Choose {heard} under 'Language of the speech' and press Transcribe again.")
    shutil.rmtree(WORK_DIR / "run", ignore_errors=True)
    entry = history.create(job)  # copies the audio in; job["audio"] now points at that copy
    yield from _job(job, entry, [], job["start"] or 0.0)


def resume(partial):
    """Continue a stopped transcription from its last word, with the same audio and settings."""
    if not partial:
        raise gr.Error("Nothing to resume: press Transcribe.")
    job = partial["job"]
    check_ram(job["model"], job["quantize"], job["use_lm"])
    yield stage(f"⏳ Resuming at {fmt_dur(partial['at'])}…")
    yield from _job(job, Path(partial["entry"]), partial["segments"], partial["at"])


def _job(job, entry, prior, at):
    """run()/resume() progress for the UI around _work()."""
    txt_path, srt_path = out_file(job["title"], ".txt"), out_file(job["title"], ".srt")
    yield stage("⏳ Reading the audio…")
    clip = gr.skip()
    if not prior and (job["start"] or job["end"]):  # let the user hear what is transcribed
        try:
            clip = gr.Audio(value=str(cut_clip(job["audio"], out_file(job["title"], " part.mp3"),
                                               job["start"], job["end"])),
                            visible=True, label="Transcribed part")
        except RuntimeError:
            pass
    t0 = first = speakers_first = None
    segments, audio_sec, was_loaded = list(prior), 0.0, True
    for segments, frac, resume_at, audio_sec, was_loaded, phase in _work(job, entry, prior, at):
        if phase == "speakers":
            yield speakers_stage(frac, speakers_first)
            speakers_first = speakers_first or (time.monotonic(), frac)
            continue
        if t0 is None:  # before the model loads
            t0 = time.monotonic()
            if not was_loaded:
                yield stage("⏳ Loading the model into memory — the first time this takes up to a minute…")
            continue
        first = first or (time.monotonic(), frac)
        txt_path.write_text(to_text(segments), encoding="utf-8")
        srt = to_srt(segments)
        srt_path.write_text(srt, encoding="utf-8")
        progress_text = (f"⏳ Transcribing… {frac:.0%} · {fmt_dur(frac * audio_sec)} of "
                         f"{fmt_dur(audio_sec)}{time_left(first, frac)}")
        partial = {"job": job, "entry": str(entry), "segments": segments, "at": resume_at}
        yield (to_text(segments), file_button(txt_path), file_button(srt_path), srt,
               "⏳ Transcribing — press Stop any time; you can Resume later", clip,
               bar(progress_text, frac), partial, segments, str(entry))
    took = time.monotonic() - (t0 or time.monotonic())
    done_text = (f"✅ Done: {fmt_dur(audio_sec)} of audio"
                 + (f" (this part took {fmt_dur(took)})" if prior else f" in {fmt_dur(took)}")
                 + ("" if was_loaded else ", incl. loading the model") + " · saved in History")
    txt_path.write_text(to_text(segments), encoding="utf-8")
    srt_path.write_text(to_srt(segments), encoding="utf-8")
    yield (to_text(segments), file_button(txt_path), file_button(srt_path), to_srt(segments),
           done_text, clip, bar(done_text, 1, done=True), None, segments, str(entry))


def edited_text(text, title, entry, name):
    """The user typed in a text box: its .txt download (and History copy) follow."""
    path = out_file(title, " (fixed).txt" if name == "corrected.txt" else ".txt")
    path.write_text(text or "", encoding="utf-8")
    if entry:
        history.save_text(Path(entry), name, text or "")
    return file_button(path)


def learn_edits(text, segs, entry, language):
    """Corrected words of an edited transcript go to the user's dictionary (stt.dictionary);
    returns the language's learned words for the Settings box."""
    new = dictionary.learn(segs, text, str(entry or "")) if segs else []
    if not new:
        return gr.skip()
    gr.Info("Learned for next time: " + ", ".join(new[:8]) + (" …" if len(new) > 8 else ""))
    return learned_words(language)


def learned_words(language):
    return ", ".join(dictionary.words(language, limit=1000))


def set_learned_words(text, language):
    dictionary.set_words(language, text)
    return learned_words(language)


def edited_subtitles(text, segs, title, suffix):
    """When the user leaves a text box: re-time its text onto the transcript for new subtitles."""
    if not segs or not (text or "").strip():
        return gr.skip(), gr.skip()
    srt = to_srt(retime(text, segs))
    path = out_file(title, suffix)
    path.write_text(srt, encoding="utf-8")
    return file_button(path), srt


def uploaded_subtitles(path, segs, title, entry, language):
    """An .srt edited elsewhere (e.g. by Claude) becomes the result: its text the transcript,
    its cues the subtitles as they are, and its corrected words are learned."""
    cues = parse_srt(Path(path).read_text(encoding="utf-8", errors="replace")) if path else []
    if not cues:
        raise gr.Error("No subtitles found in that file — is it an .srt?")
    text, last = "", None
    for start, end, words in cues:  # a new paragraph after a pause, like the transcript
        text += ("" if last is None else "\n\n" if start - last >= 1.5 else " ") + words
        last = end
    text = unicodedata.normalize("NFC", text)
    srt = unicodedata.normalize("NFC", srt_from_cues(cues))
    srt_path = out_file(title, ".srt")
    srt_path.write_text(srt, encoding="utf-8")
    txt_button = edited_text(text, title, entry, "transcript.txt")
    gr.Info(f"Loaded {len(cues)} subtitles from the file.")
    return text, txt_button, file_button(srt_path), srt, learn_edits(text, segs, entry, language)
