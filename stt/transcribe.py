"""Bangla speech recognition on CPU using Hugging Face CTC models."""
import logging
import re
import subprocess
import sys
import threading
import unicodedata
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from stt import cached_first
from stt.audio import SAMPLE_RATE, split_on_silence

# label -> (hugging face repo, extra kwargs)
MODELS = {
    "MMS 1B (Meta) - best accuracy": ("facebook/mms-1b-all", {"target_lang": "ben"}),
    "XLS-R 300M Bengali - faster": ("arijitx/wav2vec2-xls-r-300m-bengali", {}),
}
DEFAULT_MODEL = next(iter(MODELS))

# Whisper via faster-whisper (CTranslate2, int8 on CPU): label -> model, language, and for a
# Hugging Face model that has to be converted first, "convert". "chunk_sec": read the audio in
# chunks of about that length, cut at quiet moments (the Bangla model was trained on short clips
# and drops text in Whisper's usual 30 s long-form mode).
WHISPER_MODELS = {
    "Whisper large-v3-turbo - best": {"model": "large-v3-turbo", "language": "en"},
    "Whisper small - faster": {"model": "small", "language": "en"},
    "Whisper medium Bangla (Bengali.AI) - names, slow": {
        "model": "bengaliAI/tugstugi_bengaliai-asr_whisper-medium", "language": "bn",
        "convert": True, "chunk_sec": 14.0},
}
LANGUAGE_MODELS = {
    "Bangla": list(MODELS) + [m for m, spec in WHISPER_MODELS.items() if spec["language"] == "bn"],
    "English": [m for m, spec in WHISPER_MODELS.items() if spec["language"] == "en"],
}
CT2_DIR = Path.home() / ".cache" / "bangla-stt" / "ct2"  # converted Whisper models

# 5-gram Bangla word language model (~3.5 GB), shipped with the XLS-R model but
# usable with any character-level Bangla CTC model.
LM_REPO = "arijitx/wav2vec2-xls-r-300m-bengali"
LM_ALPHA, LM_BETA = 0.5, 1.5
HOTWORD_WEIGHT = 8.0

# Rough RAM (GB) each model needs once loaded, by (model, quantized).
RAM_NEEDED_GB = {
    ("MMS 1B (Meta) - best accuracy", True): 2.5, ("MMS 1B (Meta) - best accuracy", False): 4.5,
    ("XLS-R 300M Bengali - faster", True): 1.0, ("XLS-R 300M Bengali - faster", False): 1.5,
    ("Whisper large-v3-turbo - best", True): 3.0, ("Whisper large-v3-turbo - best", False): 3.0,
    ("Whisper small - faster", True): 1.5, ("Whisper small - faster", False): 1.5,
    ("Whisper medium Bangla (Bengali.AI) - names, slow", True): 3.0,
    ("Whisper medium Bangla (Bengali.AI) - names, slow", False): 3.0,
}
LM_RAM_GB = 2.0  # the LM is memory-mapped, so less than its 3.5 GB stays resident

logging.getLogger("pyctcdecode").setLevel(logging.ERROR)
_loaded: tuple[str, bool | None] | None = None  # (model_name, quantize) in memory; None: n/a
# Stop in the UI abandons a run, but a model load already under way finishes in its thread; a
# new run must wait for it instead of loading a second copy next to it.
_load_lock = threading.RLock()


def _one_at_a_time(fn):
    def locked(*args, **kwargs):
        with _load_lock:
            return fn(*args, **kwargs)
    locked.cache_clear = getattr(fn, "cache_clear", None)
    return locked


@dataclass
class Word:
    start: float
    end: float
    text: str
    speaker: int | None = None  # 1, 2, … when speakers were labelled (stt.speakers)


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


@_one_at_a_time
@lru_cache(maxsize=1)
def _load(model_name: str, quantize: bool):
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    from transformers.utils import logging as hf_logging

    hf_logging.disable_progress_bar()  # the UI shows loading; no "Loading weights" bar in the terminal

    global _loaded

    _load_whisper.cache_clear()  # only one speech model in RAM at a time
    repo, kw = MODELS[model_name]
    processor = cached_first(lambda **c: Wav2Vec2Processor.from_pretrained(repo, **kw, **c))
    if "target_lang" in kw:
        model = cached_first(lambda **c: Wav2Vec2ForCTC.from_pretrained(
            repo, target_lang=kw["target_lang"], ignore_mismatched_sizes=True, **c))
    else:
        model = cached_first(lambda **c: Wav2Vec2ForCTC.from_pretrained(repo, **c))
    model.eval()
    if quantize:
        # int8 weights for the Linear layers: ~1.7x faster on CPU, near-identical output.
        # inplace: a copying quantize briefly holds both models (~9 GB peak for MMS).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8,
                                                   inplace=True)
    _loaded = (model_name, quantize)
    return processor, model


@_one_at_a_time
@lru_cache(maxsize=1)
def _load_whisper(model_name: str):
    from faster_whisper import WhisperModel

    global _loaded
    _load.cache_clear()
    spec = WHISPER_MODELS[model_name]
    if spec.get("convert"):
        model = WhisperModel(str(_converted(spec["model"])), device="cpu", compute_type="int8")
    else:
        model = cached_first(lambda **c: WhisperModel(spec["model"], device="cpu",
                                                      compute_type="int8", **c))
    _loaded = (model_name, None)
    return model


def _converted(repo: str) -> Path:
    """A Hugging Face Whisper model in CTranslate2 int8 format, downloaded and converted once."""
    out = CT2_DIR / repo.split("/")[-1]
    if not (out / "model.bin").exists():
        from huggingface_hub import snapshot_download

        source = cached_first(lambda **c: snapshot_download(repo, **c))
        converter = Path(sys.executable).parent / "ct2-transformers-converter"
        subprocess.run([str(converter), "--model", source, "--output_dir", str(out), "--quantization",
                        "int8", "--force", "--copy_files", "tokenizer.json", "preprocessor_config.json"],
                       check=True, capture_output=True)
    return out


def is_loaded(model_name: str, quantize: bool) -> bool:
    return _loaded in ((model_name, quantize), (model_name, None))


def _labels(tokenizer, size: int) -> list[str]:
    """Tokenizer vocab as pyctcdecode labels: blank -> "", word delimiter -> " "."""
    vocab = sorted(tokenizer.get_vocab().items(), key=lambda kv: kv[1])[:size]
    labels = []
    for tok, idx in vocab:
        if idx == tokenizer.pad_token_id:
            labels.append("")
        elif tok in (tokenizer.word_delimiter_token, " "):
            labels.append(" ")
        elif tok.startswith("<") and tok.endswith(">"):
            labels.append(chr(0xE000 + idx))  # special tokens: private-use char, never a real word
        else:
            labels.append(tok)
    return labels


@_one_at_a_time
@lru_cache(maxsize=1)
def _language_model():
    """The KenLM model is big, so load it once and share it between decoders."""
    import kenlm
    from huggingface_hub import hf_hub_download
    from pyctcdecode import LanguageModel

    lm = cached_first(lambda **c: hf_hub_download(LM_REPO, "language_model/5gram.bin", **c))
    unigrams = cached_first(lambda **c: hf_hub_download(LM_REPO, "language_model/unigrams.txt", **c))
    with open(unigrams, encoding="utf-8") as f:
        words = [w.strip() for w in f if w.strip()]
    # Memory-map the model instead of copying it into RAM, so the OS can page it out.
    config = kenlm.Config()
    config.load_method = kenlm.LoadMethod.LAZY
    return LanguageModel(kenlm.Model(lm, config), words, alpha=LM_ALPHA, beta=LM_BETA)


@lru_cache(maxsize=2)
def _decoder(model_name: str, use_lm: bool, vocab_size: int):
    from pyctcdecode import Alphabet, BeamSearchDecoderCTC

    alphabet = Alphabet.build_alphabet(_labels(_tokenizer(model_name), vocab_size))
    return BeamSearchDecoderCTC(alphabet, _language_model() if use_lm else None)


@lru_cache(maxsize=2)
def _tokenizer(model_name: str):
    from transformers import Wav2Vec2CTCTokenizer

    repo, kw = MODELS[model_name]
    return cached_first(lambda **c: Wav2Vec2CTCTokenizer.from_pretrained(repo, **kw, **c))


def unload() -> None:
    """Free the speech models and language model, e.g. to make room for an LLM."""
    import gc
    from pyctcdecode import BeamSearchDecoderCTC

    global _loaded
    _loaded = None
    for cached in (_load, _load_whisper, _decoder, _language_model):
        cached.cache_clear()
    BeamSearchDecoderCTC.clear_class_models()  # decoders keep their LM in a class-level registry
    gc.collect()


def transcribe(wav: np.ndarray, model_name: str = DEFAULT_MODEL, quantize: bool = True,
               use_lm: bool = True, offset: float = 0.0, hotwords: list[str] | None = None,
               keep_from_sec: float = 0.0) -> Iterator[tuple[int, int, Segment | None]]:
    """Transcribe piece by piece, yielding (samples_done, total_samples, segment or None if silent).

    offset (seconds) is added to all timestamps, for audio cut out of a longer file.
    hotwords (names, special terms) are favoured by the decoder when the audio is close to them.
    keep_from_sec: audio before this point is only context (used when resuming); no words from it.
    Text is Unicode NFC, so the same word is the same string whichever model wrote it (e.g. য়
    as one character or as য + nukta).
    """
    if model_name in WHISPER_MODELS:
        pieces = _transcribe_whisper(wav, model_name, offset, hotwords, keep_from_sec)
    else:
        pieces = _transcribe_ctc(wav, model_name, quantize, use_lm, offset, hotwords=hotwords,
                                 keep_from_sec=keep_from_sec)
    for done, total, seg in pieces:
        if seg:
            seg = Segment(seg.start, seg.end, _nfc(seg.text),
                          [Word(w.start, w.end, _nfc(w.text), w.speaker) for w in seg.words])
        yield done, total, seg


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _transcribe_ctc(wav: np.ndarray, model_name: str = DEFAULT_MODEL, quantize: bool = True,
                    use_lm: bool = True, offset: float = 0.0, max_sec: float = 20.0,
               margin_sec: float = 1.0, lead_sec: float = 3.0,
               hotwords: list[str] | None = None,
               keep_from_sec: float = 0.0) -> Iterator[tuple[int, int, Segment | None]]:
    """MMS / XLS-R (CTC) transcription, piece by piece.

    Voiced spans longer than max_sec are read in windows of max_sec. The models garble
    words cut at a window edge, so the words in the last margin_sec of a window are thrown
    away. MMS also misses words in the first seconds of a long window, so the next window
    starts lead_sec before the cut; words there are only context and are dropped.
    offset (seconds) is added to all timestamps, for audio cut out of a longer file.
    hotwords (names, special terms) are favoured by the decoder when the audio is close to them.
    keep_from_sec: audio before this point is only context (used when resuming); no words from it.
    """
    processor, model = _load(model_name, quantize)
    boost = {"hotwords": hotwords, "hotword_weight": HOTWORD_WEIGHT} if hotwords else {}
    hop = model.config.inputs_to_logits_ratio  # samples per output frame
    max_len, margin = int(max_sec * SAMPLE_RATE), int(margin_sec * SAMPLE_RATE)
    lead = int(lead_sec * SAMPLE_RATE)

    def words_in(lo: int, hi: int, keep_from: int) -> list[tuple[int, int, str]]:
        inputs = processor(wav[lo:hi], sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with torch.inference_mode():
            logits = model(**inputs).logits[0].numpy()
        beam = _decoder(model_name, use_lm, logits.shape[1]).decode_beams(logits, beam_width=64, **boost)[0]
        words = [(lo + a * hop, lo + b * hop, w) for w, (a, b) in beam[2]]
        # Drop context words and noise read as one repeated symbol (e.g. "1111" over music).
        return [(a, b, w) for a, b, w in words
                if (a + b) // 2 >= keep_from and not re.fullmatch(r"(.)\1{2,}", w)]

    keep_from = int(keep_from_sec * SAMPLE_RATE)
    spans = [(max(s, keep_from), e) for s, e in split_on_silence(wav, max_sec) if e > keep_from]
    total, done = sum(e - s for s, e in spans), 0
    for s, e in spans:
        lo = max(0, s - lead) if s == keep_from else s  # resuming: start with some context
        while s < e:
            hi = min(e, lo + max_len)
            words = words_in(lo, hi, s)
            cut = hi
            if hi < e:
                keep = 0
                while keep < len(words) and words[keep][1] < hi - margin:
                    keep += 1
                if keep < len(words) and keep:
                    cut = (words[keep - 1][1] + words[keep][0]) // 2
                else:
                    cut = hi - margin
                words = words[:keep]
                lo = max(lo, cut - lead)
            seg = None
            if words:
                seg = Segment(offset + s / SAMPLE_RATE, offset + cut / SAMPLE_RATE,
                              " ".join(w for _, _, w in words),
                              [Word(offset + a / SAMPLE_RATE, offset + b / SAMPLE_RATE, w)
                               for a, b, w in words])
            done += cut - s
            s = cut
            yield done, total, seg


def _transcribe_whisper(wav: np.ndarray, model_name: str, offset: float, hotwords: list[str] | None,
                        keep_from_sec: float = 0.0) -> Iterator[tuple[int, int, Segment | None]]:
    """Whisper: it finds sentences, punctuation (English) and word times by itself."""
    model = _load_whisper(model_name)
    spec = WHISPER_MODELS[model_name]
    if spec.get("chunk_sec"):
        yield from _whisper_chunks(model, spec, wav, offset, hotwords, keep_from_sec)
        return
    segments, _ = model.transcribe(wav, language=spec["language"], vad_filter=True,
                                   word_timestamps=True,
                                   hotwords=", ".join(hotwords) if hotwords else None)
    total = len(wav)
    for seg in segments:  # a generator: decoding happens as we iterate, so Stop works
        # Whisper words keep their leading space (" Stanford", ".edu"); _join() relies on it.
        words = [Word(offset + w.start, offset + w.end, w.word) for w in seg.words or []
                 if (w.start + w.end) / 2 >= keep_from_sec]
        done = min(total, int(seg.end * SAMPLE_RATE))
        if not words:
            yield done, total, None
            continue
        text = _join(words) if len(words) < len(seg.words or []) else seg.text.strip()
        yield done, total, Segment(words[0].start, offset + seg.end, text, words)
    yield total, total, None


def _quietest(wav: np.ndarray, lo: int, hi: int, hop: int = 320) -> int:
    """Sample index of the lowest-energy frame in wav[lo:hi]."""
    n = (hi - lo) // hop
    energy = (wav[lo:lo + n * hop].reshape(n, hop) ** 2).mean(axis=1)
    return lo + int(np.argmin(energy)) * hop


def _whisper_chunks(model, spec: dict, wav: np.ndarray, offset: float, hotwords: list[str] | None,
                    keep_from_sec: float, search_sec: float = 4.0, silence_rms: float = 1e-3
                    ) -> Iterator[tuple[int, int, Segment | None]]:
    """Whisper on chunks of about spec["chunk_sec"], each cut at the quietest moment of its last
    search_sec; near-silent chunks are skipped (Whisper would invent text for them)."""
    size, search = int(spec["chunk_sec"] * SAMPLE_RATE), int(search_sec * SAMPLE_RATE)
    total, s = len(wav), int(keep_from_sec * SAMPLE_RATE)
    while s < total:
        e = total if total - s <= size else _quietest(wav, s + size - search, s + size)
        chunk = wav[s:e]
        words = []
        if len(chunk) > SAMPLE_RATE // 4 and float(np.sqrt((chunk ** 2).mean())) > silence_rms:
            segments, _ = model.transcribe(chunk, language=spec["language"], beam_size=1,
                                           without_timestamps=True, condition_on_previous_text=False,
                                           word_timestamps=True, vad_filter=False,
                                           hotwords=", ".join(hotwords) if hotwords else None)
            t0 = offset + s / SAMPLE_RATE
            words = [Word(t0 + w.start, t0 + w.end, w.word) for seg in segments for w in seg.words or []]
        yield e, total, (Segment(words[0].start, words[-1].end, _join(words), words) if words else None)
        s = e


SENTENCE_END = (".", "?", "!", "।", "॥")


def has_speakers(segments: list[Segment]) -> bool:
    return any(w.speaker is not None for seg in segments for w in seg.words)


def to_text(segments: list[Segment], pause_sec: float = 1.5, max_words: int = 60) -> str:
    """Join pieces into paragraphs: a new one after a pause, or once a paragraph is long
    (at a sentence end when there is punctuation, as with Whisper). With speakers labelled,
    one paragraph per turn: "Speaker 1: …"."""
    if has_speakers(segments):
        turns: list[tuple[int | None, list[Word]]] = []
        for w in (w for seg in segments for w in seg.words):
            if not turns or turns[-1][0] != w.speaker:
                turns.append((w.speaker, []))
            turns[-1][1].append(w)
        return "\n\n".join(f"Speaker {speaker}: {_join(words)}" for speaker, words in turns)
    paragraphs: list[list[str]] = []
    prev = None
    for seg in segments:
        words = sum(len(p.split()) for p in paragraphs[-1]) if paragraphs else 0
        new = (prev is None or seg.start - prev.end > pause_sec or words >= max_words
               or (words >= max_words // 2 and prev.text.endswith(SENTENCE_END)))
        if new:
            paragraphs.append([])
        paragraphs[-1].append(seg.text)
        prev = seg
    return "\n\n".join(" ".join(p) for p in paragraphs)


def _join(words: list[Word]) -> str:
    """Whisper words carry their own leading space; the Bangla decoder's words don't."""
    if any(w.text[:1].isspace() for w in words):
        return "".join(w.text for w in words).strip()
    return " ".join(w.text for w in words)


def _ts(sec: float) -> str:
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def _cues(segments: list[Segment], max_sec: float = 6.0, max_words: int = 12,
          max_gap: float = 0.8) -> list[tuple[float, float, str]]:
    """Group words into subtitle-sized cues, breaking on length, word count, pauses or a new
    speaker; a speaker's first cue starts with "Speaker N: "."""
    cues = []
    last_speaker = None

    def add(words: list[Word]):
        nonlocal last_speaker
        text, speaker = _join(words), words[0].speaker
        if speaker is not None and speaker != last_speaker:
            text = f"Speaker {speaker}: {text}"
        last_speaker = speaker
        cues.append((words[0].start, words[-1].end, text))

    for seg in segments:
        if not seg.words:
            cues.append((seg.start, seg.end, seg.text))
            continue
        cur: list[Word] = []
        for w in seg.words:
            if cur and (w.end - cur[0].start > max_sec or len(cur) >= max_words
                        or w.start - cur[-1].end > max_gap or w.speaker != cur[-1].speaker):
                add(cur)
                cur = []
            cur.append(w)
        add(cur)
    return cues


def to_srt(segments: list[Segment]) -> str:
    return "\n".join(
        f"{i}\n{_ts(start)} --> {_ts(end)}\n{text}\n"
        for i, (start, end, text) in enumerate(_cues(segments), 1)
    )
