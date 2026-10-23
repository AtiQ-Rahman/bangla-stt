"""Who speaks when (speaker diarization) with sherpa-onnx, on CPU, without any account.

Uses the pyannote 3.0 segmentation model and a WeSpeaker voice-embedding model, downloaded once
from sherpa-onnx's GitHub releases into ~/.cache/bangla-stt/diarization.
"""
import tarfile
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np

from stt.activity import doing
from stt.transcribe import Segment, Word

MODEL_DIR = Path.home() / ".cache" / "bangla-stt" / "diarization"
RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
SEGMENTATION = MODEL_DIR / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
EMBEDDING = MODEL_DIR / "wespeaker_en_voxceleb_resnet34_LM.onnx"


def _download():
    if SEGMENTATION.exists() and EMBEDDING.exists():
        return
    with doing("🌐 Downloading the speaker models (first time only)…"):
        _fetch()


def _fetch():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not SEGMENTATION.exists():
        archive = MODEL_DIR / "segmentation.tar.bz2"
        urllib.request.urlretrieve(
            f"{RELEASES}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2", archive)
        with tarfile.open(archive) as tar:
            tar.extractall(MODEL_DIR, filter="data")
        archive.unlink()
    if not EMBEDDING.exists():
        part = EMBEDDING.with_suffix(".part")
        urllib.request.urlretrieve(f"{RELEASES}/speaker-recongition-models/{EMBEDDING.name}", part)
        part.rename(EMBEDDING)


@lru_cache(maxsize=1)
def _diarizer():
    import sherpa_onnx

    _download()
    with doing("🧠 Loading the speaker models…"):
        return _build(sherpa_onnx)


def _build(sherpa_onnx):
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(SEGMENTATION))),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(EMBEDDING)),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=0.5),
        min_duration_on=0.3, min_duration_off=0.5)
    return sherpa_onnx.OfflineSpeakerDiarization(config)


def diarize(wav: np.ndarray, offset: float = 0.0) -> list[tuple[float, float, int]]:
    """(start, end, speaker) turns for 16 kHz audio; speakers are numbered 1, 2, … in order of
    first appearance; offset (seconds) is added to the times."""
    turns = _diarizer().process(wav).sort_by_start_time()
    numbers: dict[int, int] = {}
    out = []
    for turn in turns:
        speaker = numbers.setdefault(turn.speaker, len(numbers) + 1)
        out.append((offset + turn.start, offset + turn.end, speaker))
    return out


def label(segments: list[Segment], turns: list[tuple[float, float, int]]) -> list[Segment]:
    """The segments with every word's speaker set to the turn it overlaps most (or the nearest)."""
    if not turns:
        return segments

    def speaker_at(start: float, end: float) -> int:
        best, best_overlap, nearest, gap = turns[0][2], 0.0, turns[0][2], float("inf")
        for t_start, t_end, speaker in turns:
            overlap = min(end, t_end) - max(start, t_start)
            if overlap > best_overlap:
                best, best_overlap = speaker, overlap
            distance = max(t_start - end, start - t_end, 0.0)
            if distance < gap:
                nearest, gap = speaker, distance
        return best if best_overlap > 0 else nearest

    return [Segment(seg.start, seg.end, seg.text,
                    [Word(w.start, w.end, w.text, speaker_at(w.start, max(w.end, w.start + 0.05)))
                     for w in seg.words])
            for seg in segments]
