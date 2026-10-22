"""Live dictation: microphone audio arrives in small chunks; each phrase (speech up to a pause)
is transcribed in the background as soon as it ends, so text appears while the user speaks."""
import queue
import threading
from math import gcd
from typing import Callable

import numpy as np
from scipy.signal import resample_poly

from stt.audio import SAMPLE_RATE
from stt.transcribe import _quietest, Segment, to_text

FRAME = SAMPLE_RATE // 50  # 20 ms
PAUSE_SEC = 0.6  # this much quiet after speech ends a phrase
MAX_SEC = 10.0  # a longer phrase is cut at its quietest moment (text shows up sooner)
MIN_SPEECH_SEC = 0.3


def to_16k_mono(sr: int, chunk: np.ndarray) -> np.ndarray:
    """A browser chunk (int16 or float, mono or stereo, any rate) as 16 kHz mono float32."""
    chunk = np.asarray(chunk)
    if chunk.dtype.kind in "iu":
        chunk = chunk.astype(np.float32) / np.iinfo(chunk.dtype).max
    chunk = chunk.astype(np.float32)
    if chunk.ndim == 2:
        chunk = chunk.mean(axis=1)
    if sr != SAMPLE_RATE:
        g = gcd(sr, SAMPLE_RATE)
        chunk = resample_poly(chunk, SAMPLE_RATE // g, sr // g).astype(np.float32)
    return chunk


class Dictation:
    """transcribe(wav, offset_sec) -> list of segments, run in order on a background thread."""

    def __init__(self, transcribe: Callable[[np.ndarray, float], list[Segment]]):
        self._transcribe = transcribe
        self._buf = np.zeros(SAMPLE_RATE * 60, np.float32)  # grows by doubling
        self.n = 0  # samples recorded
        self.cursor = 0  # where the current phrase starts (samples)
        self.segments: list[Segment] = []
        self.error: str | None = None
        self._phrases: queue.Queue = queue.Queue()
        self._busy = 0  # phrases queued or being transcribed
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @property
    def audio(self) -> np.ndarray:
        """The whole recording so far."""
        return self._buf[:self.n]

    # -- audio in --
    def feed(self, sr: int, chunk: np.ndarray):
        chunk = to_16k_mono(sr, chunk)
        if self.n + len(chunk) > len(self._buf):
            bigger = np.zeros(max(2 * len(self._buf), self.n + len(chunk)), np.float32)
            bigger[:self.n] = self.audio
            self._buf = bigger  # (a phrase being transcribed keeps its view of the old one)
        self._buf[self.n:self.n + len(chunk)] = chunk
        self.n += len(chunk)
        self._split()

    def _loud(self, start: int) -> np.ndarray:
        """Per 20 ms frame from `start`: louder than the background noise (judged on the last
        20 s, which have pauses in them)?"""
        def rms(wav):
            k = len(wav) // FRAME
            return np.sqrt((wav[:k * FRAME].reshape(k, FRAME) ** 2).mean(axis=1))
        recent = rms(self.audio[max(0, self.n - 20 * SAMPLE_RATE):])
        floor = float(np.percentile(recent, 10)) if len(recent) else 0.0
        # (capped: in a long stretch without pauses the "background" is the voice itself)
        return rms(self.audio[start:]) > max(0.006, min(3 * floor, 0.04))

    def _split(self, final: bool = False):
        """Queue every phrase that has ended (all that is left, if final)."""
        pause = int(PAUSE_SEC * SAMPLE_RATE / FRAME)
        while True:
            loud = self._loud(self.cursor)
            if not loud.any():
                if self.n - self.cursor > 3 * SAMPLE_RATE:  # only quiet so far: skip it
                    self.cursor = self.n - SAMPLE_RATE // 2
                return
            last = len(loud) - 1 - int(np.argmax(loud[::-1]))  # the last loud frame
            spoken = loud.sum() * FRAME / SAMPLE_RATE
            if final:
                end = self.n
            elif len(loud) - 1 - last >= pause and spoken >= MIN_SPEECH_SEC:
                end = self.cursor + (last + pause // 2) * FRAME
            elif self.n - self.cursor >= MAX_SEC * SAMPLE_RATE:
                end = _quietest(self.audio, self.cursor + int((MAX_SEC - 4) * SAMPLE_RATE),
                                self.cursor + int(MAX_SEC * SAMPLE_RATE))
            else:
                return
            if spoken >= MIN_SPEECH_SEC:
                self._busy += 1
                self._phrases.put((self.cursor, end))
            self.cursor = end
            if final:
                return

    def finish(self, timeout: float = 120.0):
        """The recording ended: transcribe what is left and wait for all of it."""
        self._split(final=True)
        self._phrases.put(None)
        self._thread.join(timeout)

    # -- text out --
    def _run(self):
        while (item := self._phrases.get()) is not None:
            start, end = item
            try:
                self.segments += self._transcribe(self.audio[start:end], start / SAMPLE_RATE)
            except Exception as e:
                self.error = f"{type(e).__name__}: {e}"
            finally:
                self._busy -= 1

    @property
    def pending(self) -> bool:
        """Speech heard but not written yet."""
        return bool(self._busy) or bool(self._loud(self.cursor).any())

    def text(self) -> str:
        return to_text(self.segments)
