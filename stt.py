from __future__ import annotations

import os
import queue
import threading

import numpy as np

from noise import english_only, is_junk_line

MODEL_ALIASES = {
    "medium": "medium.en",
    "medium.en": "medium.en",
    "large": "large-v3",
    "large-v3": "large-v3",
}
MIN_SAMPLES = 1600  # 0.1s
RMS_MIN = 0.003


def resolve_model_name() -> str:
    raw = os.environ.get("WHISPER_MODEL", "medium").strip().lower()
    if raw not in MODEL_ALIASES:
        allowed = ", ".join(sorted(MODEL_ALIASES))
        raise SystemExit(f"Unknown WHISPER_MODEL={raw!r}. Use one of: {allowed}")
    return MODEL_ALIASES[raw]


def usable_audio(audio: np.ndarray) -> np.ndarray | None:
    samples = np.ascontiguousarray(audio, dtype=np.float32).reshape(-1)
    if samples.size < MIN_SAMPLES:
        return None
    if not np.isfinite(samples).all():
        return None
    if float(np.sqrt(np.mean(np.square(samples)))) < RMS_MIN:
        return None
    return samples


class Transcriber:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or resolve_model_name()
        self.backend = "metal"
        self.model = self._init_metal()

    def _init_metal(self):
        from pywhispercpp.model import Model

        return Model(
            self.model_name,
            params_sampling_strategy=0,  # greedy
            language="en",
            print_progress=False,
            print_realtime=False,
            print_special=False,
            suppress_nst=True,
            n_threads=min(4, os.cpu_count() or 4),
            beam_search={"beam_size": 1, "patience": 1.0},
            context_params={"use_gpu": True},
            redirect_whispercpp_logs_to=None,
        )

    def transcribe(self, audio: np.ndarray) -> list[str]:
        samples = usable_audio(audio)
        if samples is None:
            return []
        texts: list[str] = []
        for seg in self.model.transcribe(samples):
            text = english_only((getattr(seg, "text", "") or "").strip())
            if text and not is_junk_line(text):
                texts.append(text)
        return texts


class SttWorker:
    """Embed + Whisper off the capture/VAD thread. Drops oldest jobs if behind."""

    def __init__(self, stt: Transcriber, tracker=None, *, max_jobs: int = 4):
        self.stt = stt
        self.tracker = tracker
        self.jobs: queue.Queue = queue.Queue(maxsize=max_jobs)
        self.results: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="stt-worker")

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def submit(self, prefix: str, audio: np.ndarray) -> None:
        samples = usable_audio(audio)
        if samples is None:
            return
        try:
            self.jobs.put_nowait((prefix, samples))
        except queue.Full:
            try:
                self.jobs.get_nowait()
            except queue.Empty:
                pass
            try:
                self.jobs.put_nowait((prefix, samples))
            except queue.Full:
                pass

    def drain(self) -> list[tuple[str, list[str]]]:
        out: list[tuple[str, list[str]]] = []
        while True:
            try:
                out.append(self.results.get_nowait())
            except queue.Empty:
                break
        return out

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                prefix, audio = self.jobs.get(timeout=0.05)
            except queue.Empty:
                continue
            label = self.tracker.assign(audio, prefix=prefix) if self.tracker else prefix
            self.results.put((label, self.stt.transcribe(audio)))
