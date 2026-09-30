from __future__ import annotations

import os
import logging
import queue
import threading
import time

import numpy as np

from noise import english_only, is_junk_line

MODEL_ALIASES = {
    "tiny": "tiny.en",
    "tiny.en": "tiny.en",
    "base": "base.en",
    "base.en": "base.en",
    "small": "small.en",
    "small.en": "small.en",
    "medium": "medium.en",
    "medium.en": "medium.en",
    "large": "large-v3",
    "large-v3": "large-v3",
    "distil": "distil-small.en",
    "distil-small": "distil-small.en",
    "distil-small.en": "distil-small.en",
    "distil-medium": "distil-medium.en",
    "distil-medium.en": "distil-medium.en",
    "distil-large": "distil-large-v3",
    "distil-large-v3": "distil-large-v3",
}
DISTIL = {v for k, v in MODEL_ALIASES.items() if "distil" in k}
MIN_SAMPLES = 1600  # 0.1s
RMS_MIN = 0.003
logger = logging.getLogger(__name__)


class CheckedFeatureExtractor:
    """Contain NumPy floating-point flags and validate Whisper's features."""

    def __init__(self, extractor):
        self.extractor = extractor

    def __getattr__(self, name):
        return getattr(self.extractor, name)

    def __call__(self, *args, **kwargs):
        # NumPy's macOS matmul can report divide/overflow/invalid flags even
        # when its result agrees with a direct, finite calculation. Limit the
        # workaround to extraction and reject genuinely invalid features.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            features = self.extractor(*args, **kwargs)
        if not np.isfinite(features).all():
            raise ValueError("Whisper feature extraction produced non-finite values")
        return features


def resolve_model_name(raw: str | None = None) -> str:
    value = (raw if raw is not None else os.environ.get("WHISPER_MODEL", "medium")).strip().lower()
    if value not in MODEL_ALIASES:
        allowed = ", ".join(sorted(MODEL_ALIASES))
        raise ValueError(f"Unknown WHISPER_MODEL={value!r}. Use one of: {allowed}")
    return MODEL_ALIASES[value]


def resolve_backend(model_name: str | None = None) -> str:
    raw = (os.environ.get("WHISPER_BACKEND") or "metal").strip().lower()
    name = model_name or resolve_model_name()
    if raw in {"faster", "faster-whisper", "int8"}:
        return "faster"
    if raw == "auto":
        if name in DISTIL:
            return "faster"
        try:
            import faster_whisper  # noqa: F401

            return "faster"
        except ImportError:
            return "metal"
    if name in DISTIL and raw == "metal":
        return "faster"
    return "metal"


def usable_audio(audio: np.ndarray) -> np.ndarray | None:
    samples = np.asarray(audio)
    if samples.ndim not in (1, 2) or (samples.ndim == 2 and samples.shape[1] == 0):
        return None
    if not np.isfinite(samples).all():
        return None
    # Capture supplies normalized floating-point PCM. Reject corrupt finite
    # values before squaring; flattening stereo would also change its timing.
    if samples.size and (samples.min() < -1.0 or samples.max() > 1.0):
        return None
    if samples.ndim == 2:
        samples = samples.mean(axis=1, dtype=np.float32)
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    if samples.size < MIN_SAMPLES:
        return None
    if not np.isfinite(samples).all():
        return None
    if float(np.sqrt(np.mean(np.square(samples)))) < RMS_MIN:
        return None
    return samples


class Transcriber:
    def __init__(self, model_name: str | None = None, backend: str | None = None):
        self.model_name = model_name or resolve_model_name()
        self.backend = backend or resolve_backend(self.model_name)
        if self.backend == "faster":
            self.model = self._init_faster()
        else:
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

    def _init_faster(self):
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise SystemExit(
                "WHISPER_BACKEND=faster needs faster-whisper. "
                "Install with: pip install faster-whisper"
            ) from exc
        device = (os.environ.get("WHISPER_DEVICE") or "cpu").strip() or "cpu"
        compute = (os.environ.get("WHISPER_COMPUTE") or "int8").strip() or "int8"
        model = WhisperModel(self.model_name, device=device, compute_type=compute)
        model.feature_extractor = CheckedFeatureExtractor(model.feature_extractor)
        return model

    def warmup(self) -> None:
        """Run inference before advertising readiness; discard its output."""
        samples = np.zeros(16000, dtype=np.float32)
        if self.backend == "faster":
            self._transcribe_faster(samples)
        else:
            self._transcribe_metal(samples)

    def transcribe(self, audio: np.ndarray) -> list[str]:
        samples = usable_audio(audio)
        if samples is None:
            return []
        if self.backend == "faster":
            return self._transcribe_faster(samples)
        return self._transcribe_metal(samples)

    def _keep(self, text: str) -> str:
        text = english_only((text or "").strip())
        if text and not is_junk_line(text):
            return text
        return ""

    def _confident(self, seg) -> bool:
        try:
            need = float(os.environ.get("RECOGNITION_CONFIDENCE") or "0.70")
        except ValueError:
            need = 0.70
        need = min(0.99, max(0.0, need))
        no_speech = getattr(seg, "no_speech_prob", None)
        if no_speech is not None:
            return (1.0 - float(no_speech)) >= need
        prob = getattr(seg, "probability", None)
        if prob is None:
            prob = getattr(seg, "p", None)
        if prob is None:
            return True
        return float(prob) >= need

    def _transcribe_metal(self, samples: np.ndarray) -> list[str]:
        texts: list[str] = []
        for seg in self.model.transcribe(samples):
            if not self._confident(seg):
                continue
            text = self._keep(getattr(seg, "text", "") or "")
            if text:
                texts.append(text)
        return texts

    def _transcribe_faster(self, samples: np.ndarray) -> list[str]:
        segments, _info = self.model.transcribe(
            samples,
            language="en",
            beam_size=1,
            vad_filter=False,
            without_timestamps=True,
            condition_on_previous_text=False,
        )
        texts: list[str] = []
        for seg in segments:
            if not self._confident(seg):
                continue
            text = self._keep(getattr(seg, "text", "") or "")
            if text:
                texts.append(text)
        return texts


class SttWorker:
    """Embed + Whisper off the capture/VAD thread. Drops oldest jobs if behind."""

    def __init__(self, stt: Transcriber, tracker=None, *, max_jobs: int = 4):
        self.stt = stt
        self.tracker = tracker
        self.jobs: queue.Queue = queue.Queue(maxsize=max_jobs)
        self.results: queue.Queue = queue.Queue()
        self._stt_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="stt-worker")

    def set_transcriber(self, stt: Transcriber) -> None:
        with self._stt_lock:
            self.stt = stt

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def submit(self, prefix: str, audio: np.ndarray, *, t_end: float | None = None) -> None:
        samples = usable_audio(audio)
        if samples is None:
            return
        job = (prefix, samples.copy(), t_end if t_end is not None else time.monotonic())
        try:
            self.jobs.put_nowait(job)
        except queue.Full:
            try:
                self.jobs.get_nowait()
            except queue.Empty:
                pass
            try:
                self.jobs.put_nowait(job)
            except queue.Full:
                pass

    def drain(self) -> list[tuple[str, list[str], dict]]:
        out: list[tuple[str, list[str], dict]] = []
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
                prefix, audio, t_end = self.jobs.get(timeout=0.05)
            except queue.Empty:
                continue
            label = prefix
            if self.tracker:
                try:
                    label = self.tracker.assign(audio, prefix=prefix)
                except Exception:
                    logger.exception("Speaker tracking failed for %s; keeping source label", prefix)
            t0 = time.perf_counter()
            with self._stt_lock:
                stt = self.stt
            try:
                texts = stt.transcribe(audio)
            except Exception as exc:
                logger.exception("Transcription failed for %s", prefix)
                self.results.put((label, [], {"error": str(exc), "seg_end": t_end}))
                continue
            stt_ms = (time.perf_counter() - t0) * 1000
            self.results.put((label, texts, {"stt_ms": stt_ms, "seg_end": t_end}))
