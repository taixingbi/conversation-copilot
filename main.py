from __future__ import annotations

import argparse
import logging
import os
import queue
import re
import time
import warnings
from collections import deque
from contextlib import ExitStack
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import sounddevice as sd

from questions import (
    CYAN,
    DIM,
    YELLOW,
    QuestionExtractor,
    is_sure_question,
    paint,
    questions_path_for,
)
from speakers import SAMPLE_RATE, OnlineSpeakerTracker, SpeechVad, ensure_models, tracking_enabled
from stt import SttWorker, Transcriber, resolve_model_name

warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL.*")

LOOPBACK_HINTS = ("blackhole", "loopback", "soundflower", "vb-cable", "vb cable")
ECHO_WINDOW_SEC = 6.0
ECHO_RATIO = 0.72


def load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


_device_cache: list[tuple[int, dict]] | None = None


def input_devices():
    global _device_cache
    if _device_cache is None:
        devices = sd.query_devices()
        _device_cache = [(i, d) for i, d in enumerate(devices) if d["max_input_channels"] > 0]
    return _device_cache


def print_devices() -> None:
    default_in = sd.default.device[0]
    print("Audio input devices:")
    for i, d in input_devices():
        mark = " (default)" if i == default_in else ""
        print(f"  [{i}] {d['name']}  ({d['max_input_channels']} in){mark}")
    print("\nSet AUDIO_DEVICE / EXTERNAL_AUDIO_DEVICE to an index or name substring.")


def device_label(idx: int) -> str:
    for i, d in input_devices():
        if i == idx:
            return f"[{idx}] {d['name']}"
    return f"[{idx}]"


def resolve_device(spec: str | None, *, kind: str) -> int | None:
    """Resolve an index or case-insensitive name substring to an input device index."""
    if spec is None:
        return None
    spec = spec.strip()
    if not spec:
        return None

    devices = list(input_devices())
    if not devices:
        raise SystemExit("No audio input devices found.")

    if spec.isdigit():
        idx = int(spec)
        for i, d in devices:
            if i == idx:
                return idx
        raise SystemExit(f"{kind} device [{idx}] has no inputs.")

    needle = spec.lower()
    matches = [(i, d) for i, d in devices if needle in d["name"].lower()]
    if not matches:
        print_devices()
        raise SystemExit(f"No input device matching {spec!r} for {kind}.")
    if len(matches) > 1:
        names = ", ".join(f"[{i}] {d['name']}" for i, d in matches)
        raise SystemExit(f"Ambiguous {kind} device {spec!r}: {names}")
    return matches[0][0]


def default_mic_index() -> int:
    idx = sd.default.device[0]
    if idx is None or idx < 0:
        devices = input_devices()
        if not devices:
            raise SystemExit("No audio input devices found.")
        return devices[0][0]
    return idx


def auto_external_index(mic_idx: int) -> int | None:
    for i, d in input_devices():
        if i == mic_idx:
            continue
        name = d["name"].lower()
        if any(hint in name for hint in LOOPBACK_HINTS):
            return i
    return None


def make_callback(q: queue.Queue):
    def callback(indata, frames, t, status):
        if status:
            pass
        q.put(indata.copy())

    return callback


def is_ext_label(label: str) -> bool:
    return label == "EXT" or label.startswith("EXT-")


def is_mic_label(label: str) -> bool:
    return label == "MIC" or label.startswith("MIC-")


def emit(log_file, label: str, text: str) -> str:
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] [{label}] {text}\n"
    shown = f"[{ts}] [{label}] {text}"
    if is_ext_label(label):
        shown = paint(shown, CYAN)
    else:
        shown = paint(shown, DIM)
    print(shown, flush=True)
    log_file.write(line)
    log_file.flush()
    return ts


def _norm_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def is_speaker_echo(mic_n: str, recent_ext: deque) -> bool:
    """True when the webcam heard Zoom playing through speakers (same words as EXT)."""
    now = time.time()
    while recent_ext and now - recent_ext[0][0] > ECHO_WINDOW_SEC:
        recent_ext.popleft()
    if not mic_n:
        return True
    mic_tok = set(mic_n.split())
    for _, ext_n in recent_ext:
        if not ext_n:
            continue
        if mic_n == ext_n or mic_n in ext_n or ext_n in mic_n:
            return True
        ext_tok = set(ext_n.split())
        if not mic_tok or not ext_tok:
            continue
        if len(mic_tok & ext_tok) / min(len(mic_tok), len(ext_tok)) >= 0.7:
            return True
        if SequenceMatcher(None, mic_n, ext_n).ratio() >= ECHO_RATIO:
            return True
    return False


def drain_queue(q: queue.Queue) -> np.ndarray | None:
    chunks = []
    while True:
        try:
            chunks.append(q.get_nowait())
        except queue.Empty:
            break
    if not chunks:
        return None
    audio = np.concatenate(chunks, axis=0)
    if audio.ndim > 1:
        return audio[:, 0].astype(np.float32, copy=False)
    return audio.astype(np.float32, copy=False)


def make_extractor(transcribe_path: str) -> QuestionExtractor:
    url = (os.environ.get("FUNCTION_URL") or "").strip()
    key = (os.environ.get("INFERENCE_API_KEY") or os.environ.get("API_KEY") or "1234").strip()
    model = (os.environ.get("LLM_MODEL") or "qwen3-next-80b-a3b").strip()
    path = questions_path_for(transcribe_path)
    return QuestionExtractor(function_url=url, api_key=key, model=model, out_path=path)


def parse_args():
    p = argparse.ArgumentParser(description="Live transcribe mic + external audio")
    p.add_argument("--list-devices", action="store_true", help="List input devices and exit")
    p.add_argument("--audio-device", default=None, help="Mic device index or name")
    p.add_argument(
        "--external-audio-device",
        default=None,
        help="Loopback/system-audio device index or name",
    )
    return p.parse_args()


def main():
    args = parse_args()
    app_dir = Path(__file__).resolve().parent
    load_dotenv(app_dir / ".env")
    load_dotenv(app_dir.parent / ".env")

    if args.list_devices:
        print_devices()
        return

    mic_spec = args.audio_device if args.audio_device is not None else os.environ.get("AUDIO_DEVICE", "")
    ext_spec = (
        args.external_audio_device
        if args.external_audio_device is not None
        else os.environ.get("EXTERNAL_AUDIO_DEVICE", "")
    )

    mic_idx = resolve_device(mic_spec, kind="AUDIO")
    if mic_idx is None:
        mic_idx = default_mic_index()

    ext_idx = resolve_device(ext_spec, kind="EXTERNAL_AUDIO")
    if ext_idx is None:
        ext_idx = auto_external_index(mic_idx)
    if ext_idx == mic_idx:
        ext_idx = None

    sources = [("MIC", mic_idx)]
    if ext_idx is not None:
        sources.append(("EXT", ext_idx))

    track = tracking_enabled()
    vad_path, emb_path = ensure_models(app_dir / "models", need_embedding=track)
    vads = {label: SpeechVad(vad_path) for label, _ in sources}
    tracker = OnlineSpeakerTracker(emb_path) if track and emb_path is not None else None

    model_name = resolve_model_name()
    log_dir = os.environ.get("TRANSCRIBE_LOG_DIR", str(app_dir / "log"))
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_transcribe.txt")

    extractor = make_extractor(log_filename)

    logging.getLogger("pywhispercpp").setLevel(logging.ERROR)
    print(f"Loading Whisper {model_name} (metal greedy) ...", flush=True)
    worker = SttWorker(Transcriber(model_name), tracker)

    print(
        f"Listening  mic={device_label(mic_idx)}"
        + (f"  ext={device_label(ext_idx)}" if ext_idx is not None else "  ext=none")
        + (f"  speakers={'on' if tracker else 'off'}")
        + f"  stt={model_name}/metal",
        flush=True,
    )
    print(f"log   {log_filename}", flush=True)
    print(paint(f"qlog  {extractor.out_path}", YELLOW), flush=True)

    # EXT first so MIC can drop speaker-echo duplicates
    sources.sort(key=lambda item: 0 if item[0] == "EXT" else 1)
    queues = {label: queue.Queue() for label, _ in sources}
    recent_ext: deque = deque()
    last_said: dict[str, tuple[float, str]] = {}
    extractor.start()
    worker.start()

    with open(log_filename, "a", encoding="utf-8") as log_file, ExitStack() as stack:
        for label, idx in sources:
            stack.enter_context(
                sd.InputStream(
                    device=idx,
                    channels=1,
                    samplerate=SAMPLE_RATE,
                    callback=make_callback(queues[label]),
                )
            )

        def handle_texts(out_label: str, texts: list[str]) -> None:
            for text in texts:
                norm = _norm_text(text)
                prev = last_said.get(out_label)
                if prev and time.time() - prev[0] < 4.0 and (norm == prev[1] or norm in prev[1] or prev[1] in norm):
                    continue
                last_said[out_label] = (time.time(), norm)
                if is_ext_label(out_label):
                    recent_ext.append((time.time(), norm))
                elif is_mic_label(out_label) and recent_ext and is_speaker_echo(norm, recent_ext):
                    continue
                ts = emit(log_file, out_label, text)
                if is_ext_label(out_label) or is_sure_question(text):
                    extractor.add_ext(ts, text)

        try:
            while True:
                progressed = False
                for src_label, _ in sources:
                    audio = drain_queue(queues[src_label])
                    if audio is None:
                        continue
                    progressed = True
                    for seg in vads[src_label].accept(audio):
                        worker.submit(src_label, seg)
                for out_label, texts in worker.drain():
                    progressed = True
                    handle_texts(out_label, texts)
                if not progressed:
                    time.sleep(0.02)
        except KeyboardInterrupt:
            print("\nStopped.", flush=True)
        finally:
            worker.close()
            extractor.close()


if __name__ == "__main__":
    main()
