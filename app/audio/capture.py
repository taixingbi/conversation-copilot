from __future__ import annotations

import re
import time
from collections import deque
from difflib import SequenceMatcher

import sounddevice as sd

LOOPBACK_HINTS = ("blackhole", "loopback", "soundflower", "vb-cable", "vb cable")
ECHO_WINDOW_SEC = 6.0
ECHO_RATIO = 0.72

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


def is_ext_label(label: str) -> bool:
    return label == "EXT" or label.startswith("EXT-")


def is_mic_label(label: str) -> bool:
    return label == "MIC" or label.startswith("MIC-")


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
