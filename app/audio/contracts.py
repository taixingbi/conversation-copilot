"""Typed messages crossing capture, ASR, transcript and application boundaries."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class AudioFrame:
    source: str
    samples: np.ndarray
    t_end: float


@dataclass(frozen=True)
class TranscriptUpdate:
    label: str
    text: str
    phase: Literal["partial", "final"]
    utterance_id: str | None
    ts: str
    seg_end: float | None = None
    stt_ms: float = 0.0
    error: str | None = None
