"""Canonical reader for finalized session speech.

The persisted session log is authoritative. Live replay is a final-only
fallback for sessions with no persisted speech yet.
"""
from __future__ import annotations

import threading
from pathlib import Path

from events import BUS


class TranscriptStore:
    def __init__(self, path: Path | None = None, *, bus=None):
        self._lock = threading.RLock()
        self._finalized = set()
        self.path = Path(path) if path is not None else None
        self.bus = bus if bus is not None else BUS

    def text(self) -> str:
        with self._lock:
            return self._text()

    def _text(self) -> str:
        if self.path is not None:
            try:
                if self.path.is_file():
                    text = self.path.read_text(encoding="utf-8").strip()
                    if text:
                        return text
            except OSError:
                pass
        return "\n".join(
            f"[{ev.get('ts') or ''}] [{ev.get('label') or ''}] {ev['text']}"
            for ev in self.bus.snapshot()
            if ev.get("type") == "transcript" and ev.get("phase", "final") == "final" and ev.get("text")
        )

    def accept(self, update) -> bool:
        """Publish revisions; persist each nonempty final exactly once."""
        from dataclasses import asdict
        with self._lock:
            uid = update.utterance_id
            if uid is not None and uid in self._finalized:
                return False
            committed = update.phase == "final" and bool(update.text)
            if update.phase == "final":
                if committed and self.path is not None:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    with self.path.open("a", encoding="utf-8") as file:
                        file.write(f"[{update.ts}] [{update.label}] {update.text}\n")
                        file.flush()
                if uid is not None:
                    self._finalized.add(uid)
            self.bus.publish("transcript", **asdict(update))
            return committed
