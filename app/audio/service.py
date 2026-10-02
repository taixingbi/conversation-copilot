"""Owns the recognition worker and model reload lifecycle."""
from __future__ import annotations

import threading

from audio.stt import Transcriber, resolve_backend, resolve_model_name
from events import BUS


class AudioService:
    def __init__(self, worker, *, on_change):
        self.worker = worker
        self.on_change = on_change
        self.loading = False
        self._lock = threading.Lock()
        self._generation = 0

    def reload(self, key: str) -> None:
        if self.worker is None:
            return
        name = resolve_model_name(key)
        current = getattr(self.worker.stt, "model_name", "")
        if current == name and not self.loading:
            return
        with self._lock:
            self._generation += 1
            gen = self._generation
            self.loading = True
        BUS.publish("status", text=f"loading whisper {key}…")

        def load() -> None:
            try:
                backend = resolve_backend(name)
                stt = Transcriber(name, backend)
                with self._lock:
                    if gen != self._generation:
                        return
                    self.worker.set_transcriber(stt)
                    self.loading = False
                BUS.publish("status", text=f"whisper {key}/{backend}")
                self.on_change()
            except Exception as exc:
                with self._lock:
                    if gen == self._generation:
                        self.loading = False
                BUS.publish("status", text=f"whisper failed: {exc}")
                self.on_change()

        threading.Thread(target=load, daemon=True, name="whisper-reload").start()
