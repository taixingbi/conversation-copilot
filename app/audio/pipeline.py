"""Audio session processing; communicates with AI only through on_final."""
from __future__ import annotations

import queue
import threading
import time
from collections import deque
from contextlib import ExitStack

import numpy as np
import sounddevice as sd

from audio.capture import is_ext_label, is_mic_label, is_speaker_echo, _norm_text
from audio.contracts import AudioFrame, TranscriptUpdate
from audio.speakers import SAMPLE_RATE
from events import BUS
from console import CYAN, DIM, paint, safe_print


class AudioPipeline:
    def __init__(self, sources, vads, worker, transcripts, metrics, *, on_final):
        self.sources = sorted(sources, key=lambda item: 0 if item[0] == "EXT" else 1)
        self.vads = vads
        self.worker = worker
        self.transcripts = transcripts
        self.metrics = metrics
        self.on_final = on_final
        self.recent_ext = deque()
        self.queues = {label: queue.Queue(maxsize=250) for label, _ in sources}
        self.received = {label: threading.Event() for label, _ in sources}
        self.pending = {label: np.empty(0, dtype=np.float32) for label, _ in sources}
        self.pending_end = {}
        self.overflows = {label: threading.Event() for label, _ in sources}

    def callback(self, source):
        def capture(indata, frames, timing, status):
            if status:
                self.overflows[source].set()
            try:
                self.queues[source].put_nowait(AudioFrame(source, indata[:, 0].copy(), time.monotonic()))
            except queue.Full:
                self.overflows[source].set()
            if frames and not status:
                self.received[source].set()
        return capture

    def accept(self, frame: AudioFrame):
        # Accumulate exactly one VAD window (32 ms), then transmit and commit
        # before consuming the next window, even when capture has a backlog.
        vad = self.vads[frame.source]
        samples = np.concatenate([self.pending[frame.source], frame.samples])
        offset = 0
        while offset + vad.window_size <= samples.size:
            window = samples[offset:offset + vad.window_size]
            offset += window.size
            end = frame.t_end - (samples.size - offset) / SAMPLE_RATE
            self._accept_window(frame.source, window, end)
        self.pending[frame.source] = samples[offset:].copy()
        self.pending_end[frame.source] = frame.t_end

    def _accept_window(self, source, samples, end):
        if hasattr(self.worker, "feed_frame"):
            self.worker.feed_frame(source, samples)
        for segment in self.vads[source].accept(samples):
            self.worker.submit(source, segment["audio"], t_end=end,
                               phase=segment["phase"], utterance_id=segment["utterance_id"])

    def handle(self, update: TranscriptUpdate):
        if update.error:
            BUS.publish("status", text=f"Recognition failed: {update.error}")
            return
        label = update.label
        if update.stt_ms:
            self.metrics.observe("stt_ms", update.stt_ms)
        if update.phase == "final" and update.text:
            norm = _norm_text(update.text)
            if is_ext_label(label):
                self.recent_ext.append((time.time(), norm))
            elif is_mic_label(label) and is_speaker_echo(norm, self.recent_ext):
                update = TranscriptUpdate(label, "", "final", update.utterance_id, update.ts)
        committed = self.transcripts.accept(update)
        if committed:
            meta = paint(f"{update.ts}  {label:<6}", CYAN if is_ext_label(label) else DIM)
            safe_print(f"{meta}  {update.text}")
            self.on_final(update)

    def drain(self):
        for update in self.worker.drain():
            self.handle(update)

    def run(self):
        with ExitStack() as stack:
            for label, idx in self.sources:
                stack.enter_context(sd.InputStream(device=idx, channels=1, dtype="float32",
                    samplerate=SAMPLE_RATE, blocksize=640, callback=self.callback(label)))
            BUS.publish("readiness", ready=False, text="Waiting for audio…")
            ready = False
            try:
                while True:
                    if not ready and all(event.is_set() for event in self.received.values()):
                        ready = True
                        BUS.publish("readiness", ready=True, text="Ready")
                        safe_print("✓ Ready — listening")
                    progressed = False
                    for label, _ in self.sources:
                        if self.overflows[label].is_set():
                            self.overflows[label].clear()
                            BUS.publish("status", text=f"Audio overflow on {label}; recognition is falling behind")
                        # Round-robin source processing prevents one device starving another.
                        try:
                            frame = self.queues[label].get_nowait()
                        except queue.Empty:
                            continue
                        self.accept(frame)
                        progressed = True
                    self.drain()
                    if not progressed:
                        time.sleep(0.01)
            except KeyboardInterrupt:
                pass
        # Inputs are closed before flushing, so shutdown cannot add more frames.
        for label, _ in self.sources:
            while not self.queues[label].empty():
                self.accept(self.queues[label].get_nowait())
            if self.pending[label].size:
                self._accept_window(label, self.pending[label], self.pending_end[label])
                self.pending[label] = np.empty(0, dtype=np.float32)
            for segment in self.vads[label].flush():
                self.worker.submit(label, segment["audio"], phase="final", utterance_id=segment["utterance_id"])
        self.worker.close()
        self.drain()
