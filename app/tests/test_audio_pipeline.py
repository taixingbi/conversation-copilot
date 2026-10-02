import queue
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audio.contracts import AudioFrame, TranscriptUpdate
from audio.pipeline import AudioPipeline
from audio.stt import RecognitionQueue
from audio.transcripts import TranscriptStore
from events import EventBus
from metrics import LatencyTracker


class AudioPipelineTests(unittest.TestCase):
    def test_backlog_commit_precedes_next_audio_window(self):
        actions = []
        vad = Mock(window_size=512, buf=np.empty(0))
        vad.accept.side_effect = [
            [{"audio": np.ones(512), "phase": "final", "utterance_id": "1"}], []]
        worker = Mock()
        worker.feed_frame.side_effect = lambda source, samples: actions.append("audio")
        worker.submit.side_effect = lambda *args, **kwargs: actions.append("commit")
        pipeline = AudioPipeline([("MIC", 0)], {"MIC": vad}, worker,
                                 TranscriptStore(bus=EventBus()), LatencyTracker(), on_final=Mock())
        pipeline.accept(AudioFrame("MIC", np.ones(1024), 100))
        self.assertEqual(actions, ["audio", "commit", "audio"])
        self.assertAlmostEqual(worker.submit.call_args.kwargs["t_end"], 99.968)

    def test_store_commits_final_once_and_rejects_late_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "transcript.txt"
            bus = EventBus()
            store = TranscriptStore(path, bus=bus)
            partial = TranscriptUpdate("EXT", "Bed rock", "partial", "EXT:1", "10:00:00")
            final = TranscriptUpdate("EXT", "Bedrock", "final", "EXT:1", "10:00:00")
            self.assertFalse(store.accept(partial))
            self.assertFalse(path.exists())
            self.assertTrue(store.accept(final))
            self.assertFalse(store.accept(final))
            self.assertFalse(store.accept(partial))
            self.assertEqual(path.read_text(), "[10:00:00] [EXT] Bedrock\n")
            self.assertEqual(bus.snapshot()[0]["phase"], "final")

    def test_partial_never_triggers_ai(self):
        on_final = Mock()
        pipeline = AudioPipeline([], {}, Mock(), TranscriptStore(bus=EventBus()),
                                 LatencyTracker(), on_final=on_final)
        pipeline.handle(TranscriptUpdate("EXT", "hello", "partial", "EXT:1", "10:00:00"))
        on_final.assert_not_called()
        pipeline.handle(TranscriptUpdate("EXT", "hello", "final", "EXT:1", "10:00:00"))
        on_final.assert_called_once()

    def test_final_backpressure_preserves_order_and_capacity(self):
        jobs = RecognitionQueue(1)
        first = ("MIC", None, 0, "final", "1")
        second = ("MIC", None, 1, "final", "2")
        jobs.submit(first, active=False)
        started, done = threading.Event(), threading.Event()
        def producer():
            started.set()
            jobs.submit(second, active=True)
            done.set()
        thread = threading.Thread(target=producer)
        thread.start()
        try:
            self.assertTrue(started.wait(1))
            self.assertFalse(done.wait(0.02))
            self.assertEqual(jobs.qsize(), 1)
            self.assertEqual(jobs.get_nowait(), first)
            self.assertTrue(done.wait(1))
            self.assertEqual(jobs.get_nowait(), second)
        finally:
            if not jobs.empty():
                jobs.get_nowait()
            thread.join(timeout=1)


if __name__ == "__main__":
    unittest.main()
