import sys
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from speakers import SpeechVad, OnlineSpeakerTracker
from stt import SttWorker, Transcriber, usable_audio


class RecognitionTests(unittest.TestCase):
    def test_audio_validation_and_stereo_duration(self):
        stereo = np.full((1600, 2), 0.1, dtype=np.float32)
        mono = usable_audio(stereo)
        self.assertEqual(mono.shape, (1600,))
        np.testing.assert_allclose(mono, 0.1)
        for value in (0, np.nan, np.inf, 1e30):
            with np.errstate(all="raise"):
                self.assertIsNone(usable_audio(np.full(1600, value)))

    def test_submission_owns_audio(self):
        worker = SttWorker(Mock())
        audio = np.full(1600, 0.1, dtype=np.float32)
        worker.submit("MIC", audio)
        audio[:] = 0
        np.testing.assert_allclose(worker.jobs.get_nowait()[1], 0.1)

    def test_worker_recovers_after_failure(self):
        transcriber = Mock()
        transcriber.transcribe.side_effect = [ValueError("bad features"), ["hello"]]
        tracker = Mock()
        tracker.assign.side_effect = RuntimeError("bad embedding")
        worker = SttWorker(transcriber, tracker)
        audio = np.full(1600, 0.1, dtype=np.float32)
        worker.submit("MIC", audio)
        worker.submit("EXT", audio)
        with self.assertLogs("stt", level="ERROR"):
            worker.start()
            try:
                first = worker.results.get(timeout=2)
                second = worker.results.get(timeout=2)
            finally:
                worker.close()
        self.assertEqual(first[2]["error"], "bad features")
        self.assertEqual(second[:2], ("EXT", ["hello"]))

    def make_vad(self):
        vad = SpeechVad.__new__(SpeechVad)
        vad.vad = Mock()
        vad.vad.empty.return_value = True
        vad.vad.is_speech_detected.return_value = True
        vad.buf = np.empty(0, dtype=np.float32)
        vad._parts = []
        vad._preroll = deque(maxlen=3)
        vad._n = vad._emitted = 0
        vad.window_size = vad.min_samples = vad.chunk_samples = 4
        vad.max_samples = 16
        return vad

    def test_speech_onset_is_preserved(self):
        vad = self.make_vad()
        vad.vad.is_speech_detected.return_value = False
        self.assertEqual(vad.accept(np.full(4, 0.2)), [])
        vad.vad.is_speech_detected.return_value = True
        segments = vad.accept(np.full(4, 0.4))
        self.assertEqual(len(segments), 1)
        np.testing.assert_allclose(segments[0], [0.2] * 4 + [0.4] * 4)

    def test_warmup_consumes_lazy_inference(self):
        transcriber = Transcriber.__new__(Transcriber)
        transcriber.backend = "faster"
        consumed = []
        def segments():
            consumed.append(True)
            yield Mock(text="Hello", no_speech_prob=0)
        transcriber.model = Mock()
        transcriber.model.transcribe.return_value = (segments(), None)
        transcriber.warmup()
        self.assertEqual(consumed, [True])

    def test_exact_frame_and_no_duplicate_final(self):
        vad = self.make_vad()
        self.assertEqual(len(vad.accept(np.ones(4))), 1)
        self.assertEqual(vad.buf.size, 0)
        vad.vad.is_speech_detected.return_value = False
        self.assertEqual(vad.accept(np.zeros(4)), [])
        self.assertEqual(vad.flush(), [])

    def test_final_includes_new_tail(self):
        vad = self.make_vad()
        vad.chunk_samples = 8
        self.assertEqual(len(vad.accept(np.ones(8))), 1)
        self.assertEqual(vad.accept(np.ones(4)), [])
        final = vad.flush()
        self.assertEqual(len(final), 1)
        self.assertEqual(final[0].size, 12)
        self.assertEqual(vad.flush(), [])

    def test_sticky_speaker_does_not_corrupt_centroid(self):
        tracker = OnlineSpeakerTracker.__new__(OnlineSpeakerTracker)
        tracker.threshold = 0.5
        tracker.max_speakers = 4
        tracker._new_max = 0.38
        tracker._new_min_sec = 1
        tracker._stick_sec = 3
        tracker._now = lambda: 1
        tracker._pool = {"MIC": ["MIC-0", "MIC-1"]}
        tracker._last_name = {"MIC": "MIC-0"}
        tracker._last_t = {"MIC": 0}
        tracker.centroids = {"MIC-0": np.array([1., 0.]), "MIC-1": np.array([0., 1.])}
        tracker.counts = {"MIC-0": 1, "MIC-1": 1}
        tracker.embed = lambda *args: np.array([0., 1.])
        self.assertEqual(tracker.assign(np.ones(1600), prefix="MIC"), "MIC-0")
        np.testing.assert_array_equal(tracker.centroids["MIC-0"], [1., 0.])
        self.assertEqual(tracker.counts["MIC-0"], 1)


if __name__ == "__main__":
    unittest.main()
