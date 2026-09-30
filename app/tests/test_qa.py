import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from events import EventBus
from questions import QuestionExtractor
from settings import Runtime


class QaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.bus = EventBus()
        self.bus_patch = patch("questions.BUS", self.bus)
        self.bus_patch.start()
        self.addCleanup(self.bus_patch.stop)
        self.extractor = QuestionExtractor(
            function_url="https://example.test", api_key="test", model="nova-pro",
            out_path=Path(self.tmp.name) / "questions.txt",
        )

    def statuses(self):
        return [e["text"] for e in self.bus.snapshot() if e["type"] == "qa_status"]

    def test_completed_answer_is_published_once_with_line_breaks(self):
        answer = "First paragraph.\n\n- One\n- Two"
        self.extractor.llm.iter_qa = Mock(return_value=iter([("Explain Python", answer)]))
        self.extractor._run_job(["Explain Python"], "final", 0)
        events = [e for e in self.bus.snapshot() if e["type"] == "qa"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["answer"], answer)
        self.assertTrue(events[0]["done"])

    def test_cancel_during_inference_prevents_late_card(self):
        def cancelled(*args, **kwargs):
            self.extractor.set_auto(False)
            yield "What is Python?", "Python is a language."
        self.extractor.llm.iter_qa = cancelled
        self.extractor._run_job(["What is Python?"], "final", 0)
        self.assertFalse(any(e["type"] == "qa" for e in self.bus.snapshot()))
        self.assertFalse(self.extractor.out_path.exists())

    def test_all_nonempty_speech_reaches_model_without_content_rules(self):
        self.extractor.set_auto(True)
        with patch.object(self.extractor, "_spawn", side_effect=self.extractor._run_job):
            for text in ("Hi", "Why", "Python", "the tradeoffs please", "What is", "能介绍一下吗", "It is a language."):
                self.extractor.llm.iter_qa = Mock(return_value=iter([]))
                self.extractor.add_ext("00:00:00", text)
                self.extractor.flush(force=True)
                self.assertIn(text, self.extractor.llm.iter_qa.call_args.args[0])

    def test_new_speech_waits_for_current_request(self):
        self.extractor.set_auto(True)
        with patch.object(self.extractor, "_spawn") as spawn:
            self.extractor.add_ext("00:00:00", "Explain Python")
            self.extractor.flush(force=True)
            self.extractor.add_ext("00:00:01", "And Java")
            self.extractor.flush(force=True)
            self.assertEqual(spawn.call_count, 1)
            self.assertIn("And Java", self.extractor._queued_window)
            self.extractor._final_alive = False
            self.extractor.flush(force=True)
            self.assertEqual(spawn.call_count, 2)
            self.assertIn("And Java", spawn.call_args.args[0])

    def test_one_second_cadence_and_five_second_window(self):
        self.extractor.set_auto(True)
        with patch("questions.now_mono", return_value=100) as clock, patch.object(self.extractor, "_spawn") as spawn:
            self.extractor.add_ext("00:00:00", "old speech", t_mono=94.9)
            self.extractor.add_ext("00:00:01", "recent speech", t_mono=95.1)
            self.extractor.flush()
            self.assertEqual(spawn.call_args.args[0], ["recent speech"])
            self.extractor._final_alive = False
            clock.return_value = 100.5
            self.extractor.flush()
            self.assertEqual(spawn.call_count, 1)
            self.extractor.add_ext("00:00:02", "new question", t_mono=100.5)
            clock.return_value = 101
            self.extractor.flush()
            self.assertEqual(spawn.call_args.args[0], ["new question"])
            self.assertEqual(spawn.call_count, 2)
            self.extractor._final_alive = False
            clock.return_value = 107
            self.extractor.flush()
            self.assertEqual(spawn.call_count, 2)

    def test_async_call_keeps_only_latest_pending_snapshot(self):
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()
        calls = []
        def answer(window, **kwargs):
            calls.append(window)
            if len(calls) == 1:
                entered.set()
                release.wait(2)
            else:
                finished.set()
            return iter([])
        self.extractor.llm.iter_qa = answer
        self.extractor.set_auto(True)
        try:
            self.extractor.add_ext("00:00:00", "first question")
            self.extractor.flush(force=True)
            self.assertTrue(entered.wait(1))
            self.extractor.add_ext("00:00:01", "second question")
            self.extractor.flush(force=True)
            self.extractor.add_ext("00:00:02", "third question")
            self.extractor.flush(force=True)
            self.assertEqual(len(calls), 1)
            release.set()
            self.assertTrue(finished.wait(1))
            self.assertEqual(len(calls), 2)
            self.assertIn("third question", calls[-1])
        finally:
            release.set()
            self.extractor.close()

    def test_time_window_keeps_all_lines_without_count_limit(self):
        with patch("questions.now_mono", return_value=100):
            for i in range(20):
                self.extractor.add_ext("00:00:00", f"fragment {i}", t_mono=99)
            with self.extractor._lock:
                self.assertEqual(len(self.extractor._recent_window(100)), 20)

    def test_repeated_question_can_retry_after_skip(self):
        self.extractor.set_auto(True)
        self.extractor.llm.iter_qa = Mock(return_value=iter([]))
        with patch.object(self.extractor, "_spawn", side_effect=self.extractor._run_job) as spawn:
            self.extractor.add_ext("00:00:00", "What is Python?")
            self.extractor.flush(force=True)
            self.assertFalse(self.extractor._final_blob)
            self.extractor.llm.iter_qa.return_value = iter([("What is Python?", "Python is a language.")])
            self.extractor.add_ext("00:00:04", "What is Python?")
            self.extractor.flush(force=True)
            self.assertEqual(spawn.call_count, 2)
        self.assertIn("Python is a language.", self.extractor.out_path.read_text())

    def test_disabled_buffers_question_and_enable_answers_it(self):
        runtime = Runtime(Path(self.tmp.name) / ".env")
        runtime.extractor = self.extractor
        with patch.object(self.extractor, "_spawn") as spawn:
            self.extractor.add_ext("00:00:00", "Hello? What is Python?")
            spawn.assert_not_called()
            runtime.set_qa_enabled(True)
            self.assertTrue(self.extractor.auto_qa)
            self.assertTrue(any(c.args[1] == "final" for c in spawn.call_args_list))

    def test_enabled_speech_generates_answer_and_saves_it(self):
        self.extractor.llm.iter_qa = Mock(return_value=iter([
            ("What is Python?", "Python is a programming language."),
        ]))
        def run_final(window, kind, gen):
            if kind == "final":
                self.extractor._run_job(window, kind, gen)
        self.extractor.set_auto(True)
        with patch.object(self.extractor, "_spawn", side_effect=run_final):
            self.extractor.add_ext("00:00:00", "Hello? What is Python?")
            self.extractor.flush(force=True)
        self.assertIn("Python is a programming language.", self.extractor.out_path.read_text())
        self.assertTrue(any(e["type"] == "qa" and e.get("answer") for e in self.bus.snapshot()))
        self.assertIn("Listening for the next question…", self.statuses())

    def test_model_failure_is_visible_and_toggle_can_retry(self):
        self.extractor.llm.iter_qa = Mock(side_effect=RuntimeError("HTTP 401"))
        self.extractor._run_job(["What is Python?"], "final", 0)
        self.assertTrue(any("HTTP 401" in text for text in self.statuses()))
        with patch.object(self.extractor, "_spawn") as spawn:
            self.extractor.add_ext("00:00:00", "What is Python?")
            self.extractor.set_auto(False)
            self.extractor.set_auto(True)
            self.assertTrue(self.extractor.trigger())
            self.assertTrue(spawn.called)

    def test_skip_and_stale_error_status(self):
        self.extractor.llm.iter_qa = Mock(return_value=iter([]))
        self.extractor._run_job(["unclear speech"], "final", 0)
        self.assertTrue(any("No complete question" in text for text in self.statuses()))
        before = self.statuses()
        self.extractor.llm.iter_qa = Mock(side_effect=RuntimeError("cancelled"))
        self.extractor._run_job(["old question"], "final", -1)
        self.assertEqual(before, self.statuses())

    def test_missing_endpoint_cannot_enable_qa(self):
        runtime = Runtime(Path(self.tmp.name) / ".env")
        runtime.extractor = self.extractor
        self.extractor.llm.client.url = ""
        with self.assertRaisesRegex(ValueError, "FUNCTION_URL"):
            runtime.set_qa_enabled(True)
        self.assertFalse(self.extractor.auto_qa)


if __name__ == "__main__":
    unittest.main()
