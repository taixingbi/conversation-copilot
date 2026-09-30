import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from events import EventBus
from settings import Runtime


class SummaryTests(unittest.TestCase):
    @patch.dict(os.environ, {"SUMMARY_PROMPT": ""})
    def test_full_raw_transcript_is_sent_without_generated_answers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            transcript = "\n".join(
                f"[00:00:{i}] [MIC] {'Yes.' if i % 2 else 'What is the plan?'}"
                for i in range(120)
            )
            (root / "session_transcribe.txt").write_text(transcript)
            qpath = root / "session_questions.txt"
            qpath.write_text("A: Generated answer that nobody said.")
            runtime = Runtime(root / ".env")
            runtime.extractor = Mock(out_path=qpath)
            source = runtime.extractor.llm.client
            source.url = "https://example.test/v1/chat/completions"
            source.api_key = "test"
            source.model = "nova-pro"
            client = Mock()
            client.chat_stream.return_value = iter(["Conversation summary."])
            bus = EventBus(keep=2)
            bus.publish("transcript", text="Only the recent line", label="EXT")
            with patch("settings.BUS", bus), patch("llm.client.ChatClient", return_value=client) as factory:
                runtime._run_summary(0)
            factory.assert_called_once_with("https://example.test", "test", "nova-pro")
            source.chat_stream.assert_not_called()
            prompt = client.chat_stream.call_args.args[0]
            self.assertIn(transcript, prompt)
            self.assertNotIn("Generated answer that nobody said", prompt)
            self.assertIn("whole conversation", prompt)
            self.assertEqual(bus.snapshot()[-1]["text"], "Conversation summary.")

    def test_generated_qa_alone_is_not_a_transcript(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            qpath = root / "session_questions.txt"
            qpath.write_text("Q: Question\nA: Generated answer")
            runtime = Runtime(root / ".env")
            runtime.extractor = Mock(out_path=qpath)
            bus = EventBus()
            with patch("settings.BUS", bus):
                runtime._run_summary(0)
            runtime.extractor.llm.client.chat_stream.assert_not_called()
            self.assertEqual(bus.snapshot()[-1]["text"], "Nothing to summarize yet.")


if __name__ == "__main__":
    unittest.main()
