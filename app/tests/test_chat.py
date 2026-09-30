import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from events import EventBus
from settings import Runtime


class ChatTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.runtime = Runtime(root / ".env")
        self.runtime.extractor = Mock(out_path=root / "session_questions.txt")
        self.client = self.runtime.extractor.llm.client
        self.transcript = root / "session_transcribe.txt"
        self.transcript.write_text("[00:01] [MIC] The deadline is Friday.\n[00:02] [EXT] I will send it.")

    def test_full_transcript_and_followup_use_latest_speech(self):
        self.client.chat.side_effect = ["The deadline is Friday.", "Alex will send it."]
        first = self.runtime.ask_chat("What is the deadline?")
        self.assertEqual(len(first["messages"]), 2)
        prompt = self.client.chat.call_args.args[0]
        reference = json.loads(prompt.split("Reference data (JSON):\n", 1)[1].split("\n\nUser question:", 1)[0])
        self.assertEqual(reference["transcript"], self.transcript.read_text())
        self.transcript.write_text(self.transcript.read_text() + "\n[00:03] [MIC] Alex will send it.")
        second = self.runtime.ask_chat("Who will send it?")
        prompt = self.client.chat.call_args.args[0]
        self.assertIn("Alex will send it.", prompt)
        self.assertIn("What is the deadline?", prompt)
        self.assertEqual(len(second["messages"]), 4)

    def test_invalid_or_missing_transcript_does_not_call_model(self):
        for question in (None, " ", "x" * 8001):
            with self.assertRaises(ValueError):
                self.runtime.ask_chat(question)
        self.transcript.unlink()
        with patch("settings.BUS", EventBus()), self.assertRaisesRegex(ValueError, "No transcript"):
            self.runtime.ask_chat("What happened?")
        self.client.chat.assert_not_called()

    def test_failure_releases_lock_and_preserves_history(self):
        self.client.chat.side_effect = [RuntimeError("Unavailable"), "Friday"]
        with self.assertRaises(RuntimeError):
            self.runtime.ask_chat("When?")
        self.assertEqual(self.runtime.chat_history(), [])
        self.assertEqual(self.runtime.ask_chat("When?")["messages"][-1]["content"], "Friday")

    def test_concurrent_request_is_rejected(self):
        with self.runtime._chat_lock:
            with self.assertRaisesRegex(ValueError, "Wait"):
                self.runtime.ask_chat("When?")
        self.client.chat.assert_not_called()

    @patch.dict(os.environ, {}, clear=False)
    def test_chat_prompt_persists_and_can_reset(self):
        from llm.prompt import default_chat, conversation_chat_prompt
        self.runtime.apply(chat_prompt="Answer in one short sentence.")
        self.assertEqual(self.runtime.chat_prompt_path().read_text().strip(), "Answer in one short sentence.")
        os.environ.pop("CHAT_PROMPT", None)
        restored = Runtime(self.runtime.env_path)
        self.assertEqual(restored.snapshot()["chat_prompt"], "Answer in one short sentence.")
        self.assertIn("Answer in one short sentence.", conversation_chat_prompt("Why?", "Hello", []))
        restored.apply(chat_prompt="")
        self.assertFalse(restored.chat_prompt_path().exists())
        self.assertEqual(restored.snapshot()["chat_prompt"], default_chat())
        with self.assertRaisesRegex(ValueError, "too long"):
            restored.apply(chat_prompt="x" * 8001)


if __name__ == "__main__":
    unittest.main()
