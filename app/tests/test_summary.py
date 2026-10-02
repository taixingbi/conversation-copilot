import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audio.transcripts import TranscriptStore
from events import EventBus
from runtime import Runtime


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
            runtime.transcripts = TranscriptStore(root / "session_transcribe.txt", bus=EventBus())
            runtime.extractor = Mock(out_path=qpath)
            source = runtime.extractor.llm.client
            source.url = "https://example.test/v1/chat/completions"
            source.api_key = "test"
            source.model = "nova-pro"
            client = Mock()
            client.chat_stream.return_value = iter(["Conversation summary."])
            bus = EventBus(keep=2)
            bus.publish("transcript", text="Only the recent line", label="EXT")
            with patch("ai.summary.BUS", bus), patch("ai.client.ChatClient", return_value=client) as factory:
                runtime._run_summary(0)
            factory.assert_called_once_with("https://example.test", "test", "nova-pro")
            source.chat_stream.assert_not_called()
            prompt = client.chat_stream.call_args.args[0]
            self.assertIn(transcript, prompt)
            self.assertNotIn("Generated answer that nobody said", prompt)
            self.assertIn("whole conversation", prompt)
            self.assertEqual(bus.snapshot()[-1]["text"], "Conversation summary.")

    def test_summary_entrypoint_accepts_real_extractor_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = Runtime(Path(tmp) / ".env")
            runtime.extractor = SimpleNamespace(llm=SimpleNamespace(client=SimpleNamespace(enabled=True)))
            with patch("ai.summary.threading.Thread") as thread, patch("ai.summary.BUS", EventBus()):
                self.assertEqual(runtime.start_summary(), {"ok": True})
                thread.return_value.start.assert_called_once()

    def test_deleted_summary_aborts_client_and_never_republishes(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = Runtime(Path(tmp) / ".env")
            source = Mock(url="https://example.test/v1/chat/completions", api_key="test", model="test")
            runtime.extractor = Mock()
            runtime.extractor.llm.client = source
            runtime.transcripts = Mock()
            runtime.transcripts.text.return_value = "Some finalized conversation."
            client = Mock()
            def stream(*args, **kwargs):
                yield "First"
                runtime.clear_summary()
                yield "Late"
            client.chat_stream.side_effect = stream
            bus = EventBus()
            with patch("ai.summary.BUS", bus), patch("ai.client.ChatClient", return_value=client):
                runtime._run_summary(0)
            client.abort.assert_called_once()
            self.assertEqual(bus.snapshot()[-1]["type"], "summary_gone")

    def test_generated_qa_alone_is_not_a_transcript(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            qpath = root / "session_questions.txt"
            qpath.write_text("Q: Question\nA: Generated answer")
            runtime = Runtime(root / ".env")
            runtime.transcripts = TranscriptStore(root / "session_transcribe.txt", bus=EventBus())
            runtime.extractor = Mock(out_path=qpath)
            bus = EventBus()
            with patch("ai.summary.BUS", bus):
                runtime._run_summary(0)
            runtime.extractor.llm.client.chat_stream.assert_not_called()
            self.assertEqual(bus.snapshot()[-1]["text"], "Nothing to summarize yet.")


if __name__ == "__main__":
    unittest.main()
