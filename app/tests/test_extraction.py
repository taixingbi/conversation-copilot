import json
import queue
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from events import EventBus
from ai.answer import LlmAnswerer
from ai.extract import ExtractedQuestion, parse_extraction
from ai.questions import QuestionExtractor


class ExtractionContractTests(unittest.TestCase):
    def test_wait_and_invalid_output(self):
        self.assertIsNone(parse_extraction("WAIT"))
        self.assertIsNone(parse_extraction('{"action":"WAIT"}'))
        for raw in ('QUESTION: Why?', '[]', '{"action":"NEW","question":""}',
                    '{"action":"UPDATE","question":"Why?","context":null}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_extraction(raw)

    def test_one_fast_model_call_contains_state_and_speaker_contract(self):
        llm = LlmAnswerer(function_url="https://example.test", api_key="test", model="large", fast_model="small")
        llm.client = Mock()
        llm.client.chat.return_value = json.dumps({"action": "UPDATE", "question": "How did you determine the safe concurrency?"})
        result = llm.extract_latest(["[MIC-3] how did you determine the safe concurrency"],
                                    current="How did you determine concurrency?", previous=["What is Bedrock?"],
                                    speakers=("MIC-3",))
        self.assertEqual(result.action, "UPDATE")
        llm.client.chat.assert_called_once()
        self.assertEqual(llm.client.chat.call_args.kwargs["model"], "small")
        prompt = llm.client.chat.call_args.args[0]
        for text in ("MIC-3", "safe concurrency", "What is Bedrock?", "WAIT", "walk me through"):
            self.assertIn(text, prompt)
        llm.client.chat_stream.assert_not_called()


class LiveExtractionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.bus = EventBus()
        patcher = patch("ai.questions.BUS", self.bus)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.ex = QuestionExtractor(function_url="https://example.test", api_key="test", model="small",
                                    out_path=Path(self.tmp.name) / "ai.questions.txt")
        self.ex.set_auto(True)
        self.ex.llm.extract_latest = Mock(return_value=None)
        patcher = patch.object(self.ex, "_spawn", side_effect=self.ex._run_job)
        patcher.start()
        self.addCleanup(patcher.stop)

    def cards(self):
        return [e for e in self.bus.snapshot() if e["type"] == "qa"]

    def test_wait_does_not_repeat_inference_without_new_speech(self):
        with patch("ai.questions.now_mono", return_value=100) as clock:
            self.ex.add_ext("", "[EXT] how did you", t_mono=100)
            self.ex.flush()
            clock.return_value = 102
            self.ex.flush()
            self.ex.add_ext("", "[EXT] how did you", t_mono=102)
            self.ex.flush()
            self.ex.llm.extract_latest.assert_called_once()
            self.assertEqual(self.cards(), [])
            clock.return_value = 104
            self.ex.add_ext("", "[EXT] determine the limit", t_mono=104)
            self.ex.flush()
            self.assertEqual(self.ex.llm.extract_latest.call_count, 2)

    def test_mic_is_context_only_and_ext_can_trigger(self):
        self.ex.add_ext("", "[MIC-1] what is the limit")
        self.ex.flush(force=True)
        self.ex.llm.extract_latest.assert_not_called()
        self.ex.add_ext("", "[EXT-2] how did you measure it")
        self.ex.flush(force=True)
        lines = self.ex.llm.extract_latest.call_args.args[0]
        self.assertEqual(len(lines), 2)
        self.assertEqual(self.ex.llm.extract_latest.call_args.kwargs["speakers"], ("EXT",))

    def test_explicit_microphone_speaker(self):
        self.ex.target_speakers = ("MIC-3",)
        self.ex.add_ext("", "[MIC-1] what about latency")
        self.ex.flush(force=True)
        self.ex.llm.extract_latest.assert_not_called()
        self.ex.add_ext("", "[MIC-3] walk me through capacity testing")
        self.ex.flush(force=True)
        self.ex.llm.extract_latest.assert_called_once()
        self.assertFalse(self.ex._eligible("MIC-30"))

    def test_cumulative_asr_merges_only_same_speaker(self):
        self.ex.add_ext("", "[EXT-1] how did you determine")
        self.ex.add_ext("", "[EXT-1] how did you determine concurrency")
        self.ex.add_ext("", "[EXT-2] how did you determine concurrency")
        self.assertEqual(len(self.ex._history), 2)
        self.assertIn("concurrency", self.ex._history[0][1])

    def test_update_preserves_id_and_replay_contains_one_card(self):
        first = ExtractedQuestion("How did you determine concurrency?", "Capacity testing")
        second = ExtractedQuestion("How did you determine the safe concurrency?", "Capacity testing", "UPDATE")
        self.ex._on_extracted(first, "final")
        old_id = self.cards()[0]["question_id"]
        self.ex._on_extracted(second, "final")
        cards = self.cards()
        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0]["question_id"], old_id)
        self.assertEqual(cards[0]["replaces_question"], first.question)
        self.assertEqual(len(self.ex.previous_questions), 0)
        replay = self.bus.subscribe()
        replay_cards = []
        while True:
            try:
                event = replay.get_nowait()
            except queue.Empty:
                break
            if event["type"] == "qa":
                replay_cards.append(event)
        self.assertEqual(replay_cards, cards)
        records = [json.loads(line) for line in self.ex.out_path.read_text().splitlines()]
        self.assertEqual([r["action"] for r in records], ["NEW", "UPDATE"])
        self.assertEqual(records[0]["id"], records[1]["id"])

    def test_new_followup_archives_current_and_duplicates_do_not_publish(self):
        self.ex._on_extracted(ExtractedQuestion("How did you measure capacity?"), "final")
        first = self.ex.current_question
        self.ex._on_extracted(ExtractedQuestion("How did you measure capacity!"), "final")
        self.assertEqual(len(self.cards()), 1)
        self.ex._on_extracted(ExtractedQuestion("And what about latency?"), "final")
        self.assertNotEqual(self.ex.current_question["id"], first["id"])
        self.assertEqual(list(self.ex.previous_questions), [first])
        self.assertEqual(len(self.cards()), 2)
        self.ex.forget_all()
        self.assertIsNone(self.ex.current_question)
        self.assertEqual(list(self.ex.previous_questions), [])

    def test_question_without_context_is_displayed(self):
        self.ex.llm.extract_latest.return_value = ExtractedQuestion("Why?")
        self.ex.add_ext("", "[EXT] why")
        self.ex.flush(force=True)
        self.assertEqual(self.cards()[0]["question"], "Why?")
        self.assertEqual(self.cards()[0]["context"], "")

    def test_wait_preserves_current_question(self):
        self.ex._on_extracted(ExtractedQuestion("How did you measure capacity?"), "final")
        before = self.ex.current_question
        self.ex.add_ext("", "[EXT] and how did you")
        self.ex.flush(force=True)
        self.assertEqual(self.ex.current_question, before)
        self.assertEqual(len(self.cards()), 1)


if __name__ == "__main__":
    unittest.main()
