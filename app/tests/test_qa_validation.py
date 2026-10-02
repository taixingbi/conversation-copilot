import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai.answer import LlmAnswerer


class QaValidationTests(unittest.TestCase):
    def setUp(self):
        self.answerer = LlmAnswerer(function_url="https://example.test", api_key="test", model="nova-pro")
        self.answerer.client = Mock()

    def test_rejected_question_never_generates_or_displays_answer(self):
        for text in ("What is", "Why does the purple tomorrow compile?", "Hello"):
            with self.subTest(text=text):
                self.answerer.client.chat.return_value = "SKIP"
                self.answerer.client.chat_stream.return_value = iter(["SKIP"])
                self.assertEqual(list(self.answerer.iter_qa([text])), [])
                self.assertEqual(self.answerer.qa_from_ext([text]), ("", ""))


    def test_partial_output_is_not_visible(self):
        self.answerer.client.chat.return_value = "YES"
        consumed = []
        def stream(*args, **kwargs):
            for part in ("Q: What", " is Python?\nA:", " Python is a language."):
                consumed.append(part)
                yield part
        self.answerer.client.chat_stream.side_effect = stream
        result = self.answerer.iter_qa(["What is Python?"])
        self.assertEqual(next(result), ("What is Python?", "Python is a language."))
        self.assertEqual(len(consumed), 3)
        self.assertEqual(list(result), [])

    def test_skip_or_missing_answer_never_creates_card(self):
        self.answerer.client.chat.return_value = "YES"
        for parts in (["Q: What is Python?"], ["Q: What is Python?\nA: Guess", "\nSKIP"]):
            self.answerer.client.chat_stream.return_value = iter(parts)
            self.assertEqual(list(self.answerer.iter_qa(["What is Python?"])), [])

    def test_agent_duplicate_check_before_display(self):
        self.answerer.answered_questions = ["What is Python?"]
        self.answerer.client.chat.return_value = "DUPLICATE"
        self.answerer.client.chat_stream.return_value = iter(["Q: What is the Python programming language?\nA: Python is a language."])
        self.assertEqual(list(self.answerer.iter_qa(["computer language Python"])), [])
        prompt = self.answerer.client.chat.call_args.args[0]
        self.assertIn("What is Python?", prompt)
        self.assertIn("What is the Python programming language?", prompt)

    def test_agent_accepts_specific_subject_after_general_answer(self):
        self.answerer.answered_questions = ["What is a programming language?"]
        self.answerer.client.chat.return_value = "NEW\n\nThis asks about a specific language."
        self.answerer.client.chat_stream.return_value = iter(["Q: What is Python?\nA: Python is a language."])
        self.assertEqual(list(self.answerer.iter_qa(["computer language Python"])), [("What is Python?", "Python is a language.")])

    def test_uncertain_duplicate_result_does_not_display_answer(self):
        self.answerer.answered_questions = ["What is Python?"]
        self.answerer.client.chat.return_value = "unclear"
        self.answerer.client.chat_stream.return_value = iter(["Q: What is Python?\nA: Python is a language."])
        with self.assertRaisesRegex(RuntimeError, "Could not validate"):
            list(self.answerer.iter_qa(["Python?"]))


if __name__ == "__main__":
    unittest.main()
