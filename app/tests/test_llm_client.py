import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from llm.client import ChatClient


class ChatClientTests(unittest.TestCase):
    def setUp(self):
        self.client = ChatClient("https://example.test", "test", "nova-pro")

    def test_partial_stream_error_is_not_reported_as_success(self):
        def broken(*args):
            yield "partial answer"
            raise RuntimeError("connection lost")
        self.client._iter_stream = broken
        self.client.chat = Mock()
        stream = self.client.chat_stream("question")
        self.assertEqual(next(stream), "partial answer")
        with self.assertRaisesRegex(RuntimeError, "connection lost"):
            next(stream)
        self.client.chat.assert_not_called()
        self.assertEqual(self.client._cancels, [])

    def test_cancelled_fallback_does_not_yield_late_answer(self):
        self.client._iter_stream = Mock(return_value=iter([]))
        def fallback(*args, **kwargs):
            self.client.abort()
            return "late answer"
        self.client.chat = fallback
        self.assertEqual(list(self.client.chat_stream("question")), [])

    def test_empty_stream_still_falls_back(self):
        self.client._iter_stream = Mock(return_value=iter([]))
        self.client.chat = Mock(return_value="answer")
        self.assertEqual(list(self.client.chat_stream("question")), ["answer"])


if __name__ == "__main__":
    unittest.main()
