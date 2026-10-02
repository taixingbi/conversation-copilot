import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai.client import ChatClient


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

    def test_abort_closes_sync_response_and_rejects_late_content(self):
        resp = Mock()
        def read():
            self.client.abort()
            return b'{"choices":[{"message":{"content":"late"}}]}'
        resp.read.side_effect = read
        self.client._request = Mock(return_value=resp)
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            self.client.chat("question")
        resp.close.assert_called()
        self.assertEqual(self.client._cancels, [])
        self.assertEqual(self.client._resps, [])

    def test_fork_has_independent_cancellation(self):
        other = self.client.fork()
        resp = Mock()
        other._track(resp)
        self.client.abort()
        resp.close.assert_not_called()
        other.abort()
        resp.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
