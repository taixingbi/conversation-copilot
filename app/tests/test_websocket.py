import socket
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ui.server import _ws_read


class WebSocketTests(unittest.TestCase):
    def test_idle_timeout_is_not_disconnect(self):
        connection = Mock()
        connection.recv.side_effect = socket.timeout("timed out")
        with self.assertRaises(socket.timeout):
            _ws_read(connection)

    def test_closed_socket_is_disconnect(self):
        connection = Mock()
        connection.recv.return_value = b""
        self.assertIsNone(_ws_read(connection))

    def test_partial_mask_survives_timeout(self):
        connection = Mock()
        connection.recv.side_effect = [b"\x81\x81\x01", socket.timeout(), b"\x02\x03\x04\x40"]
        pending = bytearray()
        with self.assertRaises(socket.timeout):
            _ws_read(connection, pending)
        self.assertEqual(_ws_read(connection, pending), b"A")
        self.assertEqual(pending, bytearray())

    def test_back_to_back_frames_are_retained(self):
        connection = Mock()
        connection.recv.return_value = b"\x81\x01A\x81\x01B"
        pending = bytearray()
        self.assertEqual(_ws_read(connection, pending), b"A")
        self.assertEqual(_ws_read(connection, pending), b"B")
        connection.recv.assert_called_once()


if __name__ == "__main__":
    unittest.main()
