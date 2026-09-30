import socket
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
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


if __name__ == "__main__":
    unittest.main()
