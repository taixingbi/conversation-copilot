import signal
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, call

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from main import close_overlay


class ShutdownTests(unittest.TestCase):
    @patch("main.os.killpg")
    @patch("main.os.name", "posix")
    def test_terminates_group_even_if_launcher_exited(self, killpg):
        proc = Mock(pid=12345)
        proc.poll.return_value = 0
        close_overlay(proc)
        killpg.assert_called_once_with(12345, signal.SIGTERM)
        proc.wait.assert_called_once_with(timeout=2)

    @patch("main.os.killpg")
    @patch("main.os.name", "posix")
    def test_forces_exit_after_timeout(self, killpg):
        proc = Mock(pid=12345)
        proc.wait.side_effect = [subprocess.TimeoutExpired("overlay", 2), 0]
        close_overlay(proc)
        self.assertEqual(killpg.call_args_list, [
            call(12345, signal.SIGTERM), call(12345, signal.SIGKILL),
        ])

    @patch("main.os.killpg", side_effect=ProcessLookupError)
    @patch("main.os.name", "posix")
    def test_already_closed_is_safe(self, killpg):
        close_overlay(Mock(pid=12345))
        close_overlay(None)


if __name__ == "__main__":
    unittest.main()
