import ast
import sys
import tempfile
import unittest
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))

from audio.transcripts import TranscriptStore
from events import EventBus


class ComponentBoundaryTests(unittest.TestCase):
    def test_ai_and_audio_do_not_import_each_other_or_ui(self):
        for component, forbidden in (("ai", {"audio", "ui", "settings", "runtime", "application", "main"}),
                                     ("audio", {"ai", "ui", "settings", "runtime", "application", "main"})):
            for path in (APP / component).glob("*.py"):
                tree = ast.parse(path.read_text())
                imports = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imports.extend(alias.name for alias in node.names)
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        imports.append(node.module)
                with self.subTest(path=path.name):
                    self.assertFalse({name.split(".")[0] for name in imports} & forbidden)

    def test_ai_reader_excludes_unconfirmed_speech(self):
        bus = EventBus()
        bus.publish("transcript", phase="partial", text="Bed rock", utterance_id="EXT:1")
        reader = TranscriptStore(bus=bus)
        self.assertEqual(reader.text(), "")
        bus.publish("transcript", phase="final", text="Bedrock", utterance_id="EXT:1", label="EXT")
        self.assertIn("Bedrock", reader.text())
        self.assertNotIn("Bed rock", reader.text())

    def test_persisted_session_is_authoritative_over_short_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "transcribe.txt"
            path.write_text("[10:00:00] [MIC] Earlier speech.\n[10:01:00] [EXT] Later speech.")
            bus = EventBus(keep=1)
            bus.publish("transcript", phase="final", text="Later speech.")
            self.assertEqual(TranscriptStore(path, bus=bus).text(), path.read_text())


if __name__ == "__main__":
    unittest.main()
