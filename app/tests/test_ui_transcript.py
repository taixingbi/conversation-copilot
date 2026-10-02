import shutil
import subprocess
import unittest
from pathlib import Path


@unittest.skipUnless(shutil.which("node"), "Node is required for UI event tests")
class UiTranscriptTests(unittest.TestCase):
    def test_same_text_final_and_distinct_utterances_update_ui(self):
        html = (Path(__file__).resolve().parents[1] / "ui/overlay.html").read_text()
        handler = html[html.index("    function onEvent(ev)"):html.index("    function connect()")]
        setup = '''
const seenEv = new Set();
let lastTx = "";
const txLast = {};
const esc = String;
const transcript = {children: [], scrollHeight: 0, scrollTop: 0, clientHeight: 0,
  appendChild(row) { this.children.push(row); }};
const document = {createElement() { return {dataset: {}, style: {}, remove() {}}; }};
'''
        exercise = '''
const ev = {type: "transcript", t: 1, ts: "10:00:00", label: "MIC", text: "Bedrock", utterance_id: "MIC:1"};
onEvent({...ev, phase: "partial"});
onEvent({...ev, phase: "final"});
if (transcript.children.length !== 1 || transcript.children[0].style.opacity !== "1") throw Error("final lost");
onEvent({...ev, phase: "final", utterance_id: "MIC:2"});
if (transcript.children.length !== 2) throw Error("distinct utterance lost");
'''
        result = subprocess.run(["node", "-e", setup + handler + exercise], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
