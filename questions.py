from __future__ import annotations

import re
import sys
import threading
import time
from collections import deque
from difflib import SequenceMatcher
from pathlib import Path

from llm.answer import LlmAnswerer, short_answer
from noise import is_skip_ext

DIM = "\033[2m"
CYAN = "\033[36m"
YELLOW = "\033[1;33m"
GREEN = "\033[1;32m"
RESET = "\033[0m"


def paint(text: str, style: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"{style}{text}{RESET}"

ASKS = re.compile(
    r"\?|^(what|why|how|when|where|who|which|tell me|can you|could you|do you|did you|"
    r"let'?s|explain|compare|walk me|describe|difference)\b",
    re.I,
)
INCOMPLETE = re.compile(
    r"^(what|why|how|when|where|who|which|tell me|can you|could you)"
    r"(\s+is|\s+are|\s+was|\s+were|\s+about)?\s*\??$",
    re.I,
)
STEM = re.compile(
    r"^(what|why|how|when|where|who|which)(\s+is|\s+are|\s+was|\s+were)?\s*\??$",
    re.I,
)
STOP_TAIL = {"the", "a", "an", "of", "and", "or", "to", "for", "in", "on"}
EXT_WINDOW = 5
OPEN_TAIL = STOP_TAIL | {"different", "difference", "between", "versus", "vs", "compare"}


def questions_path_for(transcribe_path: str | Path) -> Path:
    p = Path(transcribe_path)
    name = p.name
    if name.endswith("_transcribe.txt"):
        name = name[: -len("_transcribe.txt")] + "_questions.txt"
    else:
        name = p.stem + "_questions.txt"
    return p.with_name(name)


def is_sure_question(text: str) -> bool:
    t = re.sub(r"\s+", " ", text.strip())
    if not t or is_skip_ext(t) or INCOMPLETE.match(t) or STEM.match(t):
        return False
    words = re.findall(r"[A-Za-z0-9]+", t)
    if len(words) < 3 or words[-1].lower() in STOP_TAIL:
        return False
    return bool(ASKS.search(t) or t.endswith("?"))


def _is_open_fragment(text: str) -> bool:
    """True if this line still needs more EXT before a question is complete."""
    t = re.sub(r"[.]+$", "", text.strip())
    if not t or STEM.match(t) or INCOMPLETE.match(t):
        return True
    words = re.findall(r"[A-Za-z0-9]+", t)
    if not words:
        return True
    return words[-1].lower() in OPEN_TAIL


def ext_window(ext_lines: list[str]) -> list[str]:
    return [b.strip() for b in ext_lines if b.strip()][-EXT_WINDOW:]


def ready_for_llm(ext_lines: list[str], *, allow_partial: bool = False) -> bool:
    """True when EXT lines look like a question. Does not build the question text."""
    window = ext_window(ext_lines)
    if not window:
        return False
    blob = re.sub(r"\s+", " ", " ".join(window)).strip()
    asking = bool(ASKS.search(blob) or "?" in blob)
    if not asking:
        return False
    if _is_open_fragment(window[-1]) and not allow_partial:
        return False
    if len(window) == 1:
        return is_sure_question(window[0])
    return True


class QuestionExtractor:
    def __init__(
        self,
        *,
        function_url: str,
        api_key: str,
        model: str,
        out_path: Path,
        idle_sec: float = 0.45,
        max_interval_sec: float = 4.0,
    ):
        self.llm = LlmAnswerer(function_url=function_url, api_key=api_key, model=model)
        self.out_path = Path(out_path)
        self.idle_sec = idle_sec
        self.max_interval_sec = max_interval_sec
        self._pending: list[str] = []
        self._history: deque = deque(maxlen=EXT_WINDOW)
        self._known: list[str] = []
        self._lock = threading.Lock()
        self._last_ext = 0.0
        self._last_call = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.touch(exist_ok=True)
        if not self._thread.is_alive():
            self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self.flush(force=True)
        if self._thread.is_alive():
            self._thread.join(timeout=2)

    def add_ext(self, ts: str, text: str) -> None:
        body = text.strip()
        if is_skip_ext(body):
            return
        with self._lock:
            self._pending.append(f"[{ts}] {body}")
            self._history.append(body)
            self._last_ext = time.time()
            window = list(self._history)
        if ready_for_llm(window):
            try:
                self.flush(force=True)
            except Exception as exc:
                print(f"Question flush failed: {exc}", file=sys.stderr, flush=True)

    def _already(self, question: str) -> bool:
        return any(SequenceMatcher(None, question.lower(), k.lower()).ratio() >= 0.82 for k in self._known)

    def _write_qa(self, question: str, answer: str) -> None:
        with self._lock:
            if self._already(question):
                return
            self._known.append(question)
        ts = time.strftime("%H:%M:%S")
        answer = short_answer(answer)
        block = f"[{ts}] Q: {question}\nA: {answer}\n\n"
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with self.out_path.open("a", encoding="utf-8") as f:
            f.write(block)
            f.flush()
        print(paint(f"Q  [{ts}] {question}", YELLOW), flush=True)
        preview = answer.replace("\n", " ")
        if len(preview) > 180:
            preview = preview[:177] + "..."
        print(paint(f"A  {preview}", GREEN), flush=True)

    def _consume_pending(self, sent: list[str]) -> None:
        remaining = list(self._pending)
        for item in sent:
            try:
                remaining.remove(item)
            except ValueError:
                pass
        self._pending = remaining

    def flush(self, force: bool = False) -> None:
        with self._lock:
            pending = list(self._pending)
            last_ext = self._last_ext
            last_call = self._last_call
        if not pending:
            return
        idle = time.time() - last_ext if last_ext else 0
        waited = time.time() - last_call if last_call else 999
        words = sum(len(p.split()) for p in pending)
        ready = force or (idle >= self.idle_sec and words >= 2) or (
            waited >= self.max_interval_sec and idle >= self.idle_sec
        )
        if not ready:
            return
        with self._lock:
            window = list(self._history)
        window = ext_window(window)
        if not ready_for_llm(window, allow_partial=force or idle >= 6):
            if not force and idle < 6:
                return
            with self._lock:
                self._consume_pending(pending)
                self._last_call = time.time()
            return
        with self._lock:
            self._consume_pending(pending)
            self._last_call = time.time()
        if not self.llm.enabled:
            print("   (no LLM — set FUNCTION_URL to build questions)", flush=True)
            return
        try:
            question, answer = self.llm.qa_from_ext(window)
        except Exception as exc:
            print(f"LLM question failed: {exc}", file=sys.stderr, flush=True)
            return
        if question and answer:
            self._write_qa(question, answer)
        else:
            shown = " | ".join(window[-3:])
            print(paint(f"   (LLM skip) {shown}", DIM), flush=True)

    def _loop(self) -> None:
        while not self._stop.wait(0.12):
            try:
                self.flush()
            except Exception as exc:
                print(f"Question flush failed: {exc}", file=sys.stderr, flush=True)
