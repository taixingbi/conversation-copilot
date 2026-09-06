from __future__ import annotations

import re
import sys
import threading
import time
from collections import deque
from difflib import SequenceMatcher
from pathlib import Path

from llm.answer import LlmAnswerer, short_answer
from metrics import LatencyTracker, now_mono
from noise import is_skip_ext

DIM = "\033[2m"
CYAN = "\033[36m"
YELLOW = "\033[1;33m"
GREEN = "\033[1;32m"
RESET = "\033[0m"

io_lock = threading.Lock()
_stream_open = False


def paint(text: str, style: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"{style}{text}{RESET}"


def safe_print(text: str = "", style: str = "", *, end: str = "\n") -> None:
    """Print without tearing a live answer line when transcript also writes."""
    global _stream_open
    with io_lock:
        if _stream_open and end == "\n":
            print(flush=True)
            _stream_open = False
        shown = paint(text, style) if style else text
        print(shown, end=end, flush=True)
        if end != "\n":
            _stream_open = True


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
        return is_sure_question(window[0]) if not allow_partial else bool(ASKS.search(window[0]) or "?" in window[0])
    return True


def _norm_blob(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def is_extension(prev: str, new: str) -> bool:
    a, b = _norm_blob(prev), _norm_blob(new)
    if not a or not b or a == b:
        return False
    return b.startswith(a) or a in b


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
        fast_model: str = "",
        metrics: LatencyTracker | None = None,
    ):
        self.llm = LlmAnswerer(
            function_url=function_url, api_key=api_key, model=model, fast_model=fast_model
        )
        self.out_path = Path(out_path)
        self.idle_sec = idle_sec
        self.max_interval_sec = max_interval_sec
        self.metrics = metrics or LatencyTracker()
        self._pending: list[str] = []
        self._history: deque = deque(maxlen=EXT_WINDOW)
        self._known: deque = deque(maxlen=40)
        self._lock = threading.Lock()
        self._last_ext = 0.0
        self._last_call = 0.0
        self._t_last_ext = 0.0
        self._t_ext_end = 0.0
        self._t_shown = 0.0
        self._last_stt_ms = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._fast_gen = 0
        self._final_gen = 0
        self._fast_alive = False
        self._final_alive = False
        self._draft_blob = ""
        self._final_blob = ""
        self._shown_q = ""
        self._shown_a = ""
        self._shown_kind = ""
        self._wrote = False
        self._q_printed = False
        self._logged_ext = False

    def start(self) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.touch(exist_ok=True)
        if not self._thread.is_alive():
            self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self.flush(force=True)
        deadline = time.time() + 3
        while (self._fast_alive or self._final_alive) and time.time() < deadline:
            time.sleep(0.05)
        if self._shown_q and self._shown_a and not self._wrote:
            self._write_qa(self._shown_q, self._shown_a, replace=False)
        if self._thread.is_alive():
            self._thread.join(timeout=2)
        summary = self.metrics.summary()
        if summary:
            safe_print(f"lat  {summary}", DIM)

    def add_ext(self, ts: str, text: str, *, t_mono: float | None = None, stt_ms: float = 0.0) -> None:
        body = text.strip()
        if is_skip_ext(body):
            return
        stamp = t_mono if t_mono is not None else now_mono()
        with self._lock:
            if self._history and is_extension(self._history[-1], body):
                self._history[-1] = body
                if self._pending:
                    self._pending[-1] = f"[{ts}] {body}"
            else:
                self._pending.append(f"[{ts}] {body}")
                self._history.append(body)
            self._last_ext = time.time()
            self._t_last_ext = stamp
            if stt_ms:
                self._last_stt_ms = stt_ms
            window = list(self._history)
        if ready_for_llm(window):
            try:
                self.flush(force=True)
            except Exception as exc:
                safe_print(f"Question flush failed: {exc}")
        elif ready_for_llm(window, allow_partial=True):
            self._kick("draft", window)

    def _already(self, question: str) -> bool:
        return any(SequenceMatcher(None, question.lower(), k.lower()).ratio() >= 0.82 for k in self._known)

    def _write_qa(self, question: str, answer: str, *, replace: bool) -> None:
        with self._lock:
            if not replace and self._already(question):
                return
            if not self._already(question):
                self._known.append(question)
        ts = time.strftime("%H:%M:%S")
        block = f"[{ts}] Q: {question}\nA: {answer}\n\n"
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with self.out_path.open("a", encoding="utf-8") as f:
            f.write(block)
            f.flush()
        self._wrote = True

    def _consume_pending(self, sent: list[str]) -> None:
        remaining = list(self._pending)
        for item in sent:
            try:
                remaining.remove(item)
            except ValueError:
                pass
        self._pending = remaining

    def _blob(self, window: list[str]) -> str:
        return _norm_blob(" ".join(window))

    def _kick(self, kind: str, window: list[str]) -> None:
        if not self.llm.enabled:
            return
        blob = self._blob(window)
        if not blob:
            return
        if kind == "draft":
            related = bool(
                self._draft_blob
                and (blob == self._draft_blob or is_extension(self._draft_blob, blob))
            )
            if related and (self._fast_alive or self._shown_kind in {"draft", "fast", "final"}):
                return
            self._fast_gen += 1
            self._fast_alive = True
            self._draft_blob = blob
            self._spawn(window, "draft", self._fast_gen)
            return
        if blob == self._final_blob:
            return
        self._t_ext_end = self._t_last_ext or now_mono()
        if self._t_shown:
            self._record_ext_latency()
        self._final_gen += 1
        self._final_alive = True
        self._final_blob = blob
        if not self._shown_a and not self._fast_alive:
            self._fast_gen += 1
            self._fast_alive = True
            self._draft_blob = blob
            self._spawn(window, "fast", self._fast_gen)
        self._spawn(window, "final", self._final_gen)

    def _spawn(self, window: list[str], kind: str, gen: int) -> None:
        threading.Thread(
            target=self._run_job,
            args=(list(window), kind, gen),
            daemon=True,
            name=f"llm-{kind}",
        ).start()

    def _stale(self, kind: str, gen: int) -> bool:
        if kind in {"draft", "fast"}:
            return gen != self._fast_gen
        return gen != self._final_gen

    def _run_job(self, window: list[str], kind: str, gen: int) -> None:
        t0 = time.perf_counter()
        first = True
        question = answer = ""
        try:
            for question, answer in self.llm.iter_qa(window, kind=kind):
                if self._stale(kind, gen):
                    return
                if first and (question or answer):
                    self.metrics.observe("llm_ttft_ms", (time.perf_counter() - t0) * 1000)
                    first = False
                if question or answer:
                    self._on_partial(question, answer, kind)
                    if answer:
                        self._mark_shown(kind)
            if self._stale(kind, gen):
                return
            self.metrics.observe("llm_total_ms", (time.perf_counter() - t0) * 1000)
            if question and answer:
                self._on_done(question, answer, kind)
            elif kind == "final" and self._shown_q and self._shown_a and not self._wrote:
                self._write_qa(self._shown_q, self._shown_a, replace=False)
        except Exception as exc:
            safe_print(f"LLM {kind} failed: {exc}")
        finally:
            if kind in {"draft", "fast"}:
                self._fast_alive = False
            else:
                self._final_alive = False

    def _record_ext_latency(self) -> None:
        if self._logged_ext or not self._t_shown or not self._t_ext_end:
            return
        self.metrics.observe("ext_to_answer_ms", (self._t_shown - self._t_ext_end) * 1000)
        self._logged_ext = True

    def _mark_shown(self, kind: str) -> None:
        if not self._t_shown:
            self._t_shown = now_mono()
            if kind in {"draft", "fast"} and self._t_last_ext:
                self.metrics.observe("ext_to_draft_ms", (self._t_shown - self._t_last_ext) * 1000)
        self._record_ext_latency()

    def _on_partial(self, question: str, answer: str, kind: str) -> None:
        if kind in {"draft", "fast"} and self._shown_kind == "final":
            return
        restart = kind == "final" and self._shown_kind in {"draft", "fast"}
        self._render(question, answer, kind, restart=restart)

    def _on_done(self, question: str, answer: str, kind: str) -> None:
        global _stream_open
        max_sent = 1 if kind in {"draft", "fast"} else 4
        answer = short_answer(answer, max_sentences=max_sent)
        if not question or not answer:
            return
        if kind in {"draft", "fast"} and self._shown_kind == "final":
            return
        preview = answer.replace("\n", " ")
        if len(preview) > 180:
            preview = preview[:177] + "..."
        same = self._shown_q == question and self._shown_a == preview and self._shown_kind == kind
        if same:
            with io_lock:
                if _stream_open:
                    print(flush=True)
                    _stream_open = False
        else:
            self._render(question, answer, kind, restart=True, done=True)
        self._mark_shown(kind)
        if kind == "final":
            self._fast_gen += 1
            self._write_qa(question, answer, replace=self._wrote)
            self._print_lat()
            self._reset_turn()

    def _reset_turn(self) -> None:
        self._t_shown = 0.0
        self._t_ext_end = 0.0
        self._q_printed = False
        self._shown_q = ""
        self._shown_a = ""
        self._shown_kind = ""
        self._wrote = False
        self._logged_ext = False

    def _print_lat(self) -> None:
        extra = f"  stt {self._last_stt_ms:.0f}ms" if self._last_stt_ms else ""
        ttft = self.metrics.values("llm_ttft_ms")
        if ttft:
            extra += f"  ttft {ttft[-1]:.0f}ms"
        safe_print(f"lat  {self.metrics.line('ext_to_answer_ms', label='ext→answer')}{extra}", DIM)

    def _render(self, question: str, answer: str, kind: str, *, restart: bool, done: bool = False) -> None:
        global _stream_open
        preview = answer.replace("\n", " ")
        if len(preview) > 180:
            preview = preview[:177] + "..."
        tag = "A~" if kind in {"draft", "fast"} else "A "
        style = DIM if kind in {"draft", "fast"} else GREEN
        extends = bool(self._shown_a and preview.startswith(self._shown_a))
        restart = restart or not extends
        with io_lock:
            if question and (not self._q_printed or question != self._shown_q):
                if _stream_open:
                    print(flush=True)
                    _stream_open = False
                ts = time.strftime("%H:%M:%S")
                print(paint(f"Q  [{ts}] {question}", YELLOW), flush=True)
                self._q_printed = True
                self._shown_q = question
            if not preview:
                self._shown_kind = kind
                return
            if restart and _stream_open:
                print(flush=True)
                _stream_open = False
            if done or restart or not _stream_open:
                print(paint(f"{tag} {preview}", style), end="" if not done else "\n", flush=True)
                _stream_open = not done
            else:
                print(paint(preview[len(self._shown_a) :], style), end="", flush=True)
                _stream_open = True
            if done:
                _stream_open = False
        self._shown_a = preview
        self._shown_kind = kind

    def flush(self, force: bool = False) -> None:
        with self._lock:
            pending = list(self._pending)
            last_ext = self._last_ext
            last_call = self._last_call
            window = list(self._history)
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
        window = ext_window(window)
        allow_partial = force or idle >= 6
        if ready_for_llm(window, allow_partial=True) and not ready_for_llm(window, allow_partial=allow_partial):
            self._kick("draft", window)
            if not force and idle < 6:
                return
        if not ready_for_llm(window, allow_partial=allow_partial):
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
            safe_print("   (no LLM — set FUNCTION_URL to build questions)")
            return
        self._kick("final", window)

    def _loop(self) -> None:
        while not self._stop.wait(0.12):
            try:
                self.flush()
            except Exception as exc:
                safe_print(f"Question flush failed: {exc}")
