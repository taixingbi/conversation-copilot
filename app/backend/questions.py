from __future__ import annotations

import os
import re
import sys
import threading
import time
from collections import deque
from pathlib import Path

from events import BUS
from llm.answer import LlmAnswerer
from memory import ProfileIndex, SessionMemory
from metrics import LatencyTracker, now_mono

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


def questions_path_for(transcribe_path: str | Path) -> Path:
    p = Path(transcribe_path)
    name = p.name
    if name.endswith("_transcribe.txt"):
        name = name[: -len("_transcribe.txt")] + "_questions.txt"
    else:
        name = p.stem + "_questions.txt"
    return p.with_name(name)


def _norm_blob(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


class QuestionExtractor:
    def __init__(
        self,
        *,
        function_url: str,
        api_key: str,
        model: str,
        out_path: Path,
        interval_sec: float = 1.0,
        window_sec: float = 5.0,
        fast_model: str = "",
        metrics: LatencyTracker | None = None,
        memory: SessionMemory | None = None,
        profile: ProfileIndex | None = None,
    ):
        self.memory = memory or SessionMemory()
        self.profile = profile
        self.llm = LlmAnswerer(
            function_url=function_url,
            api_key=api_key,
            model=model,
            fast_model=fast_model,
            memory=self.memory,
            profile=profile,
        )
        self.out_path = Path(out_path)
        self.interval_sec = interval_sec
        self.window_sec = window_sec
        self.metrics = metrics or LatencyTracker()
        self._history: deque = deque()
        self._known: list[str] = []
        self._queued_window: list[str] | None = None
        self._lock = threading.RLock()
        self._dispatch_lock = threading.Lock()
        self._last_call = 0.0
        self._t_last_ext = 0.0
        self._t_ext_end = 0.0
        self._t_shown = 0.0
        self._last_stt_ms = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._final_gen = 0
        self._final_alive = False
        self._final_blob = ""
        self._shown_q = ""
        self._shown_a = ""
        self._logged_ext = False
        self.auto_qa = False

    def set_auto(self, on: bool) -> None:
        self.auto_qa = bool(on)
        if not self.auto_qa:
            self._interrupt()
        BUS.publish("qa_status", text="Listening for a question…" if self.auto_qa else "Q&A is off. Open QA to start.")

    def start(self) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.touch(exist_ok=True)
        if not self._thread.is_alive():
            self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self.set_auto(False)
        if self._thread.is_alive():
            self._thread.join(timeout=2)
        summary = self.metrics.summary()
        if summary:
            safe_print(f"lat  {summary}", DIM)

    def add_ext(self, ts: str, text: str, *, t_mono: float | None = None, stt_ms: float = 0.0) -> None:
        body = text.strip()
        if not body:
            return
        stamp = t_mono if t_mono is not None else now_mono()
        with self._lock:
            self._history.append((stamp, body))
            self._recent_window(now_mono())
            self._t_last_ext = stamp
            if stt_ms:
                self._last_stt_ms = stt_ms

    def _recent_window(self, now: float) -> list[str]:
        # Caller holds _lock. Speech timestamps, not word counts, bound the window.
        cutoff = now - self.window_sec
        self._history = deque((stamp, text) for stamp, text in self._history if stamp >= cutoff)
        return [text for stamp, text in self._history if stamp <= now]

    def trigger(self) -> bool:
        """Evaluate the current five-second window when Q/A is enabled."""
        with self._lock:
            window = self._recent_window(now_mono())
        if not window:
            self._seed_from_bus()
            with self._lock:
                window = self._recent_window(now_mono())
        if not window:
            BUS.publish("qa_status", text="Listening for a question…")
            return False
        self.flush(force=True)
        return True

    def _seed_from_bus(self) -> None:
        wall_now, mono_now = time.time(), now_mono()
        with self._lock:
            for ev in BUS.snapshot():
                if ev.get("type") != "transcript" or not ev.get("text"):
                    continue
                age = wall_now - ev.get("t", 0)
                if 0 <= age <= self.window_sec:
                    self._history.append((mono_now - age, f"[{ev.get('label') or ''}] {ev['text']}"))

    def set_model(self, model: str) -> None:
        fast = (os.environ.get("LLM_FAST_MODEL") or "").strip()
        self.llm.set_model(model, sync_fast=not fast)

    def forget(self, question: str) -> None:
        q = (question or "").strip()
        if not q:
            return
        with self._lock:
            self._known = [k for k in self._known if _norm_blob(k) != _norm_blob(q)]
            current = bool(
                (self._shown_q and _norm_blob(self._shown_q) == _norm_blob(q))
                or (self._final_blob and _norm_blob(self._final_blob) == _norm_blob(q))
            )
        if current:
            self._interrupt()
        self.memory.forget(q)

    def forget_all(self) -> None:
        with self._lock:
            self._known.clear()
        self._interrupt()
        self.memory.clear()

    def _write_qa(self, question: str, answer: str) -> None:
        ts = time.strftime("%H:%M:%S")
        block = f"[{ts}] Q: {question}\nA: {answer}\n\n"
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with self.out_path.open("a", encoding="utf-8") as f:
            f.write(block)
            f.flush()
        with self._lock:
            self._known.append(question)

    def _blob(self, window: list[str]) -> str:
        return _norm_blob(" ".join(window))

    def _kick(self, kind: str, window: list[str], *, force: bool = False) -> None:
        if kind == "draft" or not self.llm.enabled:
            return
        blob = self._blob(window)
        if not blob:
            return
        if force:
            self._interrupt()
        with self._lock:
            if self._final_alive or not self.auto_qa or self._stop.is_set():
                return
            self._t_ext_end = self._t_last_ext or now_mono()
            self._final_gen += 1
            self._final_alive = True
            self._final_blob = blob
            gen = self._final_gen
        self._spawn(window, "final", gen)

    def _interrupt(self) -> None:
        with self._lock:
            self._final_gen += 1
            self._final_alive = False
            self._final_blob = ""
            self._queued_window = None
            self._reset_turn()
        self.llm.client.abort()
        BUS.publish("status", text="interrupted")

    def _spawn(self, window: list[str], kind: str, gen: int) -> None:
        if kind == "final":
            BUS.publish("qa_status", text="Generating answer…")
        threading.Thread(
            target=self._run_job,
            args=(list(window), kind, gen),
            daemon=True,
            name=f"llm-{kind}",
        ).start()

    def _stale(self, kind: str, gen: int) -> bool:
        return gen != self._final_gen

    def _run_job(self, window: list[str], kind: str, gen: int) -> None:
        t0 = time.perf_counter()
        question = answer = ""
        try:
            with self._lock:
                if self._stale(kind, gen):
                    return
                self.llm.answered_questions = list(self._known)
            for question, answer in self.llm.iter_qa(window, kind=kind):
                if self._stale(kind, gen):
                    return
            # A completed response is committed atomically with cancellation.
            with self._lock:
                if self._stale(kind, gen):
                    return
                elapsed = (time.perf_counter() - t0) * 1000
                self.metrics.observe("llm_total_ms", elapsed)
                if question and answer:
                    self.metrics.observe("llm_ttft_ms", elapsed)
                    self._on_done(question, answer, kind)
                    BUS.publish("qa_status", text="Listening for the next question…")
                else:
                    BUS.publish("qa_status", text="No complete question found. Listening…")
        except Exception as exc:
            safe_print(f"LLM {kind} failed: {exc}")
            with self._lock:
                if not self._stale(kind, gen):
                    BUS.publish("qa_status", text=f"Q&A failed: {exc}. Close and reopen QA to retry.")
        finally:
            queued = None
            with self._lock:
                if gen == self._final_gen:
                    self._final_alive = False
                    if not question or not answer:
                        self._final_blob = ""
                    queued, self._queued_window = self._queued_window, None
            if queued and self.auto_qa and not self._stop.is_set():
                self._kick("final", queued)

    def _record_ext_latency(self) -> None:
        if self._logged_ext or not self._t_shown or not self._t_ext_end:
            return
        self.metrics.observe("ext_to_answer_ms", (self._t_shown - self._t_ext_end) * 1000)
        self._logged_ext = True

    def _mark_shown(self, kind: str) -> None:
        if not self._t_shown:
            self._t_shown = now_mono()
        self._record_ext_latency()

    def _on_done(self, question: str, answer: str, kind: str) -> None:
        if not question or not answer:
            return
        self._mark_shown(kind)
        self._write_qa(question, answer)
        self.memory.add(question, answer)
        self._render(question, answer, kind)
        self._print_lat()
        self._reset_turn()

    def _reset_turn(self) -> None:
        self._t_shown = 0.0
        self._t_ext_end = 0.0
        self._shown_q = ""
        self._shown_a = ""
        self._logged_ext = False

    def _print_lat(self) -> None:
        extra = f"  stt {self._last_stt_ms:.0f}ms" if self._last_stt_ms else ""
        ttft = self.metrics.values("llm_ttft_ms")
        if ttft:
            extra += f"  ttft {ttft[-1]:.0f}ms"
        line = f"lat  {self.metrics.line('ext_to_answer_ms', label='ext→answer')}{extra}"
        safe_print(line, DIM)
        BUS.publish("lat", text=line, **self._lat_fields())

    def _lat_fields(self) -> dict:
        ext = self.metrics.values("ext_to_answer_ms")
        ttft = self.metrics.values("llm_ttft_ms")
        return {
            "ext_ms": ext[-1] if ext else None,
            "ttft_ms": ttft[-1] if ttft else None,
            "stt_ms": self._last_stt_ms or None,
        }

    def _render(self, question: str, answer: str, kind: str) -> None:
        self._shown_q = question
        self._shown_a = answer
        ts = time.strftime("%H:%M:%S")
        safe_print(f"Q  [{ts}] {question}", YELLOW)
        safe_print(f"A  {answer}", GREEN)
        BUS.publish(
            "qa", question=question, answer=answer, kind=kind,
            done=True, restart=True, **self._lat_fields(),
        )

    def flush(self, force: bool = False) -> None:
        with self._dispatch_lock:
            with self._lock:
                now = now_mono()
                if not force and self._last_call and now - self._last_call < self.interval_sec:
                    return
                if not self.llm.enabled:
                    return
                window = self._recent_window(now)
                self._last_call = now
                if not window:
                    return
                if self._final_alive:
                    # Keep the newest snapshot while the asynchronous request completes.
                    self._queued_window = window
                    return
            self._kick("final", window)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_sec):
            if not self.auto_qa:
                continue
            try:
                self.flush()
            except Exception as exc:
                safe_print(f"Question flush failed: {exc}")
