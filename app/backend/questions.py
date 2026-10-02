from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from collections import deque
from pathlib import Path
from uuid import uuid4

from events import BUS
from llm.answer import LlmAnswerer
from llm.extract import ExtractedQuestion
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
        window_sec: float = 45.0,
        target_speakers: tuple[str, ...] = ("EXT",),
        fast_model: str = "",
        metrics: LatencyTracker | None = None,
        memory: SessionMemory | None = None,
        profile: ProfileIndex | None = None,
    ):
        if interval_sec <= 0 or window_sec <= 0:
            raise ValueError("QA interval and window must be positive")
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
        self.target_speakers = tuple(s.strip().upper() for s in target_speakers if s.strip())
        self.current_question: dict | None = None
        self.previous_questions: deque = deque(maxlen=30)
        self._revision = 0
        self._dispatched_revision = -1
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
            label, words = self._split_speaker(body)
            if self._history:
                old_stamp, old = self._history[-1]
                old_label, old_words = self._split_speaker(old)
                if label == old_label and 0 <= stamp - old_stamp <= 3:
                    before, after = self._fingerprint(old_words), self._fingerprint(words)
                    if after == before or (after and before.startswith(after + " ")):
                        return
                    if before and after.startswith(before + " "):
                        self._history.pop()
            self._history.append((stamp, body))
            if self._eligible(label):
                self._revision += 1
            self._recent_window(now_mono())
            self._t_last_ext = stamp
            if stt_ms:
                self._last_stt_ms = stt_ms

    @staticmethod
    def _fingerprint(text: str) -> str:
        return re.sub(r"[^\w]+", " ", text.casefold()).strip()

    @staticmethod
    def _split_speaker(text: str) -> tuple[str, str]:
        match = re.match(r"^\[([^]]+)\]\s*(.*)$", text, re.S)
        return (match[1].upper(), match[2]) if match else ("", text)

    def _eligible(self, label: str) -> bool:
        return not label or any(label == s or (s in {"EXT", "MIC"} and label.startswith(s + "-"))
                                for s in self.target_speakers)

    def _recent_window(self, now: float) -> list[str]:
        # Caller holds _lock. Speech timestamps, not word counts, bound the window.
        cutoff = now - self.window_sec
        self._history = deque((stamp, text) for stamp, text in self._history if stamp >= cutoff)
        return [text for stamp, text in self._history if stamp <= now]

    def trigger(self) -> bool:
        """Evaluate the recent transcript window when Q/A is enabled."""
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
                    self.add_ext("", f"[{ev.get('label') or ''}] {ev['text']}", t_mono=mono_now - age)

    def set_model(self, model: str) -> None:
        fast = (os.environ.get("LLM_FAST_MODEL") or "").strip()
        self.llm.set_model(model, sync_fast=not fast)

    def forget(self, question: str) -> None:
        q = (question or "").strip()
        if not q:
            return
        with self._lock:
            self._known = [k for k in self._known if _norm_blob(k) != _norm_blob(q)]
            self.previous_questions = deque(
                (item for item in self.previous_questions if _norm_blob(item["question"]) != _norm_blob(q)),
                maxlen=30,
            )
            current = bool(
                (self.current_question and _norm_blob(self.current_question["question"]) == _norm_blob(q))
                or (self._shown_q and _norm_blob(self._shown_q) == _norm_blob(q))
                or (self._final_blob and _norm_blob(self._final_blob) == _norm_blob(q))
            )
            if current:
                self.current_question = None
                self._interrupt()
        self.memory.forget(q)

    def forget_all(self) -> None:
        with self._lock:
            self._known.clear()
            self.current_question = None
            self.previous_questions.clear()
            self._interrupt()
        self.memory.clear()

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
            BUS.publish("qa_status", text="Identifying the latest question…")
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
        question = ""
        try:
            with self._lock:
                if self._stale(kind, gen):
                    return
                current = self.current_question["question"] if self.current_question else ""
                previous = [item["question"] for item in self.previous_questions]
            result = self.llm.extract_latest(window, current=current, previous=previous, speakers=self.target_speakers)
            if result:
                question = result.question
            # A completed response is committed atomically with cancellation.
            with self._lock:
                if self._stale(kind, gen):
                    return
                elapsed = (time.perf_counter() - t0) * 1000
                self.metrics.observe("llm_total_ms", elapsed)
                if question:
                    self.metrics.observe("llm_ttft_ms", elapsed)
                    self._on_extracted(result, kind)
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
                    if not question:
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

    def _on_extracted(self, result: ExtractedQuestion, kind: str) -> None:
        question, context = result.question, result.context
        fingerprint = self._fingerprint(question)
        current = self.current_question
        if current and fingerprint == current["fingerprint"]:
            return
        if any(item["fingerprint"] == fingerprint for item in self.previous_questions):
            return
        update = result.action == "UPDATE" and current is not None
        replaced = current["question"] if update else ""
        item = {"id": current["id"] if update else uuid4().hex,
                "question": question, "context": context, "fingerprint": fingerprint}
        self._mark_shown(kind)
        # Append revisions as an audit trail; live state and replay retain one card per id.
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with self.out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({**item, "t": time.time(), "action": "UPDATE" if update else "NEW"}, ensure_ascii=False) + "\n")
        if current and not update:
            self.previous_questions.append(current)
        self.current_question = item
        if update:
            self._known = [q for q in self._known if q != replaced]
        self._known.append(question)
        self._known = self._known[-31:]
        BUS.publish("qa", question=question, context=context, answer=context,
                    question_id=item["id"], replaces_question=replaced,
                    action="UPDATE" if update else "NEW", mode="extraction",
                    kind=kind, done=True, restart=True, **self._lat_fields())
        safe_print(f"Q  {question}", YELLOW)
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

    def flush(self, force: bool = False) -> None:
        with self._dispatch_lock:
            with self._lock:
                now = now_mono()
                if not force and self._last_call and now - self._last_call < self.interval_sec:
                    return
                if not self.llm.enabled or not self.auto_qa:
                    return
                if not force and self._revision == self._dispatched_revision:
                    return
                window = self._recent_window(now)
                self._last_call = now
                if not window or not any(self._eligible(self._split_speaker(line)[0]) for line in window):
                    return
                self._dispatched_revision = self._revision
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
