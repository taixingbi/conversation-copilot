from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from events import BUS
from stt import MODEL_ALIASES, Transcriber, resolve_backend, resolve_model_name

THEME_OPTIONS = ("black", "white")
THEME_ALIASES = {
    "black": "black",
    "mac-black": "black",
    "mac black": "black",
    "dark": "black",
    "white": "white",
    "mac-white": "white",
    "mac white": "white",
    "light": "white",
}

WHISPER_OPTIONS = [
    "tiny",
    "base",
    "small",
    "medium",
    "large",
    "distil-small",
    "distil-medium",
    "distil-large",
]

_WHISPER_SHORT = {
    "tiny": "tiny",
    "tiny.en": "tiny",
    "base": "base",
    "base.en": "base",
    "small": "small",
    "small.en": "small",
    "medium": "medium",
    "medium.en": "medium",
    "large": "large",
    "large-v3": "large",
    "distil": "distil-small",
    "distil-small": "distil-small",
    "distil-small.en": "distil-small",
    "distil-medium": "distil-medium",
    "distil-medium.en": "distil-medium",
    "distil-large": "distil-large",
    "distil-large-v3": "distil-large",
}


def resolve_env_path(store: Path, app_dir: Path) -> Path:
    for path in (app_dir / ".env", store / ".env"):
        if path.is_file():
            return path
    return app_dir / ".env"


def upsert_env(path: Path, updates: dict[str, str]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        raw = line.strip()
        if raw and not raw.startswith("#") and "=" in raw:
            key = raw.partition("=")[0].strip()
            if key in updates:
                out.append(f"{key}={updates[key]}")
                seen.add(key)
                continue
        out.append(line)
    if seen != set(updates) and out and out[-1].strip():
        out.append("")
    for key, value in updates.items():
        if key not in seen:
            out.append(f"{key}={value}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def theme_key(raw: str | None = None) -> str:
    value = (raw if raw is not None else os.environ.get("OVERLAY_THEME", "white")).strip().lower()
    return THEME_ALIASES.get(value, "white")


def whisper_key(raw: str | None = None) -> str:
    value = (raw if raw is not None else os.environ.get("WHISPER_MODEL", "medium")).strip().lower()
    return _WHISPER_SHORT.get(value, "medium")


class Runtime:
    """Live overlay settings: persist to .env and apply without a full restart."""

    def __init__(self, env_path: Path, *, prompt_dir: Path | None = None) -> None:
        self.env_path = env_path
        self.prompt_dir = prompt_dir or env_path.parent / "prompt"
        self.extractor = None
        self.worker = None
        self._lock = threading.Lock()
        self._whisper_gen = 0
        self._summary_gen = 0
        self._chat_lock = threading.Lock()
        self._chat_history: list[dict] = []
        self.whisper_loading = False
        self._load_prompt()

    def prompt_path(self) -> Path:
        return self.prompt_dir / "qa_prompt.txt"

    def summary_prompt_path(self) -> Path:
        return self.prompt_dir / "summary_prompt.txt"

    def chat_prompt_path(self) -> Path:
        return self.prompt_dir / "chat_prompt.txt"

    def _load_prompt(self) -> None:
        path = self.prompt_path()
        if path.is_file():
            os.environ["QA_PROMPT"] = path.read_text(encoding="utf-8").strip()
        sp = self.summary_prompt_path()
        if sp.is_file():
            os.environ["SUMMARY_PROMPT"] = sp.read_text(encoding="utf-8").strip()
        cp = self.chat_prompt_path()
        if cp.is_file():
            os.environ["CHAT_PROMPT"] = cp.read_text(encoding="utf-8").strip()

    def _write_prompt_file(self, path: Path, text: str) -> None:
        if text:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text + "\n", encoding="utf-8")
        elif path.is_file():
            path.unlink()

    def snapshot(self) -> dict:
        from llm.prompt import effective_prompt, effective_summary_prompt, effective_chat_prompt

        llm = (os.environ.get("LLM_MODEL") or "nova-pro").strip()
        try:
            confidence = float(os.environ.get("RECOGNITION_CONFIDENCE") or "0.70")
        except ValueError:
            confidence = 0.70
        return {
            "llm_model": llm,
            "summary_model": (os.environ.get("SUMMARY_MODEL") or llm).strip(),
            "whisper_model": whisper_key(),
            "whisper_options": list(WHISPER_OPTIONS),
            "whisper_loading": self.whisper_loading,
            "theme": theme_key(),
            "theme_options": list(THEME_OPTIONS),
            "prompt": effective_prompt(),
            "summary_prompt": effective_summary_prompt(),
            "chat_prompt": effective_chat_prompt(),
            "chat_model": llm,
            "recognition_confidence": min(0.99, max(0.0, confidence)),
            "qa_enabled": bool(self.extractor.auto_qa) if self.extractor is not None else False,
        }

    def apply(
        self,
        *,
        llm_model: str | None = None,
        whisper_model: str | None = None,
        theme: str | None = None,
        prompt: str | None = None,
        summary_model: str | None = None,
        summary_prompt: str | None = None,
        chat_prompt: str | None = None,
        recognition_confidence: float | None = None,
    ) -> dict:
        updates: dict[str, str] = {}
        if llm_model is not None:
            model = llm_model.strip()
            if not model or "\n" in model:
                raise ValueError("Model is empty")
            updates["LLM_MODEL"] = model
            os.environ["LLM_MODEL"] = model
            if self.extractor is not None:
                self.extractor.set_model(model)
        if whisper_model is not None:
            raw = whisper_model.strip().lower()
            if raw not in MODEL_ALIASES:
                raise ValueError(f"Unknown WHISPER_MODEL={whisper_model!r}")
            key = whisper_key(raw)
            updates["WHISPER_MODEL"] = key
            os.environ["WHISPER_MODEL"] = key
            self._reload_whisper(key)
        if theme is not None:
            key = theme_key(theme)
            updates["OVERLAY_THEME"] = key
            os.environ["OVERLAY_THEME"] = key
        if prompt is not None:
            text = str(prompt).replace("\0", "").strip()
            if len(text) > 8000:
                raise ValueError("Prompt is too long")
            os.environ["QA_PROMPT"] = text
            self._write_prompt_file(self.prompt_path(), text)
        if summary_model is not None:
            model = summary_model.strip()
            if not model or "\n" in model:
                raise ValueError("Summary model is empty")
            updates["SUMMARY_MODEL"] = model
            os.environ["SUMMARY_MODEL"] = model
        if summary_prompt is not None:
            text = str(summary_prompt).replace("\0", "").strip()
            if len(text) > 8000:
                raise ValueError("Summary prompt is too long")
            os.environ["SUMMARY_PROMPT"] = text
            self._write_prompt_file(self.summary_prompt_path(), text)
        if chat_prompt is not None:
            text = str(chat_prompt).replace("\0", "").strip()
            if len(text) > 8000:
                raise ValueError("Chat prompt is too long")
            os.environ["CHAT_PROMPT"] = text
            self._write_prompt_file(self.chat_prompt_path(), text)
        if recognition_confidence is not None:
            try:
                value = float(recognition_confidence)
            except (TypeError, ValueError) as exc:
                raise ValueError("Recognition confidence must be a number") from exc
            value = min(0.99, max(0.0, value))
            updates["RECOGNITION_CONFIDENCE"] = f"{value:.2f}"
            os.environ["RECOGNITION_CONFIDENCE"] = updates["RECOGNITION_CONFIDENCE"]
        if updates:
            upsert_env(self.env_path, updates)
        snap = self.snapshot()
        BUS.publish("config", **snap)
        return snap

    def forget_question(self, question: str) -> None:
        q = (question or "").strip()
        if not q:
            self.clear_questions()
            return
        if self.extractor is not None:
            self.extractor.forget(q)
        BUS.drop_qa(q)
        BUS.publish("qa_gone", question=q)

    def clear_questions(self) -> None:
        if self.extractor is not None:
            self.extractor.forget_all()
        BUS.drop_qa(None)
        BUS.publish("qa_gone", question="")

    def clear_summary(self) -> None:
        with self._lock:
            self._summary_gen += 1
        BUS.drop_summary()
        BUS.publish("summary_gone")

    def set_qa_enabled(self, enabled: bool) -> dict:
        if self.extractor is None:
            raise ValueError("runtime unavailable")
        if enabled and not self.extractor.llm.enabled:
            raise ValueError("Set FUNCTION_URL in .env and restart to enable Q&A")
        self.extractor.set_auto(bool(enabled))
        if enabled and self.extractor.llm.enabled:
            self.extractor.trigger()
        snap = self.snapshot()
        BUS.publish("config", **snap)
        return snap

    def start_summary(self) -> dict:
        if self.extractor is None or not getattr(self.extractor, "llm", None):
            raise ValueError("runtime unavailable")
        if not self.extractor.llm.enabled:
            raise ValueError("LLM is not configured")
        with self._lock:
            self._summary_gen += 1
            gen = self._summary_gen
        threading.Thread(target=self._run_summary, args=(gen,), daemon=True, name="summary").start()
        BUS.publish("summary", text="Starting summary…", done=False)
        return {"ok": True}

    def _read_text(self, path: Path) -> str:
        try:
            if path.is_file():
                return path.read_text(encoding="utf-8").strip()
        except OSError:
            return ""
        return ""

    def _session_transcript(self) -> str:
        if self.extractor is not None:
            qpath = Path(self.extractor.out_path)
            tname = qpath.name
            if tname.endswith("_questions.txt"):
                tpath = qpath.with_name(tname[: -len("_questions.txt")] + "_transcribe.txt")
            else:
                tpath = qpath.with_name(qpath.stem + "_transcribe.txt")
            transcript = self._read_text(tpath)
            if transcript:
                return transcript
        lines = [
            f"[{ev.get('ts') or ''}] [{ev.get('label') or ''}] {ev.get('text')}"
            for ev in BUS.snapshot()
            if ev.get("type") == "transcript" and ev.get("text")
        ]
        return "\n".join(lines)

    def chat_history(self) -> list[dict]:
        with self._lock:
            return [dict(message) for message in self._chat_history]

    def ask_chat(self, question: str) -> dict:
        from llm.prompt import conversation_chat_prompt

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Enter a question about the conversation")
        question = question.strip()
        if len(question) > 8000:
            raise ValueError("Question is too long (maximum 8,000 characters)")
        if self.extractor is None or not self.extractor.llm.enabled:
            raise ValueError("LLM is not configured")
        if not self._chat_lock.acquire(blocking=False):
            raise ValueError("Wait for the current chat answer before sending another question")
        try:
            transcript = self._session_transcript()
            if not transcript:
                raise ValueError("No transcript yet. Start a conversation first.")
            answer = self.extractor.llm.client.chat(
                conversation_chat_prompt(question, transcript, self.chat_history()),
                max_tokens=900,
            ).strip()
            if not answer:
                raise RuntimeError("Chat returned an empty answer. Try again.")
            with self._lock:
                self._chat_history.extend([
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ])
            return {"messages": self.chat_history()}
        finally:
            self._chat_lock.release()

    def _run_summary(self, gen: int) -> None:
        from llm.client import ChatClient, strip_think
        from llm.prompt import summary_prompt

        if gen != self._summary_gen:
            return
        transcript = self._session_transcript()
        if not transcript:
            if gen == self._summary_gen:
                BUS.publish("summary", text="Nothing to summarize yet.", done=True)
            return
        try:
            buf = ""
            last = 0.0
            model = (os.environ.get("SUMMARY_MODEL") or "").strip() or None
            source = self.extractor.llm.client
            # Summary streams must not be aborted when Q/A is toggled or cleared.
            client = ChatClient(source.url.removesuffix("/v1/chat/completions"), source.api_key, source.model)
            for delta in client.chat_stream(
                summary_prompt(transcript=transcript),
                max_tokens=900,
                model=model,
            ):
                if gen != self._summary_gen:
                    return
                if not delta:
                    continue
                buf += delta
                now = time.time()
                if now - last >= 0.12:
                    last = now
                    BUS.publish("summary", text=strip_think(buf), done=False)
            if gen != self._summary_gen:
                return
            text = strip_think(buf).strip() or "Summary was empty."
            BUS.publish("summary", text=text, done=True)
        except Exception as exc:
            if gen == self._summary_gen:
                BUS.publish("summary", text=f"Summary failed: {exc}", done=True)

    def _reload_whisper(self, key: str) -> None:
        if self.worker is None:
            return
        name = resolve_model_name(key)
        current = getattr(self.worker.stt, "model_name", "")
        if current == name and not self.whisper_loading:
            return
        with self._lock:
            self._whisper_gen += 1
            gen = self._whisper_gen
            self.whisper_loading = True
        BUS.publish("status", text=f"loading whisper {key}…")

        def load() -> None:
            try:
                backend = resolve_backend(name)
                stt = Transcriber(name, backend)
                with self._lock:
                    if gen != self._whisper_gen:
                        return
                    self.worker.set_transcriber(stt)
                    self.whisper_loading = False
                BUS.publish("status", text=f"whisper {key}/{backend}")
                BUS.publish("config", **self.snapshot())
            except Exception as exc:
                with self._lock:
                    if gen == self._whisper_gen:
                        self.whisper_loading = False
                BUS.publish("status", text=f"whisper failed: {exc}")
                BUS.publish("config", **self.snapshot())

        threading.Thread(target=load, daemon=True, name="whisper-reload").start()
