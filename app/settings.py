from __future__ import annotations

import math
import os
from pathlib import Path

from audio.stt import MODEL_ALIASES

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


class Settings:
    """Configuration persistence; no running services or UI events."""

    def __init__(self, env_path: Path, *, prompt_dir: Path | None = None):
        self.env_path = env_path
        self.prompt_dir = prompt_dir or env_path.parent / "prompt"
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
        for value in (llm_model, summary_model):
            if value is not None and (not isinstance(value, str) or not value.strip() or "\n" in value or "\r" in value):
                raise ValueError("Model is invalid")
        if whisper_model is not None and (not isinstance(whisper_model, str) or whisper_model.strip().lower() not in MODEL_ALIASES):
            raise ValueError("Unknown WHISPER_MODEL")
        for value in (prompt, summary_prompt, chat_prompt):
            if value is not None and len(str(value).replace("\0", "").strip()) > 8000:
                raise ValueError("Prompt is too long")
        if recognition_confidence is not None:
            try:
                if not math.isfinite(float(recognition_confidence)):
                    raise ValueError("Recognition confidence must be finite")
            except (TypeError, ValueError) as exc:
                raise ValueError("Recognition confidence must be a finite number") from exc
        updates: dict[str, str] = {}
        if llm_model is not None:
            model = llm_model.strip()
            if not model or "\n" in model:
                raise ValueError("Model is empty")
            updates["LLM_MODEL"] = model
            os.environ["LLM_MODEL"] = model
        if whisper_model is not None:
            raw = whisper_model.strip().lower()
            if raw not in MODEL_ALIASES:
                raise ValueError(f"Unknown WHISPER_MODEL={whisper_model!r}")
            key = whisper_key(raw)
            updates["WHISPER_MODEL"] = key
            os.environ["WHISPER_MODEL"] = key
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
        return updates
