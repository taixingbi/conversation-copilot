from __future__ import annotations

import os
from pathlib import Path

from ai.service import AiService
from audio.transcripts import TranscriptStore
from audio.service import AudioService
from events import BUS
from settings import Settings, whisper_key, theme_key, WHISPER_OPTIONS, THEME_OPTIONS


class Runtime:
    def __init__(self, env_path: Path, *, prompt_dir: Path | None = None):
        self.settings = Settings(env_path, prompt_dir=prompt_dir)
        self.transcripts = None
        self.audio = None
        self.ai = AiService(self._session_transcript)

    @property
    def extractor(self):
        return self.ai.extractor

    @extractor.setter
    def extractor(self, value):
        self.ai.extractor = value

    @property
    def worker(self):
        return self.audio.worker if self.audio is not None else None

    @worker.setter
    def worker(self, value):
        self.audio = AudioService(value, on_change=lambda: BUS.publish("config", **self.snapshot()))

    def prompt_path(self):
        return self.settings.prompt_path()

    def summary_prompt_path(self):
        return self.settings.summary_prompt_path()

    def chat_prompt_path(self):
        return self.settings.chat_prompt_path()

    @property
    def env_path(self):
        return self.settings.env_path

    def snapshot(self) -> dict:
        from ai.prompt import effective_prompt, effective_summary_prompt, effective_chat_prompt

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
            "whisper_available": self.worker is None or self.worker.stt.model_name != "streaming",
            "whisper_loading": self.audio.loading if self.audio is not None else False,
            "theme": theme_key(),
            "theme_options": list(THEME_OPTIONS),
            "prompt": effective_prompt(),
            "summary_prompt": effective_summary_prompt(),
            "chat_prompt": effective_chat_prompt(),
            "chat_model": llm,
            "recognition_confidence": min(0.99, max(0.0, confidence)),
            "qa_enabled": bool(self.extractor.auto_qa) if self.extractor is not None else False,
        }

    def apply(self, **changes):
        if self.audio is not None and self.worker.stt.model_name == "streaming":
            model = changes.get("whisper_model")
            if model is not None and whisper_key(model) != whisper_key():
                raise ValueError("Whisper model switching is unavailable in streaming mode")
            changes.pop("whisper_model", None)
        self.settings.apply(**changes)
        if changes.get("llm_model") is not None and self.extractor is not None:
            self.extractor.set_model(changes["llm_model"])
        if changes.get("whisper_model") is not None:
            self._reload_whisper(whisper_key(changes["whisper_model"]))
        snap = self.snapshot()
        BUS.publish("config", **snap)
        return snap

    def set_qa_enabled(self, enabled):
        self.ai.set_qa_enabled(enabled)
        snap = self.snapshot()
        BUS.publish("config", **snap)
        return snap

    def forget_question(self, question):
        return self.ai.forget_question(question)

    def clear_questions(self):
        return self.ai.clear_questions()

    def clear_summary(self):
        return self.ai.clear_summary()

    def start_summary(self):
        return self.ai.start_summary()

    def chat_history(self):
        return self.ai.chat_history()

    def ask_chat(self, question):
        return self.ai.ask_chat(question)

    def _run_summary(self, gen):
        return self.ai.summary._run_summary(gen)

    def _session_transcript(self) -> str:
        if self.transcripts is not None:
            return self.transcripts.text()
        return TranscriptStore(bus=BUS).text()

    def _reload_whisper(self, key):
        if self.audio is not None:
            self.audio.reload(key)
