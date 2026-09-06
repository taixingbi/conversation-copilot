from __future__ import annotations

import json
from pathlib import Path

_PATH = Path(__file__).resolve().parent.parent / "prompt" / "qa_instructions.json"


def _load() -> dict:
    return json.loads(_PATH.read_text(encoding="utf-8"))


def _build(focus: str) -> str:
    return _load()["base"].replace("{focus}", focus)


def _instructions_for(ext_lines: list[str]) -> str:
    data = _load()
    blob = " ".join(ext_lines).lower()
    keys = [k for k in data.get("keywords", {}) if k in blob]
    focus = data["keywords"][max(keys, key=len)] if keys else data["default"]
    return _build(focus)


def qa_prompt(ext_lines: list[str]) -> str:
    lines = "\n".join(f"- {line}" for line in ext_lines if line.strip())
    return f"{_instructions_for(ext_lines)}\n\nInterviewer lines:\n{lines or '(none)'}\n"
