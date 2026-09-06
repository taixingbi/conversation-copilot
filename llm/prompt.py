from __future__ import annotations

import json
from pathlib import Path

_PATH = Path(__file__).resolve().parent.parent / "prompt" / "qa_instructions.json"


def _load() -> dict:
    return json.loads(_PATH.read_text(encoding="utf-8"))


def _build(template: str, focus: str) -> str:
    return template.replace("{focus}", focus)


def _focus_for(ext_lines: list[str], data: dict) -> str:
    blob = " ".join(ext_lines).lower()
    keys = [k for k in data.get("keywords", {}) if k in blob]
    return data["keywords"][max(keys, key=len)] if keys else data["default"]


def _instructions_for(ext_lines: list[str], *, kind: str = "base") -> str:
    data = _load()
    template = data.get(kind) or data["base"]
    return _build(template, _focus_for(ext_lines, data))


def qa_prompt(ext_lines: list[str], *, kind: str = "base") -> str:
    lines = "\n".join(f"- {line}" for line in ext_lines if line.strip())
    return f"{_instructions_for(ext_lines, kind=kind)}\n\nInterviewer lines:\n{lines or '(none)'}\n"
