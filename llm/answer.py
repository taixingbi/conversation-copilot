from __future__ import annotations

import re

from llm.client import ChatClient
from llm.prompt import qa_prompt

_Q = re.compile(r"^Q:\s*", re.I)
_A = re.compile(r"^A:\s*", re.I)


def short_answer(text: str) -> str:
    """Keep 1-2 short sentences."""
    t = re.sub(r"\s+", " ", (text or "").strip())
    t = re.sub(r"^A:\s*", "", t, flags=re.I)
    if not t:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", t)
    return " ".join(parts[:2]).strip()


class LlmAnswerer:
    """One LLM call: reconstruct Q from EXT lines and answer it. No notes or extra context."""

    def __init__(self, *, function_url: str, api_key: str, model: str) -> None:
        self.client = ChatClient(function_url, api_key, model)

    @property
    def enabled(self) -> bool:
        return self.client.enabled

    def qa_from_ext(self, ext_lines: list[str]) -> tuple[str, str]:
        content = self.client.chat(qa_prompt(ext_lines), max_tokens=220)
        return parse_qa(content)


def parse_qa(content: str) -> tuple[str, str]:
    raw = (content or "").strip()
    if not raw or raw.upper() == "SKIP":
        return "", ""
    question, answer_parts = "", []
    for line in raw.splitlines():
        t = line.strip()
        if t.upper() == "SKIP":
            return "", ""
        if _Q.match(t):
            question = _Q.sub("", t).strip()
            answer_parts = []
        elif _A.match(t):
            answer_parts.append(_A.sub("", t).strip())
        elif question:
            answer_parts.append(t)
    return question, short_answer(" ".join(answer_parts))
