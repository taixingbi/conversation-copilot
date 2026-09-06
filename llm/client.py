from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections.abc import Iterator


def strip_think(content: str) -> str:
    return re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()


def _text_of(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if value.get("text"):
            return str(value["text"])
        if value.get("content"):
            return _text_of(value["content"])
        return ""
    if isinstance(value, list):
        return "".join(_text_of(p) for p in value)
    return str(value)


def _choice_text(payload: dict) -> str:
    choice = (payload.get("choices") or [{}])[0]
    delta = choice.get("delta") or {}
    if delta:
        return _text_of(delta.get("content"))
    msg = choice.get("message") or {}
    return _text_of(msg.get("content"))


class ChatClient:
    """OpenAI-compatible chat completions client (Bedrock Function URL)."""

    def __init__(self, function_url: str, api_key: str, model: str) -> None:
        base = (function_url or "").strip().rstrip("/")
        self.url = f"{base}/v1/chat/completions" if base else ""
        self.api_key = api_key
        self.model = model

    @property
    def enabled(self) -> bool:
        return bool(self.url)

    def _headers(self, *, stream: bool = False) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "x-api-key": self.api_key,
        }
        if stream:
            headers["Accept"] = "text/event-stream"
        return headers

    def _body(self, prompt: str, max_tokens: int, model: str, stream: bool) -> bytes:
        return json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0,
                "stream": stream,
            }
        ).encode("utf-8")

    def _request(self, prompt: str, max_tokens: int, model: str, stream: bool):
        if not self.url:
            raise RuntimeError("FUNCTION_URL is empty")
        req = urllib.request.Request(
            self.url,
            data=self._body(prompt, max_tokens, model, stream),
            method="POST",
            headers=self._headers(stream=stream),
        )
        try:
            return urllib.request.urlopen(req, timeout=90)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {detail[:500]}") from exc

    def _content_of(self, payload: dict) -> str:
        if payload.get("error") or payload.get("errorType"):
            raise RuntimeError(payload.get("detail") or payload.get("error") or payload)
        return strip_think(_choice_text(payload))

    def chat(self, prompt: str, max_tokens: int = 120, *, model: str | None = None) -> str:
        used = (model or self.model).strip()
        with self._request(prompt, max_tokens, used, False) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return self._content_of(payload)

    def chat_stream(
        self, prompt: str, max_tokens: int = 120, *, model: str | None = None
    ) -> Iterator[str]:
        """Yield content deltas. Falls back to one-shot if the proxy is not SSE."""
        used = (model or self.model).strip()
        got = False
        try:
            for piece in self._iter_stream(prompt, max_tokens, used):
                if piece:
                    got = True
                    yield piece
            if got:
                return
        except Exception:
            if got:
                raise
        text = self.chat(prompt, max_tokens=max_tokens, model=used)
        if text:
            yield text

    def _iter_stream(self, prompt: str, max_tokens: int, model: str) -> Iterator[str]:
        with self._request(prompt, max_tokens, model, True) as resp:
            ctype = (resp.headers.get("Content-Type") or "").lower()
            if "event-stream" in ctype or "text/plain" in ctype:
                yield from self._iter_sse(resp)
                return
            raw = resp.read().decode("utf-8", errors="replace")
        stripped = raw.lstrip()
        if stripped.startswith("data:"):
            yield from self._iter_sse_text(raw)
            return
        yield self._content_of(json.loads(raw))

    def _iter_sse(self, resp) -> Iterator[str]:
        while True:
            raw = resp.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace").strip()
            piece = self._sse_data(line)
            if piece:
                yield piece

    def _iter_sse_text(self, raw: str) -> Iterator[str]:
        for line in raw.splitlines():
            piece = self._sse_data(line.strip())
            if piece:
                yield piece

    def _sse_data(self, line: str) -> str:
        if not line.startswith("data:"):
            return ""
        data = line[5:].strip()
        if not data or data == "[DONE]":
            return ""
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            return ""
        if payload.get("error") or payload.get("errorType"):
            raise RuntimeError(payload.get("detail") or payload.get("error") or payload)
        return _choice_text(payload)
