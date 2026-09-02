from __future__ import annotations

import json
import re
import urllib.error
import urllib.request


def strip_think(content: str) -> str:
    return re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()


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

    def chat(self, prompt: str, max_tokens: int = 120) -> str:
        if not self.url:
            raise RuntimeError("FUNCTION_URL is empty")
        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0,
                "stream": False,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            self.url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "x-api-key": self.api_key,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {detail[:500]}") from exc
        content = (
            payload.get("choices") or [{}]
        )[0].get("message", {}).get("content") or ""
        if isinstance(content, dict):
            content = content.get("content") or json.dumps(content)
        if payload.get("error") or payload.get("errorType"):
            raise RuntimeError(payload.get("detail") or payload.get("error") or payload)
        return strip_think(str(content))
