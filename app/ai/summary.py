from __future__ import annotations

import os
import threading
import time

from events import BUS


class ConversationSummary:
    def __init__(self, transcript_reader, client_provider):
        self._session_transcript = transcript_reader
        self._client_provider = client_provider
        self._lock = threading.Lock()
        self._summary_gen = 0
        self._clients = {}

    def clear_summary(self) -> None:
        with self._lock:
            self._summary_gen += 1
            clients = list(self._clients.values())
            self._clients.clear()
            BUS.drop_summary()
            BUS.publish("summary_gone")
        for client in clients:
            client.abort()

    def start_summary(self) -> dict:
        source = self._client_provider()
        if source is None:
            raise ValueError("runtime unavailable")
        if not source.enabled:
            raise ValueError("LLM is not configured")
        with self._lock:
            self._summary_gen += 1
            gen = self._summary_gen
            clients = list(self._clients.values())
            self._clients.clear()
            BUS.publish("summary", text="Starting summary…", done=False)
        for client in clients:
            client.abort()
        threading.Thread(target=self._run_summary, args=(gen,), daemon=True, name="summary").start()
        return {"ok": True}

    def _publish(self, gen, **payload):
        with self._lock:
            if gen == self._summary_gen:
                BUS.publish("summary", **payload)

    def _run_summary(self, gen: int) -> None:
        from ai.client import ChatClient, strip_think
        from ai.prompt import summary_prompt

        if gen != self._summary_gen:
            return
        transcript = self._session_transcript()
        if not transcript:
            if gen == self._summary_gen:
                self._publish(gen, text="Nothing to summarize yet.", done=True)
            return
        try:
            buf = ""
            last = 0.0
            model = (os.environ.get("SUMMARY_MODEL") or "").strip() or None
            source = self._client_provider()
            # Summary streams must not be aborted when Q/A is toggled or cleared.
            client = ChatClient(source.url.removesuffix("/v1/chat/completions"), source.api_key, source.model)
            with self._lock:
                if gen != self._summary_gen:
                    return
                self._clients[gen] = client
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
                    self._publish(gen, text=strip_think(buf), done=False)
            if gen != self._summary_gen:
                return
            text = strip_think(buf).strip() or "Summary was empty."
            self._publish(gen, text=text, done=True)
        except Exception as exc:
            if gen == self._summary_gen:
                self._publish(gen, text=f"Summary failed: {exc}", done=True)

        finally:
            with self._lock:
                self._clients.pop(gen, None)
