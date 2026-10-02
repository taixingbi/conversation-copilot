from __future__ import annotations

import threading



class ConversationChat:
    def __init__(self, transcript_reader, client_provider):
        self._session_transcript = transcript_reader
        self._client_provider = client_provider
        self._lock = threading.Lock()
        self._chat_lock = threading.Lock()
        self._chat_history = []

    def chat_history(self) -> list[dict]:
        with self._lock:
            return [dict(message) for message in self._chat_history]

    def ask_chat(self, question: str) -> dict:
        from ai.prompt import conversation_chat_prompt

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Enter a question about the conversation")
        question = question.strip()
        if len(question) > 8000:
            raise ValueError("Question is too long (maximum 8,000 characters)")
        client = self._client_provider()
        if client is None or not client.enabled:
            raise ValueError("LLM is not configured")
        if not self._chat_lock.acquire(blocking=False):
            raise ValueError("Wait for the current chat answer before sending another question")
        try:
            transcript = self._session_transcript()
            if not transcript:
                raise ValueError("No transcript yet. Start a conversation first.")
            answer = client.chat(
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
