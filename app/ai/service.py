"""AI application operations exposed by the runtime facade.

Consumes finalized transcript through the facade's reader; owns QA, chat and
summary behavior. Does not import audio engines or UI presentation code.
"""
from __future__ import annotations


from events import BUS


class AiService:
    def __init__(self, transcript_reader):
        self.extractor = None
        self._session_transcript = transcript_reader
        from ai.chat import ConversationChat
        from ai.summary import ConversationSummary
        provider = lambda: self.extractor.llm.client if self.extractor is not None else None
        self.chat = ConversationChat(transcript_reader, lambda: provider().fork() if provider() is not None else None)
        self.summary = ConversationSummary(transcript_reader, provider)

    def close(self):
        self.clear_summary()
        if self.extractor is not None:
            self.extractor.close()

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

    def set_qa_enabled(self, enabled: bool) -> dict:
        if self.extractor is None:
            raise ValueError("runtime unavailable")
        if enabled and not self.extractor.llm.enabled:
            raise ValueError("Set FUNCTION_URL in .env and restart to enable Q&A")
        self.extractor.set_auto(bool(enabled))
        if enabled and self.extractor.llm.enabled:
            self.extractor.trigger()
        return {"qa_enabled": bool(self.extractor.auto_qa)}

    def clear_summary(self):
        return self.summary.clear_summary()

    def start_summary(self):
        return self.summary.start_summary()

    def chat_history(self):
        return self.chat.chat_history()

    def ask_chat(self, question):
        return self.chat.ask_chat(question)
