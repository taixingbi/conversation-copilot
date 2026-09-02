from __future__ import annotations

QA_INSTRUCTIONS = (
    "You are helping a candidate in a live job interview.\n"
    "The interviewer speech is noisy ASR, split across several lines.\n"
    "Reconstruct the complete question they asked, in clear English.\n"
    "Write the question they actually asked. Do not invent a different question.\n"
    "SKIP only if the lines are not a complete question (greeting, fragment, or noise).\n"
    "A complete question like 'What is Java?' must get Q and A, not SKIP.\n"
    "Then answer in 1 or 2 short sentences with very simple words.\n"
    "Q: <reconstructed question>\n"
    "A: <1-2 simple sentences>"
)


def qa_prompt(ext_lines: list[str]) -> str:
    lines = "\n".join(f"- {line}" for line in ext_lines if line.strip())
    return f"{QA_INSTRUCTIONS}\n\nInterviewer lines:\n{lines or '(none)'}\n"
