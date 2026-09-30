from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.answer import LlmAnswerer, parse_qa
from llm.client import ChatClient

CASES = [
    (["What is...", "different.", "between Python and Java."], ("python", "java")),
    (["What is RAG, and how does it work?"], ("rag",)),
    (["How do you evaluate a RAG system?"], ("rag", "evaluat")),
    (["How do you reduce hallucination in an LLM application?"], ("hallucin",)),
    (["How do you choose an LLM for a production system?"], ("llm", "product")),
    (
        ["How would you design and scale a production AI application?"],
        ("design", "scale", "product"),
    ),
]


def load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def fail(msg: str) -> None:
    print(f"FAIL  {msg}", file=sys.stderr)
    raise SystemExit(1)


def ok(msg: str) -> None:
    print(f"ok    {msg}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test inference using FUNCTION_URL; no AWS profile needed.")
    parser.add_argument("--model", help="Override LLM_MODEL for this test")
    parser.add_argument("--quick", action="store_true", help="Send only one hello request")
    args = parser.parse_args()
    load_dotenv(ROOT.parent.parent / ".env")
    load_dotenv(ROOT / ".env")
    load_dotenv(ROOT.parent / ".env")
    url = (os.environ.get("FUNCTION_URL") or "").strip()
    key = (os.environ.get("INFERENCE_API_KEY") or "").strip() or "1234"
    model = (args.model or os.environ.get("LLM_MODEL") or "nova-pro").strip()

    q, a = parse_qa(
        "Q: What is the difference between Python and Java?\n"
        "A: Python is easier to write. Java is compiled."
    )
    if "python" not in q.lower() or "java" not in q.lower() or not a:
        fail("local parse_qa")
    ok("parse_qa")

    if not url:
        fail("FUNCTION_URL is empty — set it in the project root .env")

    print(f"URL   {url.rstrip('/')}/v1/chat/completions")
    print(f"model {model}")

    client = ChatClient(url, key, model)
    hello = client.chat("Say hello in one short sentence.", max_tokens=64)
    if not hello:
        fail("chat returned empty")
    ok(f"chat  {hello[:120]}")
    if args.quick:
        print("Inference smoke test passed.")
        return

    streamed = "".join(client.chat_stream("Say hi in three words.", max_tokens=32))
    if not streamed.strip():
        fail("chat_stream returned empty")
    ok(f"stream {streamed[:120]}")

    llm = LlmAnswerer(function_url=url, api_key=key, model=model)
    for i, (ext, needles) in enumerate(CASES, 1):
        question, answer = llm.qa_from_ext(ext)
        if not question:
            fail(f"case {i} no Q (SKIP?)  ext={ext}")
        if not answer:
            fail(f"case {i} no A  Q={question}")
        blob = f"{question} {answer}".lower()
        if not any(n in blob for n in needles):
            fail(f"case {i} off-topic  need {needles}  Q={question}")
        ok(f"Q{i}   {question}")
        ok(f"A{i}   {answer}")
    print("LLM smoke test passed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        fail(str(exc))
