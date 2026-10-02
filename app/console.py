import sys
import threading

DIM = "\033[2m"
CYAN = "\033[36m"
YELLOW = "\033[1;33m"
GREEN = "\033[1;32m"
RESET = "\033[0m"

io_lock = threading.Lock()
_stream_open = False


def paint(text: str, style: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"{style}{text}{RESET}"


def safe_print(text: str = "", style: str = "", *, end: str = "\n") -> None:
    """Print without tearing a live answer line when transcript also writes."""
    global _stream_open
    with io_lock:
        if _stream_open and end == "\n":
            print(flush=True)
            _stream_open = False
        shown = paint(text, style) if style else text
        print(shown, end=end, flush=True)
        if end != "\n":
            _stream_open = True
