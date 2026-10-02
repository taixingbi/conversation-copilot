from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

from events import BUS
from console import YELLOW, paint

def start_overlay(app_dir: Path, args, runtime=None) -> tuple[object | None, subprocess.Popen | None]:
    from ui.server import OverlayServer

    embedded = (os.environ.get("COPILOT_EMBEDDED") or "").strip().lower() in {"1", "true", "yes"}
    env_off = (os.environ.get("OVERLAY") or "1").strip().lower() in {"0", "false", "no", "off"}
    open_window = (not embedded) and (args.overlay or (not args.no_overlay and not env_off))
    try:
        server = OverlayServer(BUS, runtime=runtime)
        server.start()
    except OSError as exc:
        print(f"overlay http failed: {exc}", flush=True)
        return None, None
    print(paint(f"ui    {server.url}", YELLOW), flush=True)
    BUS.publish("status", text=f"overlay {server.url}")
    proc = None
    if open_window:
        proc = _launch_overlay_window(app_dir, server.url)
    return server, proc


def _launch_overlay_window(app_dir: Path, url: str) -> subprocess.Popen | None:
    electron = app_dir / "node_modules" / ".bin" / "electron"
    if electron.is_file():
        env = os.environ.copy()
        env.pop("ELECTRON_RUN_AS_NODE", None)
        return subprocess.Popen(
            [str(electron), str(app_dir), url], cwd=str(app_dir), env=env,
            start_new_session=(os.name == "posix"),
        )
    if sys.platform == "darwin":
        script = Path(__file__).resolve().parent / "overlay.py"
        return subprocess.Popen(
            [sys.executable, str(script), "--url", url], start_new_session=True,
        )
    print("overlay window skipped — run:  npm install && npm start", flush=True)
    return None


def close_overlay(proc) -> None:
    """Stop the owned launcher and its Electron children together."""
    if proc is None:
        return
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGTERM)
        elif proc.poll() is None:
            proc.terminate()
        proc.wait(timeout=2)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            proc.kill()
        proc.wait(timeout=2)
