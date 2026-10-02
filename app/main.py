"""CLI entrypoint; component wiring lives in application.py."""
import argparse
import signal

from application import run


def parse_args():
    p = argparse.ArgumentParser(description="Live transcribe mic + external audio")
    p.add_argument("--list-devices", action="store_true", help="List input devices and exit")
    p.add_argument("--audio-device", default=None, help="Mic device index or name")
    p.add_argument(
        "--external-audio-device",
        default=None,
        help="Loopback/system-audio device index or name",
    )
    p.add_argument("--overlay", action="store_true", help="Open the floating overlay window")
    p.add_argument("--no-overlay", action="store_true", help="Do not open the overlay window")
    return p.parse_args()


def main():
    run(parse_args())


if __name__ == "__main__":
    def stop_script(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_script)
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
