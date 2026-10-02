from __future__ import annotations

import logging
import os
import sys
import warnings
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path


from events import BUS
from ai.memory import ProfileIndex, SessionMemory
from metrics import LatencyTracker
from runtime import Runtime
from settings import resolve_env_path
from ai.questions import QuestionExtractor, questions_path_for
from console import YELLOW, paint
from audio.speakers import SAMPLE_RATE, OnlineSpeakerTracker, SpeechVad, ensure_models, tracking_enabled
from audio.transcripts import TranscriptStore
from audio.capture import resolve_device, default_mic_index, auto_external_index, print_devices, device_label
from audio.pipeline import AudioPipeline
from ui.lifecycle import start_overlay, close_overlay
from audio.stt import WebSocketSttWorker, SttWorker, Transcriber, resolve_backend, resolve_model_name

warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL.*")

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


def make_extractor(transcribe_path: str, metrics: LatencyTracker, app_dir: Path) -> QuestionExtractor:
    url = (os.environ.get("FUNCTION_URL") or "").strip()
    key = (os.environ.get("INFERENCE_API_KEY") or "").strip() or "1234"
    model = (os.environ.get("LLM_MODEL") or "nova-pro").strip()
    fast = (os.environ.get("LLM_FAST_MODEL") or "").strip()
    path = questions_path_for(transcribe_path)
    n = int(os.environ.get("QA_HISTORY") or "6")
    profile_dir = Path(os.environ.get("PROFILE_DIR") or (app_dir / "profile"))
    return QuestionExtractor(
        function_url=url,
        api_key=key,
        model=model,
        fast_model=fast,
        out_path=path,
        window_sec=float(os.environ.get("QA_WINDOW_SEC") or "45"),
        interval_sec=float(os.environ.get("QA_INTERVAL_SEC") or "1"),
        target_speakers=tuple(s.strip().upper() for s in (os.environ.get("QA_TARGET_SPEAKERS") or "EXT").split(",") if s.strip()),
        metrics=metrics,
        memory=SessionMemory(n),
        profile=ProfileIndex(profile_dir),
    )


def data_dir(app_dir: Path) -> Path:
    raw = (os.environ.get("COPILOT_DATA_DIR") or "").strip()
    if raw:
        path = Path(raw)
        path.mkdir(parents=True, exist_ok=True)
        return path
    return app_dir


def run(args):
    if getattr(sys, "frozen", False):
        app_dir = Path(sys.executable).resolve().parent
        bundle = Path(getattr(sys, "_MEIPASS", app_dir))
    else:
        app_dir = Path(__file__).resolve().parent
        bundle = app_dir
    store = data_dir(app_dir)
    env_path = resolve_env_path(store, app_dir) if getattr(sys, "frozen", False) else app_dir.parent / ".env"
    load_dotenv(env_path)
    load_dotenv(store / ".env")
    load_dotenv(app_dir / ".env")
    load_dotenv(bundle / ".env")
    prompt_dir = store / "prompt" if getattr(sys, "frozen", False) else app_dir.parent / "prompt"
    runtime = Runtime(env_path, prompt_dir=prompt_dir)

    if args.list_devices:
        print_devices()
        return

    mic_spec = args.audio_device if args.audio_device is not None else os.environ.get("AUDIO_DEVICE", "")
    ext_spec = (
        args.external_audio_device
        if args.external_audio_device is not None
        else os.environ.get("EXTERNAL_AUDIO_DEVICE", "")
    )

    mic_idx = resolve_device(mic_spec, kind="AUDIO")
    if mic_idx is None:
        mic_idx = default_mic_index()

    ext_idx = resolve_device(ext_spec, kind="EXTERNAL_AUDIO")
    if ext_idx is None:
        ext_idx = auto_external_index(mic_idx)
    if ext_idx == mic_idx:
        ext_idx = None

    sources = [("MIC", mic_idx)]
    if ext_idx is not None:
        sources.append(("EXT", ext_idx))

    ws_url = os.environ.get("STT_WS_URL", "").strip()
    track = tracking_enabled() and not ws_url
    vad_path, emb_path = ensure_models(store / "models", need_embedding=track)
    vads = {label: SpeechVad(vad_path) for label, _ in sources}
    tracker = OnlineSpeakerTracker(emb_path) if track and emb_path is not None else None

    model_name = resolve_model_name()
    backend = resolve_backend(model_name)
    default_log_dir = store / "log" if getattr(sys, "frozen", False) or os.environ.get("COPILOT_DATA_DIR") else app_dir.parent / "log"
    log_dir = os.environ.get("TRANSCRIBE_LOG_DIR", str(default_log_dir))
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_transcribe.txt")

    metrics = LatencyTracker()
    extractor = make_extractor(log_filename, metrics, store)

    logging.getLogger("pywhispercpp").setLevel(logging.ERROR)
    print("Connecting streaming ASR ..." if ws_url else f"Loading Whisper {model_name} ({backend}) ...", flush=True)
    worker = WebSocketSttWorker(ws_url) if ws_url else SttWorker(Transcriber(model_name, backend), tracker)
    chunk = os.environ.get("STT_CHUNK_SEC", "0.4")
    fast = (os.environ.get("LLM_FAST_MODEL") or "").strip()

    print(
        f"Listening  mic={device_label(mic_idx)}"
        + (f"  ext={device_label(ext_idx)}" if ext_idx is not None else "  ext=none")
        + (f"  speakers={'on' if tracker else 'off'}")
        + f"  stt={model_name}/{backend}"
        + f"  chunk={chunk}s"
        + (f"  fast={fast}" if fast else ""),
        flush=True,
    )
    print(f"log   {log_filename}", flush=True)
    print(paint(f"qlog  {extractor.out_path}", YELLOW), flush=True)
    n_notes = len(extractor.profile.chunks) if extractor.profile else 0
    if n_notes:
        print(f"profile  {n_notes} chunks from {os.environ.get('PROFILE_DIR') or (app_dir / 'profile')}", flush=True)

    runtime.transcripts = TranscriptStore(Path(log_filename))
    runtime.extractor = extractor
    runtime.worker = worker
    overlay = overlay_proc = None
    with ExitStack() as cleanup:
        cleanup.callback(runtime.ai.close)
        cleanup.callback(worker.close)
        def stop_ui():
            BUS.publish("readiness", ready=False, text="Stopped")
            try:
                close_overlay(overlay_proc)
            finally:
                if overlay:
                    overlay.close()
        cleanup.callback(stop_ui)
        overlay, overlay_proc = start_overlay(app_dir, args, runtime)
        BUS.publish("readiness", ready=False, text="Warming up recognition…")
        worker.stt.warmup()

        extractor.start()
        worker.start()

        pipeline = AudioPipeline(sources, vads, worker, runtime.transcripts, metrics,
                                 on_final=lambda update: extractor.add_ext(
                                     update.ts, f"[{update.label}] {update.text}",
                                     t_mono=update.seg_end, stt_ms=update.stt_ms))
        pipeline.run()
