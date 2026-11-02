"""Internal child-process entry point used by Dataset Studio's local scheduler."""
from __future__ import annotations

import json
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

from training_framework import RTDETRAdapter, YOLOAdapter, TrainingConfig


PREFIX = "DATASET_STUDIO_EVENT "
ADAPTERS = {"ultralytics-yolo": YOLOAdapter, "ultralytics-rtdetr": RTDETRAdapter}


def emit(name, *args, **fields):
    if fields: args = (*args, fields)
    print(PREFIX + json.dumps({"event": name, "args": args,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds")},
        ensure_ascii=False, default=str), flush=True)


def main():
    if len(sys.argv) != 3:
        print("Expected internal job configuration and control paths.", file=sys.stderr)
        return 2
    worker = None
    stop_requested = threading.Event()
    failure_seen = threading.Event()
    control_done = threading.Event()
    control_path = Path(sys.argv[2])

    def control_reader():
        while not control_done.wait(.1):
            try:
                if not control_path.is_file():
                    continue
                command = json.loads(control_path.read_text(encoding="utf-8"))
                control_path.unlink(missing_ok=True)
                if command.get("type") == "stop":
                    stop_requested.set()
                    if worker is not None:
                        worker.request_stop()
            except (OSError, ValueError, AttributeError):
                continue

    # A job-scoped control file avoids blocking stdin pipe shutdown on Windows.
    control_thread = threading.Thread(target=control_reader, name="dataset-studio-control", daemon=False)
    control_thread.start()
    try:
        config_path = Path(sys.argv[1])
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        emit("bridge_stage", "configuration_loaded", path=str(config_path))
        adapter_cls = ADAPTERS.get(payload.get("adapter_id"))
        if adapter_cls is None:
            raise ValueError("Unsupported local training adapter.")
        data = payload.get("config", {})
        data.setdefault("extra", {}).pop("_isolated_process", None)
        cpu_threads = data.setdefault("extra", {}).pop("_cpu_threads", None)
        data["extra"].pop("_process_timeout_seconds", None)
        if cpu_threads:
            try:
                import torch
                torch.set_num_threads(max(1, int(cpu_threads)))
            except Exception as exc:
                emit("message", f"Could not apply CPU thread limit: {exc}")
        config = TrainingConfig(**data)
        emit("bridge_stage", "training_adapter_created", adapter_id=payload.get("adapter_id"))
        worker = adapter_cls().create_worker(config)
        for signal_name in ("message", "progress", "epoch_progress", "finished_ok", "failed", "lifecycle", "metrics_row"):
            signal = getattr(worker, signal_name, None)
            if signal is not None:
                def forward(*args, name=signal_name):
                    if name == "failed": failure_seen.set()
                    emit(name, *args)
                signal.connect(forward)
        if stop_requested.is_set():
            worker.request_stop()
        worker.run()
        return 1 if failure_seen.is_set() else 0
    except Exception as exc:
        emit("failed", f"{type(exc).__name__}: {exc}")
        return 1
    finally:
        control_done.set()
        control_thread.join(timeout=2)


if __name__ == "__main__":
    raise SystemExit(main())
