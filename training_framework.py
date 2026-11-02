"""Extensible local training adapters with a shared job/configuration contract."""
from __future__ import annotations

import csv
import importlib.metadata
import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml
from PySide6.QtCore import QObject, Signal, QThread

from hardware import detect_hardware
from yolo_trainer import TrainingWorker
from dataset_manager import DatasetManager
from config import app_state_file


LIFECYCLE = ("created", "queued", "preparing", "running", "validating", "completed", "failed", "cancelled")


@dataclass(frozen=True)
class AdapterCapabilities:
    adapter_id: str
    architecture: str
    task_types: tuple[str, ...]
    model_versions: tuple[str, ...]
    supported_parameters: frozenset[str]
    supports_pretrained: bool = True
    supports_resume: bool = True
    supports_validation: bool = True
    supports_checkpoints: bool = True
    supports_export: bool = False
    hardware: tuple[str, ...] = ("cpu", "cuda")
    description: str = ""


@dataclass
class TrainingConfig:
    architecture: str
    model_version: str
    initial_weights: str
    dataset_yaml: str
    output_directory: str
    experiment_name: str = "train"
    device: str = "cpu"
    epochs: int = 100
    batch_size: int = 16
    image_size: int = 640
    learning_rate: Optional[float] = None
    weight_decay: Optional[float] = None
    optimizer: Optional[str] = None
    scheduler: str = "default"
    validate_each_epoch: bool = True
    mosaic: Optional[float] = None
    workers: int = 0
    patience: Optional[int] = None
    save_period: int = -1
    precision: str = "fp32"
    freeze: Optional[int] = None
    resume: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def _dataset_info(dataset_yaml: str) -> dict:
    path = Path(dataset_yaml).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Dataset configuration does not exist: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise ValueError(f"Could not read dataset configuration: {exc}") from exc
    names = payload.get("names", [])
    if isinstance(names, dict):
        try:
            names = [str(value) for _key, value in sorted(names.items(), key=lambda item: int(item[0]))]
        except (TypeError, ValueError):
            raise ValueError("Dataset class names must have contiguous numeric IDs.")
    if not isinstance(names, list) or not names or any(not str(name).strip() for name in names):
        raise ValueError("Dataset configuration must define at least one class name.")
    if len({str(name).casefold() for name in names}) != len(names):
        raise ValueError("Dataset class names must be unique.")
    if int(payload.get("nc", len(names))) != len(names):
        raise ValueError("Dataset nc does not match the number of configured class names.")
    train = payload.get("train")
    if not train:
        raise ValueError("Dataset configuration is missing a train split.")
    if not payload.get("val"):
        raise ValueError("Ultralytics detection training requires a validation split path, even when epoch validation is disabled.")
    manager = DatasetManager(); manager.open(path.parent)
    if not manager.info or manager.info.annotation_format != "yolo":
        raise ValueError("The selected training adapters currently require a YOLO detection dataset.")
    if manager.classes != [str(name) for name in names]:
        raise ValueError("Dataset YAML class names differ from the dataset class definition.")
    validation = manager.validate_dataset()
    errors = [issue for issue in validation["issues"] if issue["severity"] == "error"]
    if errors:
        issue = errors[0]
        raise ValueError(f"Dataset validation failed: {issue['message']} ({issue.get('image') or issue.get('annotation')})")
    if not manager.image_records():
        raise ValueError("The selected dataset has no images.")
    return {"path": path, "payload": payload, "names": [str(n) for n in names], "validation": validation}


def _validate_common(config, precision_values):
    if config.epochs < 1 or config.batch_size == 0 or config.batch_size < -1 or config.image_size < 32:
        raise ValueError("Epochs, batch size, or image size is invalid.")
    if config.workers < 0 or config.save_period < -1 or (config.patience is not None and config.patience < 0):
        raise ValueError("Worker count, patience, or checkpoint period is invalid.")
    if config.learning_rate is not None and config.learning_rate <= 0:
        raise ValueError("Initial learning rate must be positive.")
    if config.weight_decay is not None and config.weight_decay < 0:
        raise ValueError("Weight decay cannot be negative.")
    if config.optimizer not in (None, "auto", "SGD", "Adam", "AdamW"):
        raise ValueError("The selected optimizer is not supported by this adapter.")
    if config.scheduler not in ("default", "cosine"):
        raise ValueError("The selected scheduler is not supported by this adapter.")
    if config.precision not in precision_values:
        raise ValueError(f"Precision must be one of: {', '.join(sorted(precision_values))}.")


class TrainingAdapter(ABC):
    """Architecture boundary: all framework-specific behavior stays in adapters."""
    capabilities: AdapterCapabilities

    @abstractmethod
    def validate_config(self, config: TrainingConfig) -> dict:
        """Validate architecture-specific model and dataset constraints."""

    @abstractmethod
    def create_worker(self, config: TrainingConfig) -> QThread:
        """Return a background worker exposing the shared training signals."""

    def configuration_summary(self, config: TrainingConfig) -> dict:
        self.validate_config(config)
        params = {key: value for key, value in config.to_dict().items()
                  if key in self.capabilities.supported_parameters or key in
                  {"architecture", "model_version", "initial_weights", "dataset_yaml", "output_directory", "experiment_name"}}
        return {"adapter": self.capabilities.adapter_id, "capabilities": asdict(self.capabilities), "parameters": params}

    @abstractmethod
    def load_model(self, weights: str):
        """Load this architecture's model through its native library API."""

    def convert_dataset(self, dataset_yaml: str) -> Path:
        """Return a compatible dataset reference or fail; no silent format claims."""
        return Path(dataset_yaml).expanduser().resolve()

    def checkpoint_paths(self, config: TrainingConfig) -> dict:
        root = Path(config.output_directory) / config.experiment_name / "weights"
        return {"best": str(root / "best.pt"), "last": str(root / "last.pt")}

    def extract_metrics(self, run_directory: Path) -> list[dict]:
        path = Path(run_directory) / "results.csv"
        if not path.is_file(): return []
        with path.open("r", encoding="utf-8", newline="") as stream:
            return [{key.strip(): value.strip() for key, value in row.items()}
                    for row in csv.DictReader(stream)]

    def export_model(self, weights: str, **options):
        if not self.capabilities.supports_export:
            raise NotImplementedError(f"{self.capabilities.architecture} adapter does not implement model export.")
        raise NotImplementedError("Native export is not implemented by this adapter yet.")


class CustomPyTorchAdapter(TrainingAdapter):
    """Trusted local extension contract for custom PyTorch training pipelines.

    Implementations are registered as Python objects by the local application
    only. This interface is deliberately not an RPC or remote-code loading
    mechanism; arbitrary user code must never be accepted from worker payloads.
    """
    trusted_local_extension = True

    @abstractmethod
    def build_model(self, config: TrainingConfig): ...

    @abstractmethod
    def build_dataset_loaders(self, config: TrainingConfig): ...

    @abstractmethod
    def build_loss(self, config: TrainingConfig, model): ...

    @abstractmethod
    def build_optimizer(self, config: TrainingConfig, model): ...

    @abstractmethod
    def build_scheduler(self, config: TrainingConfig, optimizer): ...

    @abstractmethod
    def train_epoch(self, config: TrainingConfig, model, loader, loss_fn, optimizer, scheduler): ...

    @abstractmethod
    def validate_epoch(self, config: TrainingConfig, model, loader): ...

    @abstractmethod
    def save_checkpoint(self, config: TrainingConfig, model, optimizer, scheduler, epoch: int, path: Path): ...

    @abstractmethod
    def extract_metrics(self, validation_result) -> dict: ...


class YOLOAdapter(TrainingAdapter):
    capabilities = AdapterCapabilities(
        adapter_id="ultralytics-yolo", architecture="YOLO", task_types=("detect",),
        model_versions=("YOLO26", "YOLO12", "YOLO11", "YOLOv10", "YOLOv9", "YOLOv8", "YOLOv6", "YOLOv5"),
        supported_parameters=frozenset({"epochs", "batch_size", "image_size", "learning_rate", "weight_decay", "optimizer", "scheduler", "validate_each_epoch", "mosaic", "workers", "patience", "save_period", "precision", "freeze", "resume", "device", "plots", "verbose", "save"}),
        supports_export=False, description="Ultralytics YOLO training through the existing TrainingWorker." )

    def load_model(self, weights):
        from ultralytics import YOLO
        return YOLO(weights)

    def convert_dataset(self, dataset_yaml):
        info = _dataset_info(dataset_yaml)
        return info["path"]

    def validate_config(self, config):
        if config.architecture.casefold() != "yolo":
            raise ValueError("YOLO adapter received a different architecture.")
        _validate_common(config, {"auto", "fp32", "fp16"})
        dataset = _dataset_info(config.dataset_yaml)
        weights = Path(config.initial_weights)
        if not str(config.initial_weights).strip():
            raise ValueError("Choose pretrained weights or a model configuration.")
        if config.device.startswith("cuda") and not detect_hardware()["torch_cuda_available"]:
            raise ValueError("CUDA was selected but this PyTorch installation has no CUDA support.")
        if config.precision == "fp16" and not config.device.startswith("cuda"):
            raise ValueError("FP16 training requires CUDA; select FP32 for CPU training.")
        if config.resume and not weights.is_file():
            raise ValueError("Resume weights/checkpoint do not exist.")
        return dataset

    def create_worker(self, config):
        options = dict(config.extra)
        for key, value in (("lr0", config.learning_rate), ("weight_decay", config.weight_decay), ("optimizer", config.optimizer),
                           ("workers", config.workers), ("patience", config.patience),
                           ("save_period", config.save_period), ("freeze", config.freeze)):
            if value is not None and (key != "save_period" or value >= 0):
                options[key] = value
        if config.scheduler != "default": options["cos_lr"] = config.scheduler == "cosine"
        options["val"] = bool(config.validate_each_epoch)
        if config.mosaic is not None: options["mosaic"] = config.mosaic
        if config.precision != "auto": options["amp"] = config.precision == "fp16"
        return TrainingWorker(config.initial_weights, Path(config.dataset_yaml), config.device, config.epochs,
                              config.image_size, config.batch_size, config.output_directory,
                              resume=config.resume, train_options=options, run_name=config.experiment_name)


class RTDETRWorker(QThread):
    """Separate RT-DETR execution path; it does not instantiate the YOLO facade."""
    message = Signal(str); progress = Signal(str); epoch_progress = Signal(int, int, str)
    finished_ok = Signal(str); failed = Signal(str); lifecycle = Signal(str); metrics_row = Signal(int, object)

    def __init__(self, config: TrainingConfig):
        super().__init__(); self.config = config
        self._stop_requested = threading.Event(); self._trainer = None

    def request_stop(self):
        self._stop_requested.set()
        if self._trainer is not None: self._trainer.stop = True

    def run(self):
        try:
            from ultralytics import RTDETR
            self.lifecycle.emit("preparing")
            self.message.emit(f"Loading RT-DETR model: {self.config.initial_weights}")
            model = RTDETR(self.config.initial_weights)
            self.lifecycle.emit("running")
            def on_train_start(trainer):
                self._trainer = trainer
                if self._stop_requested.is_set(): trainer.stop = True
            def on_epoch_end(trainer):
                if self._stop_requested.is_set(): trainer.stop = True
            def on_epoch_start(trainer):
                self.lifecycle.emit("running")
            def on_fit_epoch_end(trainer):
                epoch = int(getattr(trainer, "epoch", 0)) + 1
                total = int(getattr(trainer, "epochs", self.config.epochs))
                row = {str(key): float(value) for key, value in getattr(trainer, "metrics", {}).items()
                       if isinstance(value, (int, float))}
                detail = ", ".join(f"{key}={value:.4f}" for key, value in row.items())
                self.lifecycle.emit("validating"); self.metrics_row.emit(epoch, row)
                self.epoch_progress.emit(epoch, total, detail); self.progress.emit(f"{epoch}/{total} | {detail}")
            model.add_callback("on_train_start", on_train_start)
            model.add_callback("on_train_epoch_start", on_epoch_start)
            model.add_callback("on_train_epoch_end", on_epoch_end)
            model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
            if self._stop_requested.is_set():
                raise RuntimeError("Training cancelled before the trainer started.")
            options = dict(self.config.extra)
            options.update(data=str(self.config.dataset_yaml), epochs=self.config.epochs,
                           imgsz=self.config.image_size, batch=self.config.batch_size,
                           device=None if self.config.device == "auto" else self.config.device, project=str(self.config.output_directory),
                           name=self.config.experiment_name, exist_ok=True,
                           workers=self.config.workers, amp=False, val=bool(self.config.validate_each_epoch))
            if self.config.learning_rate is not None: options["lr0"] = self.config.learning_rate
            if self.config.weight_decay is not None: options["weight_decay"] = self.config.weight_decay
            if self.config.optimizer: options["optimizer"] = self.config.optimizer
            if self.config.scheduler != "default": options["cos_lr"] = self.config.scheduler == "cosine"
            if self.config.patience is not None: options["patience"] = self.config.patience
            if self.config.save_period >= 0: options["save_period"] = self.config.save_period
            if self.config.freeze is not None: options["freeze"] = self.config.freeze
            if self.config.resume:
                self.message.emit("Resuming RT-DETR training from checkpoint state…")
                results = model.train(resume=True, data=str(self.config.dataset_yaml))
            else:
                self.message.emit(f"Training RT-DETR on {self.config.device}; AMP disabled for matching stability.")
                results = model.train(**options)
            self.lifecycle.emit("validating")
            save_dir = getattr(results, "save_dir", None)
            self.finished_ok.emit(str(save_dir) if save_dir else str(Path(self.config.output_directory) / self.config.experiment_name))
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class RTDETRAdapter(TrainingAdapter):
    capabilities = AdapterCapabilities(
        adapter_id="ultralytics-rtdetr", architecture="RT-DETR", task_types=("detect",),
        model_versions=("RT-DETR-L", "RT-DETR-X", "RT-DETR-ResNet50", "RT-DETR-ResNet101"),
        supported_parameters=frozenset({"epochs", "batch_size", "image_size", "learning_rate", "weight_decay", "optimizer", "scheduler", "validate_each_epoch", "workers", "patience", "save_period", "resume", "device", "plots", "verbose", "save"}),
        supports_export=False, description="Ultralytics RT-DETR detection trainer. AMP is disabled for its matching stability caveat." )

    def load_model(self, weights):
        from ultralytics import RTDETR
        return RTDETR(weights)

    def convert_dataset(self, dataset_yaml):
        info = _dataset_info(dataset_yaml)
        return info["path"]

    def validate_config(self, config):
        if config.architecture.casefold() not in ("rt-detr", "rtdetr"):
            raise ValueError("RT-DETR adapter received a different architecture.")
        _validate_common(config, {"auto", "fp32"})
        from ultralytics import RTDETR
        from ultralytics.models.rtdetr.train import RTDETRTrainer
        del RTDETR, RTDETRTrainer  # Verify the installed runtime exposes its model and genuine trainer.
        dataset = _dataset_info(config.dataset_yaml)
        if not str(config.initial_weights).strip(): raise ValueError("Choose RT-DETR weights or architecture YAML.")
        if config.device.startswith("cuda") and not detect_hardware()["torch_cuda_available"]:
            raise ValueError("CUDA was selected but this PyTorch installation has no CUDA support.")
        if config.freeze is not None:
            raise ValueError("Layer freezing is not exposed by the RT-DETR adapter.")
        if config.mosaic is not None:
            raise ValueError("Mosaic augmentation override is not exposed by the RT-DETR adapter; use its native defaults.")
        if config.resume and not Path(config.initial_weights).is_file(): raise ValueError("Resume checkpoint does not exist.")
        return dataset

    def create_worker(self, config):
        return RTDETRWorker(config)


class AdapterRegistry:
    def __init__(self): self._adapters = {}
    def register(self, adapter: TrainingAdapter):
        key = adapter.capabilities.adapter_id
        if key in self._adapters: raise ValueError(f"Adapter already registered: {key}")
        self._adapters[key] = adapter
    def get(self, adapter_id):
        try: return self._adapters[adapter_id]
        except KeyError as exc: raise ValueError(f"Unknown training adapter: {adapter_id}") from exc
    def all(self): return tuple(self._adapters.values())


class _TrainingLaunchWorker(QThread):
    """Validate and run an adapter in one background thread, relaying common signals."""
    message = Signal(str); progress = Signal(str); epoch_progress = Signal(int, int, str)
    finished_ok = Signal(str); failed = Signal(str); lifecycle = Signal(str); metrics_row = Signal(int, object)
    dataset_validated = Signal(object); hardware_detected = Signal(object)

    def __init__(self, adapter, config, hardware_snapshot=None, parent=None):
        super().__init__(parent); self.adapter = adapter; self.config = config
        self.hardware_snapshot = hardware_snapshot
        self._stop_requested = threading.Event(); self.inner_worker = None

    def request_stop(self):
        self._stop_requested.set()
        if self.inner_worker is not None:
            self.inner_worker.request_stop()

    def run(self):
        try:
            self.lifecycle.emit("preparing")
            dataset = self.adapter.validate_config(self.config)
            self.dataset_validated.emit(dataset)
            self.hardware_detected.emit(self.hardware_snapshot or TrainingManager._hardware_metadata())
            if self._stop_requested.is_set():
                self.finished_ok.emit(str(Path(self.config.output_directory) / self.config.experiment_name)); return
            if self.config.extra.get("_isolated_process", False):
                worker = IsolatedTrainingWorker(self.adapter.capabilities.adapter_id, self.config)
            else:
                worker = self.adapter.create_worker(self.config)
            self.inner_worker = worker
            for name in ("message", "progress", "epoch_progress", "finished_ok", "failed", "lifecycle", "metrics_row"):
                if hasattr(worker, name):
                    getattr(worker, name).connect(getattr(self, name).emit)
            if self._stop_requested.is_set(): worker.request_stop()
            worker.run()
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class IsolatedTrainingWorker(QThread):
    """Run the existing architecture adapter in a child process with a JSON event protocol."""
    message = Signal(str); progress = Signal(str); epoch_progress = Signal(int, int, str)
    finished_ok = Signal(str); failed = Signal(str); lifecycle = Signal(str); metrics_row = Signal(int, object)
    _EVENT_PREFIX = "DATASET_STUDIO_EVENT "

    def __init__(self, adapter_id: str, config: TrainingConfig, parent=None):
        super().__init__(parent)
        self.adapter_id, self.config = adapter_id, config
        self._stop_requested = threading.Event(); self._process = None
        self._control_path = None
        self._process_lock = threading.Lock()

    def request_stop(self):
        self._stop_requested.set()
        with self._process_lock:
            process = self._process
            control_path = self._control_path
            if process and control_path:
                try:
                    control_path.write_text('{"type":"stop"}', encoding="utf-8")
                except OSError:
                    pass

    def run(self):
        config_path = control_path = None
        process = None
        reader = None
        terminal = None
        terminal_seen = False
        control_stopped = False
        try:
            state_dir = app_state_file("worker_jobs")
            state_dir.mkdir(parents=True, exist_ok=True)
            config_file = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".json", prefix="job-",
                                                     dir=state_dir, delete=False)
            config_path = Path(config_file.name)
            control_path = config_path.with_suffix(".control.json")
            with self._process_lock: self._control_path = control_path
            json.dump({"adapter_id": self.adapter_id, "config": self.config.to_dict()}, config_file)
            config_file.close()
            self.message.emit(self._diagnostic("configuration_written", config_path=str(config_path)))
            script = Path(__file__).resolve().parent / "compute_training_process.py"
            process = subprocess.Popen([sys.executable, str(script), str(config_path), str(control_path)],
                cwd=str(script.parent), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1)
            with self._process_lock: self._process = process
            self.message.emit(self._diagnostic("child_started", pid=process.pid))
            if self._stop_requested.is_set(): self.request_stop()
            events = queue.Queue()

            def drain_output():
                try:
                    for output_line in process.stdout:
                        events.put(output_line)
                finally:
                    events.put(None)

            reader = threading.Thread(target=drain_output, name="dataset-studio-child-output", daemon=True)
            reader.start()
            timeout_seconds = float(self.config.extra.get("_process_timeout_seconds", 24 * 60 * 60))
            if timeout_seconds <= 0: raise ValueError("Child process timeout must be positive.")
            deadline = time.monotonic() + timeout_seconds
            reader_eof = False
            while True:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Training child exceeded its {timeout_seconds:g} second timeout.")
                try:
                    line = events.get(timeout=min(0.25, max(0.01, deadline - time.monotonic())))
                except queue.Empty:
                    if reader_eof and process.poll() is not None: break
                    continue
                if line is None:
                    reader_eof = True
                    if process.poll() is not None: break
                    continue
                text = line.rstrip("\r\n")
                if text.startswith(self._EVENT_PREFIX):
                    try:
                        event = json.loads(text[len(self._EVENT_PREFIX):])
                        name, args = event.get("event"), event.get("args", [])
                        if name == "message": self.message.emit(str(args[0]))
                        elif name == "progress": self.progress.emit(str(args[0]))
                        elif name == "epoch_progress": self.epoch_progress.emit(int(args[0]), int(args[1]), str(args[2]))
                        elif name == "metrics_row": self.metrics_row.emit(int(args[0]), dict(args[1]))
                        elif name == "lifecycle": self.lifecycle.emit(str(args[0]))
                        elif name == "bridge_stage":
                            self.message.emit(self._diagnostic(str(args[0]), **(args[1] if len(args) > 1 else {})))
                        elif name in ("finished_ok", "failed"):
                            terminal_seen = True
                            terminal = (name, args)
                            self.message.emit(self._diagnostic("child_terminal_event", event=name,
                                result=args[0] if name == "finished_ok" and args else ""))
                        elif name == "control_stopped":
                            control_stopped = True
                    except (ValueError, TypeError, IndexError):
                        self.progress.emit(text)
                elif text:
                    self.progress.emit(text)
            try: code = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
                try: code = process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill(); code = process.wait(timeout=5)
            self.message.emit(self._diagnostic("child_exited", exit_code=code, control_stopped=control_stopped))
            if terminal and terminal[0] == "finished_ok" and code == 0:
                self.finished_ok.emit(str(terminal[1][0]))
            elif terminal and terminal[0] == "failed":
                detail = str(terminal[1][0]) if terminal[1] else "Child training failed."
                self.failed.emit(f"{detail} (child exit code {code})")
            elif terminal and code != 0:
                self.failed.emit(f"Child reported success but exited with code {code}.")
            elif not terminal_seen:
                self.failed.emit(f"Training subprocess exited without a completion event (exit code {code}).")
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            if process and process.poll() is None:
                try:
                    if control_path: control_path.write_text('{"type":"stop"}', encoding="utf-8")
                    process.wait(timeout=5)
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    try: process.terminate(); process.wait(timeout=5)
                    except (OSError, subprocess.TimeoutExpired):
                        try: process.kill(); process.wait(timeout=5)
                        except (OSError, subprocess.TimeoutExpired): pass
            self.failed.emit(detail)
        finally:
            if process and process.poll() is None:
                try: process.terminate(); process.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    try: process.kill(); process.wait(timeout=5)
                    except (OSError, subprocess.TimeoutExpired): pass
            if process:
                for pipe in (process.stdout,):
                    if pipe:
                        try: pipe.close()
                        except (OSError, ValueError): pass
            if reader and reader.is_alive(): reader.join(timeout=1)
            with self._process_lock: self._process = None
            if config_path:
                try: config_path.unlink(missing_ok=True)
                except OSError: pass
            if control_path:
                try: control_path.unlink(missing_ok=True)
                except OSError: pass
            with self._process_lock: self._control_path = None

    @staticmethod
    def _diagnostic(stage, **details):
        stamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        extras = " ".join(f"{key}={value}" for key, value in details.items())
        return f"[isolated-worker {stamp}] {stage}" + (f" {extras}" if extras else "")


class TrainingManager(QObject):
    lifecycle_changed = Signal(str, str)
    log_message = Signal(str)
    metrics_updated = Signal(int, object)
    job_finished = Signal(str, str)

    def __init__(self, registry=None, parent=None):
        super().__init__(parent)
        self.registry = registry or AdapterRegistry()
        if not self.registry.all():
            self.registry.register(YOLOAdapter()); self.registry.register(RTDETRAdapter())
        self.worker = None; self.config = None; self.metadata = None; self.metadata_path = None
        self.state = "created"; self._started_at = None; self._logs = []
        self.hardware_snapshot = None
        self._write_lock = threading.Lock(); self._pending_metadata = []
        self._writer_thread = None; self._writer_idle = threading.Event(); self._writer_idle.set()
        self.metadata_write_error = None
        self.experiment_database = None

    def set_experiment_database(self, database):
        """Connect persistent experiment storage without changing worker APIs."""
        self.experiment_database = database

    def start(self, adapter_id: str, config: TrainingConfig, autostart: bool = True):
        if self.worker and self.worker.isRunning(): raise RuntimeError("A training job is already running.")
        adapter = self.registry.get(adapter_id)
        worker = _TrainingLaunchWorker(adapter, config, self.hardware_snapshot, self)
        self.config, self.worker = config, worker
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", config.experiment_name):
            raise ValueError("Experiment name must be 1–64 letters, digits, dots, dashes or underscores and start with a letter or digit.")
        run_dir = Path(config.output_directory).expanduser().resolve() / config.experiment_name
        experiment_id = uuid.uuid4().hex
        self.metadata_path = run_dir / "experiments" / f"{experiment_id}.json"
        self._started_at = datetime.now(timezone.utc); self._logs = []
        dataset_version = None
        self.metadata = {
            "experiment_id": experiment_id,
            "experiment_name": config.experiment_name, "description": "",
            "adapter": adapter.capabilities.adapter_id, "architecture": config.architecture,
            "model_version": config.model_version, "initial_weights": config.initial_weights,
            "dataset": {"yaml": str(Path(config.dataset_yaml).expanduser().resolve()), "classes": [], "version": dataset_version},
            "class_mapping": [],
            "training_configuration": config.to_dict(), "hardware": None,
            "start_time": self._started_at.isoformat(), "end_time": None, "duration_seconds": None,
            "status": "created", "lifecycle": ["created"], "metrics": [],
            "checkpoints": adapter.checkpoint_paths(config),
            "logs": [], "artifacts": [], "metadata_path": str(self.metadata_path),
            "framework_versions": self._framework_versions(), "error": "",
        }
        self._persist()
        self._transition("queued"); self._transition("preparing")
        worker.message.connect(self._on_log); worker.progress.connect(self._on_log)
        worker.epoch_progress.connect(self._on_epoch)
        if hasattr(worker, "metrics_row"): worker.metrics_row.connect(self._on_metrics)
        if hasattr(worker, "lifecycle"): worker.lifecycle.connect(self._transition_from_worker)
        worker.dataset_validated.connect(self._on_dataset_validated)
        worker.hardware_detected.connect(self._on_hardware_detected)
        worker.finished_ok.connect(self._on_completed); worker.failed.connect(self._on_failed)
        if autostart:
            worker.start()
        return worker

    def _on_dataset_validated(self, dataset):
        self.metadata["dataset"]["classes"] = list(dataset["names"])
        yaml_path = Path(dataset["path"])
        self.metadata["class_mapping"] = self._read_class_mapping(yaml_path)
        merge_meta = yaml_path.parent / "merge_metadata.json"
        if merge_meta.is_file():
            try: self.metadata["dataset"]["version"] = json.loads(merge_meta.read_text(encoding="utf-8")).get("merge_id")
            except (OSError, ValueError): pass
        self._persist()

    def _on_hardware_detected(self, hardware):
        self.metadata["hardware"] = hardware; self._persist()

    def cancel(self):
        if self.worker and self.worker.isRunning():
            self.worker.request_stop()
            self._on_log("Cancellation requested; stopping at the next safe training boundary.")

    def _read_class_mapping(self, dataset_yaml):
        path = Path(dataset_yaml).parent / "merge_metadata.json"
        if path.is_file():
            try: return json.loads(path.read_text(encoding="utf-8")).get("class_mappings", [])
            except (OSError, ValueError): pass
        return []

    @staticmethod
    def _hardware_metadata():
        data = detect_hardware()
        data["gpus"] = [gpu.__dict__ for gpu in data["gpus"]]
        return data

    @staticmethod
    def _framework_versions():
        versions = {"python": __import__("sys").version.split()[0]}
        for package in ("ultralytics", "torch", "PySide6", "opencv-python"):
            try: versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError: pass
        return versions

    def _transition_from_worker(self, state):
        if state in LIFECYCLE and state not in ("created", "queued", "completed", "failed", "cancelled"):
            self._transition(state)

    def _transition(self, state):
        if not self.metadata or self.state in ("completed", "failed", "cancelled"): return
        if state == self.state: return
        self.state = state; self.metadata["status"] = state
        if not self.metadata["lifecycle"] or self.metadata["lifecycle"][-1] != state:
            self.metadata["lifecycle"].append(state)
        self._persist(); self.lifecycle_changed.emit(self.metadata["experiment_id"], state)

    def _on_log(self, message):
        self._logs.append(str(message)); self._logs = self._logs[-1500:]
        if self.metadata is not None: self.metadata["logs"] = list(self._logs)
        self.log_message.emit(str(message))

    def _on_epoch(self, epoch, total, detail):
        self._on_log(f"Epoch {epoch}/{total}: {detail}")
        self._persist()

    def _on_metrics(self, epoch, metrics):
        self.metadata["metrics"].append({"epoch": int(epoch), **dict(metrics)})
        self.metrics_updated.emit(epoch, metrics); self._persist()

    def _finish(self, state, output):
        self.state = state; self.metadata["status"] = state
        ended = datetime.now(timezone.utc)
        self.metadata["end_time"] = ended.isoformat()
        self.metadata["duration_seconds"] = max(0.0, (ended - self._started_at).total_seconds())
        self.metadata["logs"] = list(self._logs)
        if output:
            out = Path(output); self.metadata["artifacts"] = [str(path) for path in out.glob("*") if path.is_file()]
        if not self.metadata["lifecycle"] or self.metadata["lifecycle"][-1] != state:
            self.metadata["lifecycle"].append(state)
        self._persist(); self.lifecycle_changed.emit(self.metadata["experiment_id"], state)
        self.job_finished.emit(state, output)

    def _on_completed(self, output):
        stopped = bool(self.worker and getattr(self.worker, "_stop_requested", threading.Event()).is_set())
        self._finish("cancelled" if stopped else "completed", output)

    def _on_failed(self, message):
        self.metadata["error"] = str(message)
        self._on_log(message); self._finish("failed", "")

    def _persist(self):
        if not self.metadata_path or not self.metadata: return
        if self.experiment_database is not None:
            try:
                self.experiment_database.upsert_experiment(self.metadata)
            except Exception as exc:
                self.metadata_write_error = f"SQLite metadata persistence failed: {exc}"
        snapshot = json.dumps(self.metadata, indent=2, ensure_ascii=False)
        with self._write_lock:
            entry = (Path(self.metadata_path), snapshot)
            if self._pending_metadata and self._pending_metadata[-1][0] == entry[0]:
                self._pending_metadata[-1] = entry
            else:
                self._pending_metadata.append(entry)
            self._writer_idle.clear()
            if self._writer_thread is None or not self._writer_thread.is_alive():
                self._writer_thread = threading.Thread(target=self._write_pending, name="training-metadata-writer", daemon=True)
                self._writer_thread.start()

    def _write_pending(self):
        while True:
            with self._write_lock:
                entry = self._pending_metadata.pop(0) if self._pending_metadata else None
                if entry is None:
                    self._writer_thread = None
                    self._writer_idle.set()
                    return
            metadata_path, snapshot = entry
            try:
                metadata_path.parent.mkdir(parents=True, exist_ok=True)
                temp = metadata_path.with_suffix(".json.tmp")
                temp.write_text(snapshot, encoding="utf-8")
                try: os.replace(temp, metadata_path)
                except OSError:
                    metadata_path.write_text(temp.read_text(encoding="utf-8"), encoding="utf-8")
                    temp.unlink(missing_ok=True)
                self.metadata_write_error = None
            except OSError as exc:
                self.metadata_write_error = str(exc)

    def flush_metadata(self, timeout=10.0):
        """Wait for the latest metadata snapshot; intended for shutdown/tests."""
        return self._writer_idle.wait(timeout)

