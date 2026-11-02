"""Experiment persistence, model records and architecture-specific evaluation jobs."""
from __future__ import annotations

import json
import math
import statistics
import time
import uuid
from pathlib import Path

import yaml
from PySide6.QtCore import QThread, Signal

from experiment_database import ExperimentDatabase
from hardware import detect_hardware


class ExperimentManager:
    def __init__(self, database: ExperimentDatabase | None = None):
        self.database = database or ExperimentDatabase()

    def record(self, metadata: dict):
        return self.database.upsert_experiment(metadata)

    def import_legacy_metadata(self, roots):
        """Import per-run JSON files written by the Phase 3 TrainingManager."""
        imported = 0
        for root in roots:
            root = Path(root).expanduser()
            if not root.exists():
                continue
            for path in root.rglob("experiments/*.json"):
                try:
                    metadata = json.loads(path.read_text(encoding="utf-8"))
                    if (metadata.get("experiment_id") and not self.database.is_experiment_deleted(metadata["experiment_id"])
                            and self.database.get_experiment(metadata["experiment_id"]) is None):
                        self.database.upsert_experiment(metadata)
                        imported += 1
                except (OSError, ValueError, TypeError, KeyError):
                    continue
        return imported

    def delete_metadata(self, experiment_id: str, confirmed: bool = False):
        if not confirmed:
            raise PermissionError("Deleting experiment metadata requires explicit confirmation.")
        self.database.delete_experiment_metadata(experiment_id)


def extract_evaluation_metrics(results) -> tuple[dict, dict]:
    """Extract values exposed by an evaluator; missing values remain absent."""
    metrics = {}
    raw = getattr(results, "results_dict", None)
    if isinstance(raw, dict):
        for key, value in raw.items():
            try:
                val = float(value)
                if math.isfinite(val):
                    metrics[str(key)] = val
            except (TypeError, ValueError):
                continue
    box = getattr(results, "box", None)
    for key, attr in (("precision", "mp"), ("recall", "mr"), ("map50", "map50"), ("map50_95", "map")):
        if key in metrics or box is None:
            continue
        try:
            value = float(getattr(box, attr))
            if math.isfinite(value): metrics[key] = value
        except (AttributeError, TypeError, ValueError):
            pass
    if "precision" in metrics and "recall" in metrics and metrics["precision"] + metrics["recall"] > 0:
        metrics["f1"] = 2 * metrics["precision"] * metrics["recall"] / (metrics["precision"] + metrics["recall"])
    per_class = {}
    if box is not None:
        names = getattr(results, "names", {}) or {}
        arrays = {"precision": getattr(box, "p", None), "recall": getattr(box, "r", None),
                  "map50": getattr(box, "ap50", None), "map50_95": getattr(box, "maps", None)}
        count = max((len(values) for values in arrays.values() if values is not None and hasattr(values, "__len__")), default=0)
        for cid in range(count):
            name = names.get(cid, str(cid)) if isinstance(names, dict) else (names[cid] if cid < len(names) else str(cid))
            row = {}
            for metric, values in arrays.items():
                try:
                    value = float(values[cid])
                    if math.isfinite(value): row[metric] = value
                except (TypeError, ValueError, IndexError):
                    continue
            if row: per_class[str(name)] = row
    return metrics, per_class


class ModelEvaluationWorker(QThread):
    progress = Signal(str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, *, architecture, checkpoint, dataset_yaml, device="cpu", image_size=640,
                 database: ExperimentDatabase | None = None, experiment_id=None, model_id=None, parent=None):
        super().__init__(parent)
        self.architecture = architecture
        self.checkpoint = str(Path(checkpoint).expanduser().resolve())
        self.dataset_yaml = str(Path(dataset_yaml).expanduser().resolve())
        self.device, self.image_size = str(device), int(image_size)
        self.database, self.experiment_id, self.model_id = database, experiment_id, model_id

    def run(self):
        try:
            if not Path(self.checkpoint).is_file():
                raise FileNotFoundError(f"Model checkpoint is missing: {self.checkpoint}")
            if not Path(self.dataset_yaml).is_file():
                raise FileNotFoundError(f"Evaluation dataset configuration is missing: {self.dataset_yaml}")
            architecture = self.architecture.casefold()
            if architecture in ("yolo", "yolov5", "ultralytics-yolo"):
                from ultralytics import YOLO as Model
            elif architecture in ("rt-detr", "rtdetr", "ultralytics-rtdetr"):
                from ultralytics import RTDETR as Model
            else:
                raise ValueError(f"No evaluation adapter is implemented for {self.architecture}.")
            self.progress.emit("Loading model and evaluating validation split…")
            model = Model(self.checkpoint)
            results = model.val(data=self.dataset_yaml, imgsz=self.image_size,
                                device=None if self.device == "auto" else self.device)
            metrics, per_class = extract_evaluation_metrics(results)
            save_dir = Path(getattr(results, "save_dir", "")) if getattr(results, "save_dir", None) else None
            confusion = str(save_dir / "confusion_matrix.png") if save_dir and (save_dir / "confusion_matrix.png").is_file() else ""
            result = {"metrics": metrics, "per_class": per_class, "confusion_matrix_path": confusion,
                      "save_dir": str(save_dir) if save_dir else ""}
            if self.database:
                result["evaluation_id"] = self.database.save_evaluation(
                    architecture=self.architecture, checkpoint_path=self.checkpoint,
                    dataset_ref=self.dataset_yaml, metrics=metrics, per_class=per_class,
                    confusion_matrix_path=confusion,
                    settings={"device": self.device, "image_size": self.image_size},
                    experiment_id=self.experiment_id, model_id=self.model_id)
            self.completed.emit(result)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class InferenceBenchmarkWorker(QThread):
    progress = Signal(int, int, str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, *, architecture, checkpoint, dataset_yaml, device="cpu", image_size=640,
                 batch_size=1, warmup_iterations=3, measured_iterations=20,
                 database: ExperimentDatabase | None = None, experiment_id=None, model_id=None, parent=None):
        super().__init__(parent)
        self.architecture = architecture
        self.checkpoint = str(Path(checkpoint).expanduser().resolve())
        self.dataset_yaml = str(Path(dataset_yaml).expanduser().resolve())
        self.device, self.image_size, self.batch_size = str(device), int(image_size), int(batch_size)
        self.warmup_iterations, self.measured_iterations = int(warmup_iterations), int(measured_iterations)
        self.database, self.experiment_id, self.model_id = database, experiment_id, model_id

    def run(self):
        try:
            if self.warmup_iterations < 0 or self.measured_iterations < 1 or self.batch_size < 1:
                raise ValueError("Benchmark iterations and batch size are invalid.")
            if not Path(self.checkpoint).is_file():
                raise FileNotFoundError(f"Model checkpoint is missing: {self.checkpoint}")
            yaml_path = Path(self.dataset_yaml)
            if not yaml_path.is_file():
                raise FileNotFoundError(f"Benchmark dataset configuration is missing: {yaml_path}")
            payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
            split = payload.get("val") or payload.get("train")
            split_path = Path(split)
            if not split_path.is_absolute():
                split_path = (Path(payload.get("path", yaml_path.parent)) / split_path).resolve()
            images = sorted(p for p in split_path.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})
            if not images:
                raise ValueError("The benchmark dataset has no readable image paths in its validation/train split.")
            arch = self.architecture.casefold()
            if arch in ("yolo", "yolov5", "ultralytics-yolo"):
                from ultralytics import YOLO as Model
            elif arch in ("rt-detr", "rtdetr", "ultralytics-rtdetr"):
                from ultralytics import RTDETR as Model
            else:
                raise ValueError(f"No benchmark adapter is implemented for {self.architecture}.")
            model = Model(self.checkpoint)
            batch_count = self.measured_iterations + self.warmup_iterations
            selected = [str(images[i % len(images)]) for i in range(batch_count * self.batch_size)]
            times = []
            for index in range(batch_count):
                batch = selected[index * self.batch_size:(index + 1) * self.batch_size]
                start = time.perf_counter()
                model.predict(source=batch, imgsz=self.image_size, device=None if self.device == "auto" else self.device,
                              verbose=False, batch=self.batch_size)
                elapsed = time.perf_counter() - start
                if index >= self.warmup_iterations:
                    times.append(elapsed)
                self.progress.emit(index + 1, batch_count, f"Benchmark batch {index + 1}/{batch_count}")
            batch_latency_ms = statistics.fmean(times) * 1000
            avg_ms = batch_latency_ms / self.batch_size
            fps = self.batch_size * 1000 / batch_latency_ms if batch_latency_ms else None
            hardware = detect_hardware()
            hardware["gpus"] = [gpu.__dict__ for gpu in hardware["gpus"]]
            result = {"device": self.device, "input_size": self.image_size, "batch_size": self.batch_size,
                      "warmup_iterations": self.warmup_iterations, "measured_iterations": self.measured_iterations,
                      "average_latency_ms": avg_ms, "throughput_fps": fps,
                      "average_batch_latency_ms": batch_latency_ms,
                      "settings": {"dataset_yaml": self.dataset_yaml, "hardware_specific": True,
                                   "average_batch_latency_ms": batch_latency_ms, "hardware": hardware}}
            if self.database:
                result["benchmark_id"] = self.database.save_benchmark(
                    checkpoint_path=self.checkpoint, device=self.device, input_size=self.image_size,
                    batch_size=self.batch_size, warmup_iterations=self.warmup_iterations,
                    measured_iterations=self.measured_iterations, average_latency_ms=avg_ms,
                    throughput_fps=fps, settings=result["settings"], experiment_id=self.experiment_id,
                    model_id=self.model_id)
            self.completed.emit(result)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
