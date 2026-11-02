"""Validated building blocks for Dataset Studio distributed training.

This module deliberately contains data and resource operations only. Remote work
is represented by validated TrainingConfig values and content-addressed files;
there is no facility for executing commands supplied by a peer.
"""
from __future__ import annotations

import hashlib
import json
import os
import base64
import math
import re
import shutil
import tempfile
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import yaml

from training_framework import TrainingConfig


CHUNK_BYTES = 1024 * 1024
SUPPORTED_ARCHITECTURES = {"YOLO": "ultralytics-yolo", "RT-DETR": "ultralytics-rtdetr"}
_SAFE_CONFIG_FIELDS = {"architecture", "model_version", "initial_weights", "dataset_yaml",
    "output_directory", "experiment_name", "device", "epochs", "batch_size", "image_size",
    "learning_rate", "weight_decay", "optimizer", "scheduler", "validate_each_epoch",
    "mosaic", "workers", "patience", "save_period", "precision", "freeze", "resume", "extra"}


def validate_job_spec(payload: dict, *, trusted_submission: bool = False) -> TrainingConfig:
    """Accept only known architectures and TrainingConfig fields."""
    if not isinstance(payload, dict) or set(payload) - {"job_id", "experiment_id", "config", "dataset_id", "priority", "preferred_device"}:
        raise ValueError("Remote job contains unsupported fields.")
    config = payload.get("config")
    if not isinstance(config, dict) or set(config) - _SAFE_CONFIG_FIELDS:
        raise ValueError("Remote training configuration contains unsupported fields.")
    architecture = config.get("architecture")
    if not isinstance(architecture, str) or architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f"Unsupported remote architecture: {architecture!r}.")
    if config.get("extra") is not None and not isinstance(config["extra"], dict):
        raise ValueError("Architecture-specific options must be an object.")
    # Never allow a remote peer to choose an arbitrary process, module, or import.
    extra = dict(config.get("extra") or {})
    forbidden = {key for key in extra if key.startswith("_") or key in {"command", "python", "module"}}
    if forbidden:
        raise ValueError("Remote training options contain reserved internal fields.")
    unsupported_extra = set(extra) - {"plots", "verbose", "save"}
    if unsupported_extra:
        raise ValueError(f"Unsupported remote architecture options: {', '.join(sorted(unsupported_extra))}.")
    try: result = TrainingConfig(**config)
    except (TypeError, ValueError) as exc: raise ValueError(f"Remote training configuration is malformed: {exc}") from exc
    for field_name, minimum, maximum in (("epochs", 1, 100000), ("batch_size", -1, 4096),
                                         ("image_size", 32, 4096), ("workers", 0, 64),
                                         ("save_period", -1, 10000)):
        value = getattr(result, field_name)
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum or (field_name == "batch_size" and value == 0):
            raise ValueError(f"Remote {field_name} must be an integer from {minimum} to {maximum} (batch size may be -1, but not 0).")
    for field_name in ("validate_each_epoch", "resume"):
        if not isinstance(getattr(result, field_name), bool): raise ValueError(f"Remote {field_name} must be a boolean.")
    for field_name in ("architecture", "model_version", "dataset_yaml", "output_directory", "device", "precision", "scheduler"):
        if not isinstance(getattr(result, field_name), str): raise ValueError(f"Remote {field_name} must be text.")
    if result.freeze is not None and (isinstance(result.freeze, bool) or not isinstance(result.freeze, int) or not 0 <= result.freeze <= 1000):
        raise ValueError("Remote freeze value must be an integer from 0 to 1000.")
    for field_name in ("learning_rate", "weight_decay", "mosaic"):
        value = getattr(result, field_name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
            raise ValueError(f"Remote {field_name} must be a finite number.")
    if not isinstance(result.experiment_name, str) or not result.experiment_name.strip() or result.experiment_name in {".", ".."} or re.search(r"[\\/:]", result.experiment_name):
        raise ValueError("Remote experiment name must be a non-empty folder name without path separators.")
    if not isinstance(result.initial_weights, str) or not result.initial_weights.strip():
        raise ValueError("Remote model/checkpoint reference is required.")
    if not trusted_submission and not result.initial_weights.startswith("remote-model:"):
        model_name = result.initial_weights
        if (Path(model_name).name != model_name or "://" in model_name or
                not re.fullmatch(r"[A-Za-z0-9_.-]+", model_name) or Path(model_name).suffix.lower() not in {".pt", ".yaml", ".yml"}):
            raise ValueError("Worker model references must be a transferred model file or a simple .pt/.yaml preset name.")
    if result.learning_rate is not None and result.learning_rate <= 0: raise ValueError("Remote learning rate must be positive.")
    if result.weight_decay is not None and result.weight_decay < 0: raise ValueError("Remote weight decay cannot be negative.")
    if result.mosaic is not None and not 0 <= result.mosaic <= 1: raise ValueError("Remote mosaic setting must be from 0 to 1.")
    if result.architecture == "RT-DETR" and (result.freeze is not None or result.mosaic is not None):
        raise ValueError("RT-DETR does not accept layer freezing or YOLO mosaic settings.")
    if any(not isinstance(value, bool) for value in extra.values()):
        raise ValueError("Remote plots, verbose, and save options must be booleans.")
    if result.architecture == "YOLO" and result.precision not in {"auto", "fp16", "fp32"}:
        raise ValueError("YOLO precision must be auto, fp16, or fp32.")
    if result.architecture == "RT-DETR" and result.precision not in {"fp32", "auto"}:
        raise ValueError("RT-DETR precision must be fp32 or auto.")
    if not 1 <= int(result.epochs) <= 100000 or not 1 <= int(result.batch_size) <= 4096:
        raise ValueError("Epoch or batch size is outside the accepted range.")
    if result.device not in {"auto", "cpu", "cuda"} and not (result.device.startswith("cuda:") and result.device[5:].isdigit()):
        raise ValueError("Preferred device must be auto, cpu, or a CUDA device index.")
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_manifest(dataset_dir: str | Path) -> dict:
    """Create a deterministic identity from paths, sizes, and full file hashes."""
    root = Path(dataset_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset folder does not exist: {root}")
    files = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.is_symlink():
            raise ValueError(f"Datasets with symbolic links cannot be staged remotely: {path}")
        relative = path.relative_to(root).as_posix()
        if relative.startswith(".dataset_studio_recovery/"):
            continue
        files.append({"path": relative, "size": path.stat().st_size, "sha256": _sha256_file(path)})
    if not any(row["path"].lower().endswith((".yaml", ".yml")) for row in files):
        raise ValueError("Dataset manifest must include a YAML dataset configuration.")
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"dataset_id": hashlib.sha256(canonical).hexdigest(), "files": files,
            "total_bytes": sum(row["size"] for row in files)}


def _validate_manifest_identity(manifest: dict) -> dict:
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Dataset manifest must contain at least one file.")
    seen = set()
    for row in files:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise ValueError("Dataset manifest contains an invalid file entry.")
        rel = PurePosixPath(row["path"])
        if rel.is_absolute() or not rel.parts or ".." in rel.parts or "\\" in row["path"] or ":" in rel.parts[0]:
            raise ValueError("Dataset manifest contains an unsafe path.")
        if rel.as_posix() in seen: raise ValueError("Dataset manifest contains duplicate file paths.")
        seen.add(rel.as_posix())
        if isinstance(row.get("size"), bool) or not isinstance(row.get("size"), int) or row["size"] < 0:
            raise ValueError("Dataset manifest contains an invalid file size.")
        digest = row.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("Dataset manifest contains an invalid SHA-256 digest.")
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    dataset_id = hashlib.sha256(canonical).hexdigest()
    if manifest.get("dataset_id") != dataset_id:
        raise ValueError("Dataset ID does not match its manifest contents.")
    if not any(row["path"].lower().endswith((".yaml", ".yml")) for row in files):
        raise ValueError("Dataset manifest must include a YAML dataset configuration.")
    return {**manifest, "dataset_id": dataset_id}


def create_dataset_archive(dataset_dir: str | Path, archive_path: str | Path) -> dict:
    """Write a ZIP archive without reading the whole dataset into memory."""
    root, archive = Path(dataset_dir).resolve(), Path(archive_path)
    manifest = dataset_manifest(root)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as output:
        for item in manifest["files"]:
            output.write(root / PurePosixPath(item["path"]), item["path"])
    manifest["archive_size"] = archive.stat().st_size
    manifest["archive_sha256"] = _sha256_file(archive)
    return manifest


class DatasetCache:
    """Verified, content-addressed worker cache with resumable archive writes."""
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, dataset_id: str) -> Path:
        if len(dataset_id) != 64 or any(ch not in "0123456789abcdef" for ch in dataset_id):
            raise ValueError("Dataset identity must be a SHA-256 digest.")
        return self.root / dataset_id

    def contains(self, dataset_id: str) -> bool:
        target = self.path_for(dataset_id)
        marker = target / ".dataset-studio-manifest.json"
        try:
            data = _validate_manifest_identity(json.loads(marker.read_text(encoding="utf-8")))
            if data.get("dataset_id") != dataset_id: return False
            for item in data.get("files", []):
                path = (target / PurePosixPath(item["path"])).resolve()
                if not path.is_relative_to(target) or not path.is_file(): return False
                if path.stat().st_size != int(item["size"]) or _sha256_file(path) != item["sha256"]: return False
            return True
        except (OSError, ValueError, TypeError):
            return False

    def stage_archive(self, archive_path: str | Path, manifest: dict) -> Path:
        manifest = _validate_manifest_identity(manifest)
        dataset_id = manifest.get("dataset_id")
        target = self.path_for(dataset_id)
        if self.contains(dataset_id):
            return target
        if target.exists():
            raise ValueError("The worker cache entry exists but failed integrity checks. Clear that cache entry before retrying.")
        archive = Path(archive_path)
        if manifest.get("archive_sha256") and _sha256_file(archive) != manifest["archive_sha256"]:
            raise ValueError("Dataset archive hash does not match the controller manifest.")
        expected = {row["path"]: row for row in manifest.get("files", [])}
        staging = Path(tempfile.mkdtemp(prefix=f".{dataset_id[:12]}-", dir=self.root))
        try:
            with zipfile.ZipFile(archive) as source:
                for info in source.infolist():
                    rel = PurePosixPath(info.filename)
                    if (rel.is_absolute() or not rel.parts or ".." in rel.parts or "\\" in info.filename
                            or ":" in rel.parts[0]):
                        raise ValueError("Dataset archive contains an unsafe path.")
                    if info.is_dir(): continue
                    row = expected.get(rel.as_posix())
                    if row is None:
                        raise ValueError(f"Dataset archive contains an unlisted file: {info.filename}")
                    destination = staging.joinpath(*rel.parts)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    digest = hashlib.sha256(); size = 0
                    with source.open(info) as inp, destination.open("wb") as out:
                        for block in iter(lambda: inp.read(CHUNK_BYTES), b""):
                            out.write(block); digest.update(block); size += len(block)
                    if size != int(row["size"]) or digest.hexdigest() != row["sha256"]:
                        raise ValueError(f"Dataset file failed integrity validation: {info.filename}")
            if len(list(p for p in staging.rglob("*") if p.is_file())) != len(expected):
                raise ValueError("Dataset archive is missing files listed in its manifest.")
            (staging / ".dataset-studio-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            try:
                os.replace(staging, target)
            except OSError:
                # Some Windows synced folders reject directory replacement even
                # on the same volume. Publish by copy only when the target is absent.
                if target.exists(): raise
                try: shutil.copytree(staging, target)
                except Exception:
                    if target.exists(): shutil.rmtree(target, ignore_errors=True)
                    raise
                shutil.rmtree(staging)
            return target
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def cleanup(self, dataset_id: str) -> bool:
        target = self.path_for(dataset_id)
        if not target.exists():
            return False
        shutil.rmtree(target)
        return True

    def cached_datasets(self) -> list[dict]:
        rows = []
        for path in self.root.iterdir():
            if path.is_dir() and len(path.name) == 64 and all(ch in "0123456789abcdef" for ch in path.name):
                rows.append({"dataset_id": path.name, "verified": self.contains(path.name),
                    "bytes": sum(item.stat().st_size for item in path.rglob("*") if item.is_file())})
        return rows

    def check_capacity(self, required_bytes: int, safety_bytes: int = 0) -> tuple[bool, str]:
        free = shutil.disk_usage(self.root).free
        needed = max(0, int(required_bytes)) + max(0, int(safety_bytes))
        if free < needed:
            return False, f"Worker cache needs {needed / (1024 ** 3):.2f} GB free; {free / (1024 ** 3):.2f} GB is available."
        return True, f"Worker cache has sufficient storage ({free / (1024 ** 3):.2f} GB free)."


class ResumableFileReceiver:
    """Append verified transfer blocks to a partial file and finalize by hash."""
    def __init__(self, destination: str | Path, expected_size: int, expected_sha256: str):
        self.destination = Path(destination).expanduser().resolve()
        self.partial = self.destination.with_name(self.destination.name + ".partial")
        self.expected_size = int(expected_size)
        self.expected_sha256 = str(expected_sha256).lower()
        if self.expected_size < 0 or len(self.expected_sha256) != 64:
            raise ValueError("Transfer size or SHA-256 digest is invalid.")
        self.partial.parent.mkdir(parents=True, exist_ok=True)

    @property
    def offset(self) -> int:
        return self.partial.stat().st_size if self.partial.exists() else 0

    def append(self, offset: int, block: bytes) -> int:
        if not isinstance(block, bytes) or not block or len(block) > CHUNK_BYTES:
            raise ValueError("Transfer block must contain between 1 byte and 1 MiB.")
        current = self.offset
        if int(offset) != current:
            raise ValueError(f"Transfer offset mismatch: expected {current}, received {offset}.")
        if current + len(block) > self.expected_size:
            raise ValueError("Transfer block exceeds the declared file size.")
        with self.partial.open("ab") as stream:
            stream.write(block); stream.flush(); os.fsync(stream.fileno())
        return current + len(block)

    def finalize(self) -> Path:
        if self.offset != self.expected_size:
            raise ValueError(f"Transfer is incomplete: {self.offset} of {self.expected_size} bytes received.")
        if _sha256_file(self.partial) != self.expected_sha256:
            raise ValueError("Transferred file failed SHA-256 verification.")
        if self.destination.exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {self.destination}")
        os.replace(self.partial, self.destination)
        return self.destination


class RemoteJobBroker:
    """Persistent controller queue and authenticated message handler for workers."""
    def __init__(self, database, archive_root: str | Path):
        self.database = database
        self.archive_root = Path(archive_root).expanduser().resolve()
        self.archive_root.mkdir(parents=True, exist_ok=True)

    def submit(self, config: TrainingConfig | dict, dataset_dir: str | Path, *, worker_id: str | None = None,
               priority: int = 0, experiment_id: str | None = None, ram_mb: int = 0,
               minimum_free_disk_gb: float = 0) -> str:
        cfg = config.to_dict() if isinstance(config, TrainingConfig) else dict(config)
        cfg["device"] = cfg.get("device", "auto")
        config_obj = validate_job_spec({"config": cfg}, trusted_submission=True)
        dataset_root = Path(dataset_dir).expanduser().resolve()
        manifest = dataset_manifest(dataset_root)
        configured_yaml = Path(config_obj.dataset_yaml).expanduser()
        if not configured_yaml.is_absolute(): configured_yaml = (dataset_root / configured_yaml).resolve()
        try: dataset_yaml_rel = configured_yaml.resolve().relative_to(dataset_root).as_posix()
        except ValueError as exc: raise ValueError("Remote training YAML must be inside the selected dataset folder.") from exc
        if dataset_yaml_rel not in {row["path"] for row in manifest["files"]}:
            raise ValueError("Remote training YAML is not included in the selected dataset manifest.")
        job_id = uuid.uuid4().hex
        if not experiment_id:
            experiment_id = uuid.uuid4().hex
        archive_path = self.archive_root / f"{job_id}.zip"
        manifest = create_dataset_archive(dataset_dir, archive_path)
        remote_config = config_obj.to_dict(); model_asset = None; model_dataset_rel = None
        model_text = str(config_obj.initial_weights)
        model_path = Path(model_text).expanduser()
        if not model_path.is_absolute(): model_path = (Path.cwd() / model_path).resolve()
        if model_path.is_file():
            try: model_dataset_rel = model_path.relative_to(dataset_root).as_posix()
            except ValueError:
                if model_path.suffix.lower() not in {".pt", ".yaml", ".yml"}:
                    archive_path.unlink(missing_ok=True)
                    raise ValueError("Remote model files must be .pt or .yaml files.")
                model_asset = {"name": model_path.name, "size": model_path.stat().st_size,
                    "sha256": _sha256_file(model_path), "controller_model": str(model_path)}
                remote_config["initial_weights"] = f"remote-model:{model_asset['sha256']}"
        elif Path(model_text).is_absolute() or any(sep in model_text for sep in ("/", "\\")):
            archive_path.unlink(missing_ok=True)
            raise ValueError(f"Selected model/checkpoint does not exist on the controller: {model_text}")
        estimated_disk_bytes = int(manifest["total_bytes"]) * 2 + int(model_asset["size"] if model_asset else 0) + 1024 ** 3
        estimated_disk_gb = estimated_disk_bytes / (1024 ** 3)
        requirements = {"config": config_obj.to_dict(), "remote_config": remote_config,
            "model_asset": model_asset, "model_dataset_rel": model_dataset_rel,
            "dataset_id": manifest["dataset_id"],
            "dataset_manifest": manifest, "controller_archive": str(archive_path),
            "dataset_yaml_rel": dataset_yaml_rel,
            "requested_worker_id": worker_id, "ram_mb": max(0, int(ram_mb)),
            "minimum_free_disk_gb": max(estimated_disk_gb, float(minimum_free_disk_gb))}
        try:
            if self.database.get_experiment(experiment_id) is None:
                self.database.upsert_experiment({"experiment_id": experiment_id,
                    "experiment_name": config_obj.experiment_name, "architecture": config_obj.architecture,
                    "adapter": SUPPORTED_ARCHITECTURES[config_obj.architecture], "model_version": config_obj.model_version,
                    "initial_weights": config_obj.initial_weights, "training_configuration": config_obj.to_dict(),
                    "dataset": {"yaml": str(config_obj.dataset_yaml), "classes": []}, "status": "queued",
                    "remote_job": {"job_id": job_id, "worker_id": worker_id, "dataset_id": manifest["dataset_id"]}})
            self.database.create_compute_job({"id": job_id, "experiment_id": experiment_id,
                "architecture": config_obj.architecture, "requirements": requirements,
                "preferred_device": config_obj.device, "priority": int(priority), "origin": "remote",
                "dedupe_key": f"remote:{experiment_id}" if experiment_id else None})
        except Exception:
            archive_path.unlink(missing_ok=True)
            raise
        return job_id

    def _worker_active(self, worker_id: str) -> int:
        return sum(job["worker_id"] == worker_id and job["origin"] == "remote"
                   and job["state"] in {"preparing", "running", "validating"}
                   for job in self.database.list_compute_jobs())

    def offer_next(self, worker_id: str):
        worker = self.database.get_worker(worker_id)
        if not worker: return None
        for active_job in self.database.list_compute_jobs():
            if active_job["origin"] == "remote" and active_job["worker_id"] == worker_id and active_job["state"] in {"preparing", "running", "validating"}:
                if active_job.get("requirements", {}).get("cancel_requested"):
                    return {"operation": "cancel", "job_id": active_job["id"]}
        active = self._worker_active(worker_id)
        for job in self.database.list_compute_jobs():
            if job["origin"] != "remote" or job["state"] != "queued": continue
            requirements = job.get("requirements", {})
            requested = requirements.get("requested_worker_id")
            if requested and requested != worker_id: continue
            suitable, _reason = worker_suitability(worker, architecture=job["architecture"],
                device=job["preferred_device"], ram_mb=requirements.get("ram_mb", 0),
                disk_gb=requirements.get("minimum_free_disk_gb", 0), active_jobs=active,
                gpu_memory_mb=requirements.get("gpu_memory_mb", 0))
            if not suitable: continue
            if not self.database.reserve_compute_resources(job["id"], {"device_kind": "remote",
                    "dataset_bytes": requirements["dataset_manifest"]["total_bytes"]}, worker_id):
                continue
            offer = {"operation": "train", "job_id": job["id"], "experiment_id": job.get("experiment_id"),
                    "architecture": job["architecture"], "config": requirements["remote_config"],
                    "dataset_id": requirements["dataset_id"], "dataset_manifest": requirements["dataset_manifest"],
                    "dataset_yaml_rel": requirements["dataset_yaml_rel"],
                    "model_asset": ({key: value for key, value in requirements["model_asset"].items() if key != "controller_model"}
                                    if requirements.get("model_asset") else None),
                    "model_dataset_rel": requirements.get("model_dataset_rel")}
            if len(json.dumps(offer, ensure_ascii=False).encode("utf-8")) > 8 * 1024 * 1024:
                self.database.set_compute_job_state(job["id"], "failed", error="Validated job manifest exceeds the controller message limit.")
                self.database.release_compute_resources(job["id"])
                if job.get("experiment_id"):
                    experiment = self.database.get_experiment(job["experiment_id"])
                    metadata = dict(experiment.get("metadata", {})) if experiment else {}
                    metadata.update(experiment_id=job["experiment_id"], status="failed", error="Validated job manifest exceeds the controller message limit.")
                    self.database.upsert_experiment(metadata)
                return None
            if job.get("experiment_id"):
                experiment = self.database.get_experiment(job["experiment_id"])
                metadata = dict(experiment.get("metadata", {})) if experiment else {}
                remote = dict(metadata.get("remote_job", {})); remote.update(job_id=job["id"], worker_id=worker_id,
                    worker_name=worker.get("name"), worker_hardware=worker.get("hardware", {}),
                    worker_software=worker.get("software", {}))
                metadata.update(experiment_id=job["experiment_id"], remote_job=remote)
                self.database.upsert_experiment(metadata)
            return offer
        return None

    def handle(self, worker_id: str, message: dict):
        """Serve only predefined progress, transfer, and artifact operations."""
        operation = message.get("type")
        job_id = str(message.get("job_id", ""))
        job = self.database.get_compute_job(job_id)
        if not job or job["origin"] != "remote" or job["worker_id"] != worker_id:
            raise PermissionError("Worker is not authorized for this remote job.")
        if operation == "progress":
            updated = self.database.record_remote_job_update(job_id, worker_id,
                int(message.get("sequence", 0)), str(message.get("state", "running")),
                message.get("progress", {}))
            experiment_id = job.get("experiment_id")
            if experiment_id and updated.pop("_remote_update_accepted", False):
                experiment = self.database.get_experiment(experiment_id)
                metadata = dict(experiment.get("metadata", {})) if experiment else {}
                remote = dict(metadata.get("remote_job", {})); remote.update(job_id=job_id, worker_id=worker_id)
                progress_data = message.get("progress", {})
                remote["progress"] = {key: progress_data[key] for key in
                    ("epoch", "total_epochs", "message", "transfer_bytes", "transfer_total_bytes",
                     "model_transfer_bytes", "model_transfer_total", "output_directory")
                    if key in progress_data}
                metadata.update(remote_job=remote, status=updated["state"])
                metadata["start_time"] = metadata.get("start_time") or updated.get("started_at")
                if updated.get("ended_at"):
                    metadata["end_time"] = updated["ended_at"]
                    try:
                        from datetime import datetime
                        start = datetime.fromisoformat(metadata["start_time"]); end = datetime.fromisoformat(updated["ended_at"])
                        metadata["duration_seconds"] = max(0.0, (end - start).total_seconds())
                    except (TypeError, ValueError): pass
                if isinstance(progress_data.get("metrics"), dict) and isinstance(progress_data.get("epoch"), int):
                    rows = list(metadata.get("metrics", []))
                    rows = [row for row in rows if int(row.get("epoch", -1)) != progress_data["epoch"]]
                    rows.append({"epoch": progress_data["epoch"], **progress_data["metrics"]})
                    metadata["metrics"] = rows
                metadata["experiment_id"] = experiment_id
                self.database.upsert_experiment(metadata)
            elif updated.pop("_remote_update_accepted", False):
                pass
            if updated.get("state") in {"completed", "failed", "cancelled"}:
                archive = Path(job.get("requirements", {}).get("controller_archive", "")).resolve()
                if archive.is_relative_to(self.archive_root) and archive.is_file():
                    archive.unlink()
            if not message.get("request_id"): return None
            return {"request_id": message.get("request_id"), "job_id": job_id,
                    "sequence": int(message.get("sequence", 0)), "state": updated["state"]}
        if operation == "dataset_chunk":
            requirements = job.get("requirements", {})
            path = Path(requirements.get("controller_archive", ""))
            offset, size = int(message.get("offset", -1)), int(message.get("size", 0))
            if not path.is_file() or offset < 0 or not 1 <= size <= 512 * 1024:
                raise ValueError("Dataset transfer request is invalid or unavailable.")
            with path.open("rb") as stream:
                stream.seek(offset); block = stream.read(size)
            return {"request_id": message.get("request_id"), "job_id": job_id, "offset": offset, "data": base64.b64encode(block).decode("ascii"),
                    "eof": offset + len(block) >= path.stat().st_size,
                    "archive_size": path.stat().st_size, "archive_sha256": requirements["dataset_manifest"]["archive_sha256"]}
        if operation == "model_chunk":
            model = job.get("requirements", {}).get("model_asset")
            if not model: raise ValueError("This job does not include a separate model file.")
            path = Path(model.get("controller_model", ""))
            offset, size = int(message.get("offset", -1)), int(message.get("size", 0))
            if not path.is_file() or offset < 0 or not 1 <= size <= 512 * 1024:
                raise ValueError("Model transfer request is invalid or unavailable.")
            with path.open("rb") as stream:
                stream.seek(offset); block = stream.read(size)
            return {"request_id": message.get("request_id"), "job_id": job_id, "offset": offset,
                    "data": base64.b64encode(block).decode("ascii"), "size": model["size"], "sha256": model["sha256"]}
        if operation in {"artifact_start", "artifact_chunk", "artifact_finish"}:
            name = str(message.get("name", ""))
            if name not in {"best.pt", "last.pt", "results.csv", "args.yaml"}:
                raise PermissionError("Worker requested an unsupported artifact name.")
            expected_size = int(message.get("size", -1)); expected_hash = str(message.get("sha256", ""))
            if expected_size < 0 or len(expected_hash) != 64:
                raise ValueError("Artifact size or hash is invalid.")
            artifact_dir = self.archive_root / job_id / "artifacts"
            destination = artifact_dir / name
            if operation in {"artifact_start", "artifact_finish"} and destination.is_file():
                if destination.stat().st_size != expected_size or _sha256_file(destination) != expected_hash:
                    raise ValueError("An existing artifact has different content; refusing to overwrite it.")
                if operation == "artifact_start":
                    return {"request_id": message.get("request_id"), "job_id": job_id, "offset": expected_size}
                return {"request_id": message.get("request_id"), "job_id": job_id,
                        "received": str(destination), "sha256": expected_hash}
            receiver = ResumableFileReceiver(destination, expected_size, expected_hash)
            if operation == "artifact_start":
                return {"request_id": message.get("request_id"), "job_id": job_id, "offset": receiver.offset}
            if operation == "artifact_chunk":
                block = base64.b64decode(message.get("data", ""), validate=True)
                offset = receiver.append(int(message.get("offset", -1)), block)
                return {"request_id": message.get("request_id"), "job_id": job_id, "offset": offset}
            path = receiver.finalize()
            if job.get("experiment_id"):
                experiment = self.database.get_experiment(job["experiment_id"])
                metadata = dict(experiment.get("metadata", {})) if experiment else {}
                checkpoints = dict(metadata.get("checkpoints", {}))
                if name in {"best.pt", "last.pt"}:
                    checkpoints[name[:-3]] = str(path)
                config = dict(metadata.get("training_configuration", {}))
                config["output_directory"] = str(artifact_dir)
                metadata["training_configuration"] = config
                metadata.update(experiment_id=job["experiment_id"], checkpoints=checkpoints)
                self.database.upsert_experiment(metadata)
            return {"request_id": message.get("request_id"), "job_id": job_id, "received": str(path), "sha256": expected_hash}
        raise PermissionError("Unsupported remote worker operation.")


class RemoteWorkerAgent:
    """Runs only broker-issued, validated YOLO/RT-DETR jobs on an enrolled worker."""
    def __init__(self, client, events, *, cache_root: str | Path, output_root: str | Path):
        self.client, self.events = client, events
        self.cache = DatasetCache(cache_root)
        self.output_root = Path(output_root).expanduser().resolve(); self.output_root.mkdir(parents=True, exist_ok=True)
        self._active = set(); self._lock = threading.RLock(); self._rpc_condition = threading.Condition()
        self._rpc_results = {}; self._workers = {}; self._cancel_requested = set()
        self.events.remote_message.connect(self._on_message)

    def _on_message(self, message):
        if not isinstance(message, dict): return
        request_id = message.get("request_id")
        if request_id:
            with self._rpc_condition:
                self._rpc_results[request_id] = message
                self._rpc_condition.notify_all()
            return
        if message.get("type") == "command":
            command = message.get("payload")
            if not isinstance(command, dict): return
            if command.get("operation") == "cancel":
                job_id = str(command.get("job_id", ""))
                self._cancel_requested.add(job_id)
                worker = self._workers.get(job_id)
                if worker: worker.request_stop()
                return
            if command.get("operation") != "train": return
            job_id = str(command.get("job_id", ""))
            with self._lock:
                if not job_id or job_id in self._active: return
                self._active.add(job_id)
            threading.Thread(target=self._execute, args=(command,), name=f"remote-training-{job_id[:8]}", daemon=True).start()

    def _rpc(self, payload: dict, timeout: float = 60):
        request_id = uuid.uuid4().hex
        self.client.send_remote_message({**payload, "request_id": request_id})
        deadline = time.monotonic() + timeout
        with self._rpc_condition:
            while request_id not in self._rpc_results:
                remaining = deadline - time.monotonic()
                if remaining <= 0: raise TimeoutError(f"Controller did not answer {payload.get('type')} request.")
                self._rpc_condition.wait(min(remaining, 1.0))
            result = self._rpc_results.pop(request_id)
        if result.get("error"): raise RuntimeError(str(result["error"]))
        return result

    def _send_progress(self, job_id, sequence, state, **progress):
        self.client.send_remote_message({"type": "progress", "job_id": job_id,
            "sequence": sequence, "state": state, "progress": progress})

    def _receive_dataset(self, job, report=None):
        manifest = job["dataset_manifest"]
        if self.cache.contains(job["dataset_id"]): return self.cache.path_for(job["dataset_id"])
        allowed, reason = self.cache.check_capacity(int(manifest.get("total_bytes", 0)) * 2)
        if not allowed: raise OSError(reason)
        archive_size = int(manifest.get("archive_size", 0))
        archive_hash = str(manifest.get("archive_sha256", ""))
        archive_path = self.cache.root / f"{job['job_id']}.archive"
        receiver = ResumableFileReceiver(archive_path, archive_size, archive_hash)
        while receiver.offset < archive_size:
            if job["job_id"] in self._cancel_requested: raise InterruptedError("Remote job was cancelled during dataset transfer.")
            offset = receiver.offset
            result = self._rpc({"type": "dataset_chunk", "job_id": job["job_id"], "offset": offset,
                                "size": min(512 * 1024, archive_size - offset)})
            data = base64.b64decode(result.get("data", ""), validate=True)
            receiver.append(offset, data)
            if report: report("preparing", transfer_bytes=receiver.offset, transfer_total_bytes=archive_size)
        receiver.finalize()
        try: return self.cache.stage_archive(archive_path, manifest)
        finally: archive_path.unlink(missing_ok=True)

    def _receive_model(self, job, staged: Path, report=None) -> str:
        relative = job.get("model_dataset_rel")
        if relative:
            model_path = (staged / PurePosixPath(relative)).resolve()
            if not model_path.is_relative_to(staged.resolve()) or not model_path.is_file():
                raise ValueError("Model checkpoint included in the dataset is missing or unsafe.")
            return str(model_path)
        model = job.get("model_asset")
        if not model:
            path_value = str(job.get("config", {}).get("initial_weights", ""))
            if path_value.startswith("remote-model:"):
                raise ValueError("Remote model transfer metadata is missing.")
            return path_value
        digest = str(model.get("sha256", "")); suffix = Path(str(model.get("name", "model.pt"))).suffix.lower()
        if len(digest) != 64 or suffix not in {".pt", ".yaml", ".yml"}:
            raise ValueError("Remote model reference is invalid.")
        model_cache = self.cache.root / "models"; model_cache.mkdir(exist_ok=True)
        destination = model_cache / f"{digest}{suffix}"
        if destination.is_file() and destination.stat().st_size == int(model.get("size", -1)) and _sha256_file(destination) == digest:
            return str(destination)
        allowed, reason = self.cache.check_capacity(int(model.get("size", 0)))
        if not allowed: raise OSError(reason)
        receiver = ResumableFileReceiver(destination, int(model["size"]), digest)
        while receiver.offset < receiver.expected_size:
            if job["job_id"] in self._cancel_requested: raise InterruptedError("Remote job was cancelled during model transfer.")
            offset = receiver.offset
            response = self._rpc({"type": "model_chunk", "job_id": job["job_id"], "offset": offset,
                                  "size": min(512 * 1024, receiver.expected_size - offset)})
            receiver.append(offset, base64.b64decode(response.get("data", ""), validate=True))
            if report: report("preparing", model_transfer_bytes=receiver.offset, model_transfer_total=receiver.expected_size)
        return str(receiver.finalize())

    @staticmethod
    def _rewrite_dataset_yaml(staged: Path, relative_yaml: str) -> Path:
        source_yaml = (staged / PurePosixPath(relative_yaml)).resolve()
        if not source_yaml.is_relative_to(staged.resolve()) or not source_yaml.is_file():
            raise ValueError("Dataset configuration path is missing or unsafe in the staged cache.")
        payload = yaml.safe_load(source_yaml.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict): raise ValueError("Dataset YAML root must be a mapping.")
        old_root_value = payload.get("path")
        old_root = Path(str(old_root_value)).expanduser() if old_root_value else None
        for split in ("train", "val", "test"):
            value = payload.get(split)
            if value is None: continue
            values = value if isinstance(value, list) else [value]
            converted = []
            for item in values:
                candidate = Path(str(item)).expanduser()
                if candidate.is_absolute():
                    if old_root is None:
                        raise ValueError(f"Dataset YAML {split} path is absolute but has no base path.")
                    try: candidate = candidate.resolve().relative_to(old_root.resolve())
                    except ValueError as exc: raise ValueError(f"Dataset YAML {split} path escapes its dataset root.") from exc
                if ".." in candidate.parts:
                    raise ValueError(f"Dataset YAML {split} path escapes its dataset root.")
                resolved = (staged / candidate).resolve()
                if not resolved.is_relative_to(staged.resolve()) or not resolved.exists():
                    raise ValueError(f"Dataset YAML {split} path is missing from the transferred dataset: {item}")
                converted.append(candidate.as_posix())
            payload[split] = converted if isinstance(value, list) else converted[0]
        payload["path"] = str(staged.resolve())
        # Keep a conventional visible filename: some Windows-backed training
        # adapters and path validators reject dot-prefixed YAML paths.
        runtime_yaml = staged / "dataset_studio_runtime.yaml"
        runtime_yaml.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
        return runtime_yaml

    def _execute(self, job):
        job_id = job["job_id"]; last_sequence = [0]
        def report(state, **progress):
            last_sequence[0] += 1
            self._send_progress(job_id, last_sequence[0], state, **progress)
        try:
            config = validate_job_spec({"job_id": job_id, "experiment_id": job.get("experiment_id"),
                "dataset_id": job.get("dataset_id"), "config": job.get("config")})
            report("preparing", message="Validating job and preparing dataset")
            staged = self._receive_dataset(job, report)
            config.dataset_yaml = str(self._rewrite_dataset_yaml(staged, job["dataset_yaml_rel"]))
            config.initial_weights = self._receive_model(job, staged, report)
            if job_id in self._cancel_requested:
                report("cancelled", message="Cancelled before training started")
                return
            run_root = self.output_root / job_id
            run_root.mkdir(parents=True, exist_ok=False)
            config.output_directory = str(run_root)
            config.experiment_name = "training"
            config.extra = {**config.extra, "_isolated_process": True}
            if config.architecture == "YOLO":
                from training_framework import YOLOAdapter
                adapter = YOLOAdapter()
            else:
                from training_framework import RTDETRAdapter
                adapter = RTDETRAdapter()
            adapter.validate_config(config)
            adapter_id = SUPPORTED_ARCHITECTURES[config.architecture]
            report("running", message="Training started on worker")
            from training_framework import IsolatedTrainingWorker
            worker = IsolatedTrainingWorker(adapter_id, config)
            self._workers[job_id] = worker
            worker.progress.connect(lambda detail: report("running", message=detail))
            worker.epoch_progress.connect(lambda epoch, total, detail: report("running",
                epoch=epoch, total_epochs=total, message=detail))
            worker.metrics_row.connect(lambda epoch, metrics: report("running", epoch=epoch, metrics=metrics))
            worker.lifecycle.connect(lambda state: report(state if state in {"preparing", "running", "validating"} else "running"))
            results = {"path": None, "error": None}
            worker.finished_ok.connect(lambda path: results.update(path=path))
            worker.failed.connect(lambda error: results.update(error=error))
            worker.run()
            if results["error"]: raise RuntimeError(results["error"])
            if not results["path"]: raise RuntimeError("Training subprocess returned no result path.")
            if worker._stop_requested.is_set() or job_id in self._cancel_requested:
                report("cancelled", message="Training stopped by controller request")
                return
            run_dir = Path(results["path"])
            checkpoints = {name: str(run_dir / "weights" / f"{name}.pt") for name in ("best", "last")}
            available = {name: {"path": path, "size": Path(path).stat().st_size,
                "sha256": _sha256_file(Path(path))} for name, path in checkpoints.items() if Path(path).is_file()}
            if not available:
                raise RuntimeError("Training ended without a best.pt or last.pt checkpoint.")
            report("validating", message="Returning verified checkpoints and training files")
            for name, path in (("best.pt", checkpoints["best"]), ("last.pt", checkpoints["last"]),
                               ("results.csv", str(run_dir / "results.csv")), ("args.yaml", str(run_dir / "args.yaml"))):
                if job_id in self._cancel_requested:
                    report("cancelled", message="Artifact transfer stopped by controller request")
                    return
                source = Path(path)
                if not source.is_file(): continue
                size, digest = source.stat().st_size, _sha256_file(source)
                response = self._rpc({"type": "artifact_start", "job_id": job_id, "name": name,
                                      "size": size, "sha256": digest})
                offset = int(response.get("offset", 0))
                with source.open("rb") as stream:
                    stream.seek(offset)
                    while offset < size:
                        if job_id in self._cancel_requested:
                            report("cancelled", message="Artifact transfer stopped by controller request")
                            return
                        block = stream.read(min(512 * 1024, size - offset))
                        response = self._rpc({"type": "artifact_chunk", "job_id": job_id, "name": name,
                            "size": size, "sha256": digest, "offset": offset,
                            "data": base64.b64encode(block).decode("ascii")})
                        offset = int(response.get("offset", offset + len(block)))
                self._rpc({"type": "artifact_finish", "job_id": job_id, "name": name,
                           "size": size, "sha256": digest})
            report("completed", output_directory=str(run_dir), checkpoints=available)
        except InterruptedError as exc:
            report("cancelled", message=str(exc))
        except Exception as exc:
            report("failed", error=f"{type(exc).__name__}: {exc}")
        finally:
            with self._lock:
                self._active.discard(job_id); self._workers.pop(job_id, None); self._cancel_requested.discard(job_id)


def worker_suitability(worker: dict, *, architecture: str, device: str = "auto", ram_mb: int = 0,
                        disk_gb: float = 0, active_jobs: int = 0, gpu_memory_mb: int = 0) -> tuple[bool, str]:
    """Return a user-readable compatibility decision using reported telemetry."""
    if worker.get("status") != "online": return False, "Worker is offline."
    if not worker.get("approved") or not worker.get("enabled"): return False, "Worker is not approved and enabled."
    hardware = worker.get("hardware") or {}; software = worker.get("software") or {}
    supported = hardware.get("supported_training_architectures")
    if not isinstance(supported, list) or not supported:
        return False, "Worker has not reported supported training architectures."
    if architecture not in supported:
        return False, f"Worker does not report support for {architecture}."
    memory = hardware.get("memory") or {}
    if memory.get("available_mb") is not None and memory["available_mb"] < int(ram_mb):
        return False, f"Worker has {memory['available_mb']} MB RAM available; {ram_mb} MB is requested."
    free_disk = (hardware.get("disk") or {}).get("free_gb")
    if free_disk is not None and free_disk < float(disk_gb):
        return False, f"Worker has {free_disk:g} GB disk free; {disk_gb:g} GB is requested."
    if device.startswith("cuda") and not hardware.get("torch_cuda_available"):
        return False, "Worker has no usable CUDA device."
    if device.startswith("cuda"):
        gpus = hardware.get("gpus") or []
        try: index = int(device.split(":", 1)[1]) if ":" in device else 0
        except ValueError: return False, "Requested CUDA device index is invalid."
        if index < 0 or index >= len(gpus): return False, f"Worker does not report CUDA device {index}."
        free_vram = gpus[index].get("memory_free_mb")
        if free_vram is not None and int(free_vram) < int(gpu_memory_mb):
            return False, f"GPU {index} has {free_vram} MB VRAM free; {gpu_memory_mb} MB is requested."
    versions = software or hardware.get("training_framework_versions") or {}
    if not versions.get("ultralytics") or not versions.get("torch"):
        return False, "Worker has not reported PyTorch and Ultralytics versions."
    if active_jobs:
        return False, f"Worker already has {active_jobs} active job(s); conservative scheduling allows one remote job at a time."
    return True, "Online, approved, enabled, and reports required architecture and runtime."

