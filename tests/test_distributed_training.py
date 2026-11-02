import hashlib
import base64
import tempfile
import threading
import time
import unittest
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from distributed_training import (DatasetCache, RemoteJobBroker, RemoteWorkerAgent, ResumableFileReceiver, create_dataset_archive,
    dataset_manifest, validate_job_spec, worker_suitability)
from experiment_database import ExperimentDatabase


class DistributedTrainingPrimitivesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent.parent)
        self.root = Path(self.temp.name)
        self.dataset = self.root / "dataset"
        (self.dataset / "images/train").mkdir(parents=True)
        (self.dataset / "labels/train").mkdir(parents=True)
        cv2.imwrite(str(self.dataset / "images/train/a.jpg"), np.zeros((32, 32, 3), dtype=np.uint8))
        (self.dataset / "labels/train/a.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
        (self.dataset / "classes.txt").write_text("person\n", encoding="utf-8")
        (self.dataset / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/train\nnames: {0: person}\nnc: 1\n", encoding="utf-8")

    def tearDown(self): self.temp.cleanup()

    def test_manifest_identity_uses_content_hashes_and_is_deterministic(self):
        first = dataset_manifest(self.dataset)
        self.assertEqual(first, dataset_manifest(self.dataset))
        (self.dataset / "images/train/a.jpg").write_bytes(b"other-content")
        second = dataset_manifest(self.dataset)
        self.assertNotEqual(first["dataset_id"], second["dataset_id"])

    def test_archive_staging_verifies_files_and_cache_reuse(self):
        archive = self.root / "data.zip"
        manifest = create_dataset_archive(self.dataset, archive)
        cache = DatasetCache(self.root / "worker-cache")
        staged = cache.stage_archive(archive, manifest)
        self.assertTrue(cache.contains(manifest["dataset_id"]))
        self.assertEqual((staged / "images/train/a.jpg").read_bytes(), (self.dataset / "images/train/a.jpg").read_bytes())
        self.assertEqual(cache.stage_archive(archive, manifest), staged)
        self.assertTrue(cache.cleanup(manifest["dataset_id"]))

    def test_cache_detects_corruption_and_explicit_cleanup_removes_only_hash_folder(self):
        archive=self.root / "data.zip"; manifest=create_dataset_archive(self.dataset, archive)
        cache=DatasetCache(self.root / "worker-cache"); staged=cache.stage_archive(archive, manifest)
        (staged / "images/train/a.jpg").write_bytes(b"corrupt")
        self.assertFalse(cache.contains(manifest["dataset_id"]))
        row=cache.cached_datasets()[0]; self.assertFalse(row["verified"])
        with self.assertRaisesRegex(ValueError, "failed integrity"):
            cache.stage_archive(archive, manifest)
        self.assertTrue(cache.cleanup(manifest["dataset_id"]))
        (cache.root / "models").mkdir(); self.assertFalse(cache.cleanup("b" * 64))

    def test_cache_capacity_preflight_reports_insufficient_space(self):
        cache=DatasetCache(self.root / "worker-cache")
        with patch("distributed_training.shutil.disk_usage", return_value=SimpleNamespace(free=10)):
            allowed, reason=cache.check_capacity(11)
        self.assertFalse(allowed); self.assertIn("needs", reason)

    def test_runtime_dataset_yaml_uses_worker_cache_without_mutating_cached_source(self):
        archive = self.root / "data.zip"; manifest = create_dataset_archive(self.dataset, archive)
        cache = DatasetCache(self.root / "worker-cache"); staged = cache.stage_archive(archive, manifest)
        runtime_yaml = RemoteWorkerAgent._rewrite_dataset_yaml(staged, "data.yaml")
        payload = __import__("yaml").safe_load(runtime_yaml.read_text(encoding="utf-8"))
        self.assertEqual(payload["path"], str(staged.resolve()))
        self.assertTrue((staged / payload["train"]).is_dir())
        self.assertEqual(cache.contains(manifest["dataset_id"]), True)
        self.assertEqual((staged / "data.yaml").read_text(encoding="utf-8").splitlines()[0], "path: .")

    def test_corrupt_archive_file_is_rejected_without_publishing_cache(self):
        archive = self.root / "data.zip"
        manifest = create_dataset_archive(self.dataset, archive)
        corrupted = self.root / "corrupt.zip"
        with zipfile.ZipFile(corrupted, "w") as output:
            output.writestr("data.yaml", "path: .\ntrain: images/train\nval: images/train\nnames: [person]\n")
            output.writestr("images/train/a.jpg", b"corrupted")
        manifest["archive_sha256"] = hashlib.sha256(corrupted.read_bytes()).hexdigest()
        cache = DatasetCache(self.root / "worker-cache")
        with self.assertRaisesRegex(ValueError, "integrity"):
            cache.stage_archive(corrupted, manifest)
        self.assertFalse(cache.contains(manifest["dataset_id"]))

    def test_archive_traversal_is_rejected(self):
        archive = self.root / "unsafe.zip"
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr("../outside.txt", "bad")
        cache = DatasetCache(self.root / "worker-cache")
        with self.assertRaisesRegex(ValueError, "unsafe path"):
            cache.stage_archive(archive, dataset_manifest(self.dataset))

    def test_manifest_identity_mismatch_is_rejected(self):
        manifest=dataset_manifest(self.dataset)
        manifest["dataset_id"]="a" * 64
        with self.assertRaisesRegex(ValueError, "does not match"):
            DatasetCache(self.root / "worker-cache").stage_archive(self.root / "missing.zip", manifest)

    def test_resumable_file_transfer_verifies_and_refuses_overwrite(self):
        content = b"checkpoint-content"; path = self.root / "received" / "best.pt"
        receiver = ResumableFileReceiver(path, len(content), hashlib.sha256(content).hexdigest())
        split = 5
        self.assertEqual(receiver.append(0, content[:split]), split)
        resumed = ResumableFileReceiver(path, len(content), hashlib.sha256(content).hexdigest())
        self.assertEqual(resumed.offset, split)
        resumed.append(split, content[split:])
        self.assertEqual(resumed.finalize().read_bytes(), content)
        resumed.append(0, content)
        with self.assertRaises(FileExistsError): resumed.finalize()

    def test_remote_job_schema_rejects_unknown_architecture_and_internal_fields(self):
        config = {"architecture": "YOLO", "model_version": "YOLO26", "initial_weights": "yolo26n.pt",
            "dataset_yaml": "data.yaml", "output_directory": "runs", "extra": {}}
        self.assertEqual(validate_job_spec({"job_id": "j", "config": config}).architecture, "YOLO")
        with self.assertRaisesRegex(ValueError, "Unsupported remote architecture"):
            validate_job_spec({"job_id": "j", "config": {**config, "architecture": "arbitrary"}})
        with self.assertRaisesRegex(ValueError, "reserved internal"):
            validate_job_spec({"job_id": "j", "config": {**config, "extra": {"_isolated_process": True}}})
        with self.assertRaisesRegex(ValueError, "Worker model references"):
            validate_job_spec({"job_id": "j", "config": {**config, "initial_weights": "https://untrusted.example/model.pt"}})
        with self.assertRaisesRegex(ValueError, "Unsupported remote architecture options"):
            validate_job_spec({"job_id": "j", "config": {**config, "extra": {"data": "../../other.yaml"}}})
        with self.assertRaisesRegex(ValueError, "epochs must be an integer"):
            validate_job_spec({"job_id": "j", "config": {**config, "epochs": "one"}})
        with self.assertRaisesRegex(ValueError, "unsupported fields"):
            validate_job_spec({"job_id": "j", "command": "python thing.py", "config": config})

    def test_worker_suitability_explains_incompatibility_and_accepts_compatible_worker(self):
        worker = {"status": "online", "approved": True, "enabled": True,
            "hardware": {"supported_training_architectures": ["YOLO"], "torch_cuda_available": False,
                "memory": {"available_mb": 4096}, "disk": {"free_gb": 20}},
            "software": {"torch": "2.15", "ultralytics": "8.4"}}
        self.assertEqual(worker_suitability(worker, architecture="YOLO", ram_mb=1024)[0], True)
        ok, reason = worker_suitability(worker, architecture="RT-DETR")
        self.assertFalse(ok); self.assertIn("RT-DETR", reason)
        ok, reason = worker_suitability(worker, architecture="YOLO", device="cuda")
        self.assertFalse(ok); self.assertIn("CUDA", reason)

    def test_remote_broker_assigns_once_and_persists_authorized_progress(self):
        database = ExperimentDatabase(self.root / "experiments.sqlite3")
        token = b"registration"; token_hash = hashlib.sha256(token).digest()
        database.add_enrollment_token(token_hash, (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat())
        database.enroll_worker(token_hash, "worker-a", "Worker A", "", "credential",
            {"supported_training_architectures": ["YOLO"], "torch_cuda_available": False,
             "memory": {"available_mb": 8192}, "disk": {"free_gb": 50}},
            {"torch": "2.15", "ultralytics": "8.4"})
        database.set_worker_approved("worker-a", True)
        database.record_worker_heartbeat("worker-a", "session",
            {"supported_training_architectures": ["YOLO"], "torch_cuda_available": False,
             "memory": {"available_mb": 8192}, "disk": {"free_gb": 50}},
            {"torch": "2.15", "ultralytics": "8.4"})
        broker = RemoteJobBroker(database, self.root / "controller-archives")
        config = {"architecture": "YOLO", "model_version": "YOLO26", "initial_weights": "yolo26n.pt",
            "dataset_yaml": str(self.dataset / "data.yaml"), "output_directory": "runs", "epochs": 1}
        job_id = broker.submit(config, self.dataset, priority=3)
        offer = broker.offer_next("worker-a")
        self.assertEqual(offer["job_id"], job_id)
        self.assertEqual(offer["operation"], "train")
        self.assertIsNone(broker.offer_next("worker-a"))
        experiment_id = database.get_compute_job(job_id)["experiment_id"]
        with self.assertRaisesRegex(ValueError, "active job"):
            broker.submit(config, self.dataset, experiment_id=experiment_id)
        response = broker.handle("worker-a", {"type": "dataset_chunk", "job_id": job_id,
            "request_id": "req-1", "offset": 0, "size": 2048})
        self.assertEqual(response["request_id"], "req-1")
        self.assertGreater(len(response["data"]), 0)
        update = broker.handle("worker-a", {"type": "progress", "job_id": job_id, "request_id": "progress-1", "sequence": 1,
            "state": "running", "progress": {"epoch": 1, "total_epochs": 1}})
        self.assertEqual(update["state"], "running")
        self.assertEqual(database.remote_job_updates(job_id)[0]["progress"]["epoch"], 1)
        stale = broker.handle("worker-a", {"type": "progress", "job_id": job_id, "request_id": "progress-stale", "sequence": 1,
            "state": "preparing", "progress": {"message": "stale"}})
        self.assertEqual(stale["state"], "running")
        weight = b"verified-model-checkpoint"; digest = hashlib.sha256(weight).hexdigest()
        broker.handle("worker-a", {"type": "artifact_start", "job_id": job_id, "name": "best.pt",
            "size": len(weight), "sha256": digest})
        broker.handle("worker-a", {"type": "artifact_chunk", "job_id": job_id, "name": "best.pt",
            "size": len(weight), "sha256": digest, "offset": 0, "data": base64.b64encode(weight).decode()})
        received = broker.handle("worker-a", {"type": "artifact_finish", "job_id": job_id, "name": "best.pt",
            "size": len(weight), "sha256": digest})
        self.assertEqual(Path(received["received"]).read_bytes(), weight)
        self.assertEqual(database.get_experiment(job["experiment_id"] if (job := database.get_compute_job(job_id)) else "")["best_checkpoint"], received["received"])
        untrusted = b"bad-artifact"; expected = hashlib.sha256(b"good-artifact").hexdigest()
        broker.handle("worker-a", {"type": "artifact_start", "job_id": job_id, "name": "last.pt",
            "size": len(untrusted), "sha256": expected})
        broker.handle("worker-a", {"type": "artifact_chunk", "job_id": job_id, "name": "last.pt",
            "size": len(untrusted), "sha256": expected, "offset": 0, "data": base64.b64encode(untrusted).decode()})
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            broker.handle("worker-a", {"type": "artifact_finish", "job_id": job_id, "name": "last.pt",
                "size": len(untrusted), "sha256": expected})
        self.assertEqual(database.request_remote_job_cancel(job_id), "requested")
        self.assertEqual(broker.offer_next("worker-a"), {"operation": "cancel", "job_id": job_id})
        broker.handle("worker-a", {"type": "progress", "job_id": job_id, "sequence": 2,
            "state": "cancelled", "progress": {"message": "Stopped at safe checkpoint"}})
        self.assertEqual(database.get_compute_job(job_id)["state"], "cancelled")
        with self.assertRaises(PermissionError):
            broker.handle("other-worker", {"type": "progress", "job_id": job_id, "sequence": 2,
                "state": "completed", "progress": {}})

    def test_remote_worker_agent_runs_validated_adapters_and_returns_artifacts(self):
        from unittest.mock import patch
        from distributed_training import RemoteWorkerAgent
        from training_framework import IsolatedTrainingWorker

        class FakeSignal:
            def __init__(self): self.handlers=[]
            def connect(self, handler): self.handlers.append(handler)
            def emit(self, *args):
                for handler in tuple(self.handlers): handler(*args)
        class FakeIsolatedWorker:
            def __init__(self, adapter_id, config):
                self.adapter_id=adapter_id; self.config=config; self._stop_requested=threading.Event()
                self.progress=FakeSignal(); self.epoch_progress=FakeSignal(); self.metrics_row=FakeSignal()
                self.lifecycle=FakeSignal(); self.finished_ok=FakeSignal(); self.failed=FakeSignal()
            def request_stop(self): self._stop_requested.set()
            def run(self):
                self.lifecycle.emit("running"); self.epoch_progress.emit(1, 1, "loss=0.1")
                run_dir=Path(self.config.output_directory) / self.config.experiment_name; (run_dir / "weights").mkdir(parents=True)
                (run_dir / "weights/best.pt").write_bytes(b"best-model")
                (run_dir / "weights/last.pt").write_bytes(b"last-model")
                (run_dir / "results.csv").write_text("epoch,metrics/mAP50(B)\n1,0.5\n", encoding="utf-8")
                (run_dir / "args.yaml").write_text("epochs: 1\n", encoding="utf-8")
                self.finished_ok.emit(str(run_dir))

        database = ExperimentDatabase(self.root / "remote.sqlite3")
        enrollment_hash = hashlib.sha256(b"worker-enroll").digest()
        database.add_enrollment_token(enrollment_hash, (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat())
        database.enroll_worker(enrollment_hash, "worker-b", "Worker B", "", "secret", {}, {})
        database.set_worker_approved("worker-b", True)
        database.record_worker_heartbeat("worker-b", "session-b", {"supported_training_architectures": ["YOLO", "RT-DETR"],
            "torch_cuda_available": False, "memory": {"available_mb": 8192}, "disk": {"free_gb": 30}},
            {"torch": "2.15", "ultralytics": "8.4"})
        broker=RemoteJobBroker(database, self.root / "controller-archives")
        for architecture, model, precision in (("YOLO", "YOLO26", "auto"), ("RT-DETR", "RT-DETR-ResNet50", "fp32")):
            config={"architecture": architecture, "model_version": model,
                "initial_weights": "yolo26n.pt" if architecture == "YOLO" else "rtdetr-resnet50.yaml",
                "dataset_yaml": str(self.dataset / "data.yaml"), "output_directory": "runs",
                "experiment_name": "train", "device": "cpu", "epochs": 1, "batch_size": 1,
                "image_size": 64, "workers": 0, "precision": precision}
            job_id=broker.submit(config, self.dataset)
            offer=broker.offer_next("worker-b")
            self.assertEqual(offer["job_id"], job_id)
            class FakeEvents:
                remote_message=FakeSignal()
            events=FakeEvents()
            class FakeClient:
                def send_remote_message(_self, message):
                    response=broker.handle("worker-b", message)
                    if response is not None: events.remote_message.emit(response)
            client=FakeClient()
            agent=RemoteWorkerAgent(client, events, cache_root=self.root / f"cache-{architecture}",
                output_root=self.root / f"outputs-{architecture}")
            with patch("training_framework.IsolatedTrainingWorker", FakeIsolatedWorker):
                events.remote_message.emit({"type": "command", "payload": offer})
                deadline=time.monotonic()+15
                while time.monotonic()<deadline and database.get_compute_job(job_id)["state"] not in {"completed", "failed"}:
                    time.sleep(.03)
            record=database.get_compute_job(job_id)
            self.assertEqual(record["state"], "completed", record["error"] + repr(database.remote_job_updates(job_id)) + repr(agent._active))
            experiment=database.get_experiment(record["experiment_id"])
            self.assertTrue(Path(experiment["best_checkpoint"]).is_file())
            self.assertTrue(Path(experiment["last_checkpoint"]).is_file())
            self.assertEqual(experiment["metadata"]["remote_job"]["worker_id"], "worker-b")


if __name__ == "__main__": unittest.main()
