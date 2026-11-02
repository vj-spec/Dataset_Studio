import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from experiment_database import ExperimentDatabase, normalize_metrics
from experiment_manager import ExperimentManager, InferenceBenchmarkWorker, ModelEvaluationWorker, extract_evaluation_metrics
from experiment_studio import ExperimentManagerPanel, ModelComparisonPanel, ModelRegistryPanel
from training_framework import TrainingConfig, TrainingManager


def metadata(experiment_id="exp-a", status="completed", arch="YOLO", dataset="data.yaml", metric=None):
    return {
        "experiment_id": experiment_id, "experiment_name": experiment_id, "description": "",
        "adapter": "ultralytics-yolo" if arch == "YOLO" else "ultralytics-rtdetr", "architecture": arch,
        "model_version": "test", "initial_weights": "init.pt",
        "dataset": {"yaml": dataset, "classes": ["person"]}, "class_mapping": [],
        "training_configuration": {"experiment_name": experiment_id, "output_directory": "runs", "device": "cpu", "image_size": 640},
        "hardware": {"torch_cuda_available": False}, "framework_versions": {"torch": "test"},
        "start_time": "2026-11-01T10:00:00+00:00", "end_time": "2026-11-01T10:01:00+00:00",
        "duration_seconds": 60.0, "status": status, "metrics": ([{"epoch": 1, **(metric or {})}] if metric is not None else []),
        "checkpoints": {"best": "missing-best.pt", "last": "missing-last.pt"}, "logs": [], "artifacts": [],
    }


class ExperimentManagerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent.parent)
        self.root = Path(self.tmp.name)
        self.db = ExperimentDatabase(self.root / "experiments.sqlite3")

    def tearDown(self):
        self.tmp.cleanup()

    def test_experiment_creation_retrieval_restart_and_metadata_fields(self):
        manager = ExperimentManager(self.db)
        saved = manager.record(metadata(metric={"metrics/precision(B)": .75, "metrics/recall(B)": .5, "metrics/mAP50(B)": .7}))
        self.assertEqual(saved["id"], "exp-a")
        self.assertEqual(saved["classes"], ["person"])
        restarted = ExperimentDatabase(self.root / "experiments.sqlite3")
        self.assertEqual(restarted.get_experiment("exp-a")["status"], "completed")
        self.assertEqual(len(restarted.metrics("exp-a")), 1)

    def test_metrics_keep_raw_names_and_only_normalize_available_fields(self):
        normalized = normalize_metrics({"epoch": 2, "metrics/precision(B)": "0.8", "metrics/recall(B)": "0.4", "trainer/foo": 12})
        self.assertEqual(normalized["precision"], .8)
        self.assertAlmostEqual(normalized["f1"], 2 * .8 * .4 / 1.2)
        self.assertNotIn("map50", normalized)
        self.assertEqual(normalized["architecture_specific"]["trainer/foo"], 12.0)

    def test_status_transitions_and_epoch_upsert(self):
        data = metadata(status="queued", metric={"loss/train": 1.5})
        self.db.upsert_experiment(data)
        data["status"] = "running"; data["metrics"].append({"epoch": 1, "loss/train": .8})
        self.db.upsert_experiment(data)
        self.assertEqual(self.db.get_experiment("exp-a")["status"], "running")
        self.assertEqual(len(self.db.metrics("exp-a")), 1)
        self.assertEqual(self.db.metrics("exp-a")[0]["raw"]["loss/train"], .8)

    def test_training_failure_status_and_error_persist(self):
        manager = TrainingManager(); manager.set_experiment_database(self.db)
        root = self.root / "dataset"; (root / "images/train").mkdir(parents=True); (root / "labels/train").mkdir(parents=True)
        cv2.imwrite(str(root / "images/train/one.jpg"), np.zeros((64, 64, 3), np.uint8))
        (root / "labels/train/one.txt").write_text("", encoding="utf-8")
        (root / "classes.txt").write_text("person\n", encoding="utf-8")
        (root / "data.yaml").write_text(f"path: {root.as_posix()}\ntrain: images/train\nval: images/train\nnames: [person]\nnc: 1\n", encoding="utf-8")
        config = TrainingConfig(architecture="YOLO", model_version="test", initial_weights="test.yaml",
            dataset_yaml=str(root / "data.yaml"), output_directory=str(self.root / "runs"), experiment_name="failed",
            device="cpu", epochs=1, batch_size=1, image_size=64, workers=0, precision="auto")
        manager.start("ultralytics-yolo", config, autostart=False)
        manager._on_failed("synthetic training failure")
        self.assertTrue(manager.flush_metadata())
        saved = self.db.get_experiment(manager.metadata["experiment_id"])
        self.assertEqual(saved["status"], "failed"); self.assertEqual(saved["error"], "synthetic training failure")

    def test_search_filter_and_delete_metadata_never_deletes_checkpoint(self):
        checkpoint = self.root / "best.pt"; checkpoint.write_bytes(b"model")
        row = metadata(); row["checkpoints"]["best"] = str(checkpoint)
        self.db.upsert_experiment(row)
        self.assertEqual(len(self.db.list_experiments("exp-a", "Completed")), 1)
        with self.assertRaises(PermissionError): ExperimentManager(self.db).delete_metadata("exp-a")
        ExperimentManager(self.db).delete_metadata("exp-a", confirmed=True)
        self.assertIsNone(self.db.get_experiment("exp-a")); self.assertTrue(checkpoint.is_file())

    def test_deleted_legacy_metadata_does_not_reappear_after_import(self):
        legacy_root = self.root / "runs" / "trial" / "experiments"; legacy_root.mkdir(parents=True)
        legacy = metadata(); path = legacy_root / "exp-a.json"; path.write_text(json.dumps(legacy), encoding="utf-8")
        manager = ExperimentManager(self.db)
        self.assertEqual(manager.import_legacy_metadata([self.root / "runs"]), 1)
        manager.delete_metadata("exp-a", confirmed=True)
        self.assertEqual(manager.import_legacy_metadata([self.root / "runs"]), 0)
        self.assertTrue(path.is_file())

    def test_artifact_paths_can_be_missing_without_breaking_history(self):
        row = metadata(); row["artifacts"] = [str(self.root / "moved.pt")]
        self.db.upsert_experiment(row)
        self.assertFalse(Path(self.db.artifacts("exp-a")[0]["path"]).is_file())
        self.assertEqual(self.db.get_experiment("exp-a")["status"], "completed")

    def test_model_registry_version_uniqueness_preferred_and_missing_checkpoint(self):
        checkpoint = self.root / "weights.pt"; checkpoint.write_bytes(b"weights")
        first = self.db.register_model(name="Detector", version="1", architecture="YOLO", checkpoint_path=str(checkpoint), classes=["person"])
        second_path = self.root / "weights2.pt"; second_path.write_bytes(b"weights2")
        second = self.db.register_model(name="Detector", version="2", architecture="RT-DETR", checkpoint_path=str(second_path))
        with self.assertRaises(Exception): self.db.register_model(name="Detector", version="1", architecture="YOLO", checkpoint_path=str(checkpoint))
        self.db.set_preferred_model(second["id"])
        self.assertTrue(self.db.get_model(second["id"])["preferred"])
        self.assertFalse(self.db.get_model(first["id"])["preferred"])
        with self.assertRaises(FileNotFoundError): self.db.register_model(name="bad", version="1", architecture="YOLO", checkpoint_path=str(self.root / "missing.pt"))

    def test_evaluation_and_benchmark_records_persist(self):
        record = self.db.upsert_experiment(metadata())
        eval_id = self.db.save_evaluation(architecture="YOLO", checkpoint_path="best.pt", dataset_ref="val.yaml", metrics={"map50": .8}, experiment_id=record["id"])
        bench_id = self.db.save_benchmark(checkpoint_path="best.pt", device="cpu", input_size=640, batch_size=1, warmup_iterations=2,
                                          measured_iterations=10, average_latency_ms=20, throughput_fps=50, experiment_id=record["id"])
        with self.db._connect() as conn:
            self.assertEqual(conn.execute("SELECT count(*) FROM evaluations WHERE id=?", (eval_id,)).fetchone()[0], 1)
            self.assertEqual(conn.execute("SELECT throughput_fps FROM benchmarks WHERE id=?", (bench_id,)).fetchone()[0], 50)

    def test_evaluation_metric_extraction_does_not_fabricate_values(self):
        metrics, per_class = extract_evaluation_metrics(SimpleNamespace(results_dict={"metrics/mAP50(B)": .6}, box=SimpleNamespace(maps=[.5]), names={0: "person"}))
        self.assertEqual(metrics["metrics/mAP50(B)"], .6)
        self.assertNotIn("precision", metrics)
        self.assertEqual(per_class["person"]["map50_95"], .5)

    def test_evaluation_worker_uses_model_adapter_and_persists_results(self):
        ckpt = self.root / "m.pt"; ckpt.write_bytes(b"stub")
        yaml_path = self.root / "data.yaml"; yaml_path.write_text("path: .\ntrain: images\nval: images\nnames: [person]\nnc: 1\n", encoding="utf-8")
        class Fake:
            def __init__(self, path): self.path = path
            def val(self, **kwargs): return SimpleNamespace(results_dict={"metrics/mAP50(B)": .7}, box=None, names={})
        worker = ModelEvaluationWorker(architecture="YOLO", checkpoint=str(ckpt), dataset_yaml=str(yaml_path), database=self.db)
        complete, failed = [], []; worker.completed.connect(complete.append); worker.failed.connect(failed.append)
        with patch("ultralytics.YOLO", Fake): worker.run()
        self.assertFalse(failed, failed)
        self.assertEqual(complete[0]["metrics"]["metrics/mAP50(B)"], .7)
        with self.db._connect() as conn: self.assertEqual(conn.execute("SELECT count(*) FROM evaluations").fetchone()[0], 1)

    def test_training_manager_records_lifecycle_and_epoch_metrics_to_sqlite(self):
        root = self.root / "dataset"; (root / "images/train").mkdir(parents=True); (root / "labels/train").mkdir(parents=True)
        cv2.imwrite(str(root / "images/train/one.jpg"), np.zeros((64, 64, 3), np.uint8))
        (root / "labels/train/one.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        (root / "classes.txt").write_text("person\n", encoding="utf-8")
        (root / "data.yaml").write_text(f"path: {root.as_posix()}\ntrain: images/train\nval: images/train\nnames: [person]\nnc: 1\n", encoding="utf-8")
        out = self.root / "runs"
        config = TrainingConfig(architecture="YOLO", model_version="test", initial_weights="test.yaml",
            dataset_yaml=str(root / "data.yaml"), output_directory=str(out), experiment_name="trial",
            device="cpu", epochs=1, batch_size=1, image_size=64, workers=0, precision="auto")
        class FakeModel:
            def __init__(self, _path): self.callbacks = {}
            def add_callback(self, key, callback): self.callbacks[key] = callback
            def train(self, **kwargs):
                trainer = SimpleNamespace(epoch=0, epochs=1, metrics={"metrics/mAP50(B)": .5}, tloss=None, batch_i=0, nb=1, stop=False)
                self.callbacks["on_pretrain_routine_start"](trainer); self.callbacks["on_train_start"](trainer)
                self.callbacks["on_fit_epoch_end"](trainer)
                return SimpleNamespace(save_dir=str(out / "trial"))
        manager = TrainingManager(); manager.set_experiment_database(self.db)
        worker = manager.start("ultralytics-yolo", config, autostart=False)
        with patch("ultralytics.YOLO", FakeModel): worker.run()
        self.assertTrue(manager.flush_metadata())
        saved = self.db.get_experiment(manager.metadata["experiment_id"])
        self.assertEqual(saved["status"], "completed")
        self.assertEqual(saved["classes"], ["person"])
        self.assertEqual(self.db.metrics(saved["id"])[0]["raw"]["metrics/mAP50(B)"], .5)

    def test_benchmark_worker_measures_requested_batches_and_persists_settings(self):
        ckpt = self.root / "m.pt"; ckpt.write_bytes(b"stub")
        image_dir = self.root / "images"; image_dir.mkdir()
        cv2.imwrite(str(image_dir / "one.jpg"), np.zeros((16, 16, 3), np.uint8))
        yaml_path = self.root / "data.yaml"; yaml_path.write_text(f"path: {self.root.as_posix()}\ntrain: images\nval: images\nnames: [person]\nnc: 1\n", encoding="utf-8")
        class Fake:
            def __init__(self, _path): self.calls = []
            def predict(self, **kwargs): self.calls.append(kwargs); return []
        fake = Fake("m"); worker = InferenceBenchmarkWorker(architecture="YOLO", checkpoint=str(ckpt), dataset_yaml=str(yaml_path),
            batch_size=2, warmup_iterations=1, measured_iterations=2, database=self.db)
        completed, failed = [], []; worker.completed.connect(completed.append); worker.failed.connect(failed.append)
        with patch("ultralytics.YOLO", return_value=fake): worker.run()
        self.assertFalse(failed, failed)
        self.assertEqual(len(fake.calls), 3)
        self.assertTrue(all(len(call["source"]) == 2 and call["batch"] == 2 for call in fake.calls))
        with self.db._connect() as conn:
            saved = conn.execute("SELECT batch_size,warmup_iterations,measured_iterations FROM benchmarks").fetchone()
        self.assertEqual(tuple(saved), (2, 1, 2))

    def test_ui_panels_construct_against_persistent_database(self):
        class Manager: pass
        self.db.upsert_experiment(metadata())
        other = metadata("exp-b", arch="RT-DETR", dataset="other.yaml", metric={"metrics/mAP50(B)": .5})
        self.db.upsert_experiment(other)
        model_path = self.root / "m.pt"; model_path.write_bytes(b"x")
        self.db.register_model(name="M", version="1", architecture="YOLO", checkpoint_path=str(model_path))
        panels = [ExperimentManagerPanel(self.db, Manager()), ModelComparisonPanel(self.db), ModelRegistryPanel(self.db)]
        self.assertEqual(len(panels), 3)
        comparison = panels[1]
        comparison.runs.item(0).setCheckState(Qt.Checked); comparison.runs.item(1).setCheckState(Qt.Checked); comparison.compare()
        self.assertEqual(comparison.table.rowCount(), 6)
        self.assertTrue("Architectures differ" in comparison.warning.text())
        for panel in panels: panel.close()


if __name__ == "__main__":
    unittest.main()
