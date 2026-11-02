import json
import csv
import threading
import tempfile
import unittest
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication

from training_framework import (AdapterCapabilities, AdapterRegistry, RTDETRAdapter,
    RTDETRWorker, TrainingAdapter, TrainingConfig, TrainingManager, YOLOAdapter)


class TrainingFrameworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent.parent)
        self.root = Path(self.temp.name) / "data"
        (self.root / "images/train").mkdir(parents=True)
        (self.root / "labels/train").mkdir(parents=True)
        cv2.imwrite(str(self.root / "images/train/one.jpg"), np.zeros((64, 64, 3), np.uint8))
        (self.root / "labels/train/one.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
        (self.root / "classes.txt").write_text("person\n", encoding="utf-8")
        (self.root / "data.yaml").write_text(f"path: {self.root.as_posix()}\ntrain: images/train\nval: images/train\nnames:\n  0: person\nnc: 1\n", encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def config(self, **values):
        args = dict(architecture="YOLO", model_version="YOLO26", initial_weights="yolo26n.pt",
                    dataset_yaml=str(self.root / "data.yaml"), output_directory=str(self.root / "runs"),
                    device="cpu", epochs=1, batch_size=1, image_size=64, workers=0, precision="auto")
        args.update(values)
        return TrainingConfig(**args)

    def test_registry_and_adapter_capabilities(self):
        registry = AdapterRegistry(); adapter = YOLOAdapter(); registry.register(adapter)
        self.assertIs(registry.get("ultralytics-yolo"), adapter)
        self.assertIn("resume", adapter.capabilities.supported_parameters)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            registry.get("not-installed")
        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register(YOLOAdapter())

    def test_yolo_configuration_validation_and_device_compatibility(self):
        adapter = YOLOAdapter()
        dataset = adapter.validate_config(self.config())
        self.assertEqual(dataset["names"], ["person"])
        self.assertIn("epochs", adapter.configuration_summary(self.config())["parameters"])
        with self.assertRaisesRegex(ValueError, "CUDA was selected"):
            adapter.validate_config(self.config(device="cuda:0"))
        with self.assertRaisesRegex(ValueError, "FP16 training requires CUDA"):
            adapter.validate_config(self.config(precision="fp16"))

    def test_rtdetr_adapter_uses_genuine_trainer_and_model_specific_parameters(self):
        from ultralytics import __file__ as ultralytics_file
        model_yaml = Path(ultralytics_file).parent / "cfg/models/rt-detr/rtdetr-resnet50.yaml"
        config = self.config(architecture="RT-DETR", model_version="RT-DETR-ResNet50",
                             initial_weights=str(model_yaml), precision="fp32", freeze=None)
        adapter = RTDETRAdapter()
        adapter.validate_config(config)
        worker = adapter.create_worker(config)
        self.assertIsInstance(worker, RTDETRWorker)
        self.assertNotIn("precision", adapter.capabilities.supported_parameters)
        self.assertNotIn("freeze", adapter.capabilities.supported_parameters)
        with self.assertRaisesRegex(ValueError, "Precision must be one of"):
            adapter.validate_config(self.config(architecture="RT-DETR", initial_weights=str(model_yaml), precision="fp16"))

    def test_yolo_worker_lifecycle_metrics_metadata_and_checkpoint_paths(self):
        config = self.config()
        manager = TrainingManager()
        manager.hardware_snapshot = TrainingManager._hardware_metadata()
        states = []
        manager.lifecycle_changed.connect(lambda _run, state: states.append(state))
        worker = manager.start("ultralytics-yolo", config, autostart=False)
        run_dir = Path(config.output_directory) / config.experiment_name
        (run_dir / "weights").mkdir(parents=True)
        (run_dir / "weights/best.pt").write_bytes(b"best")
        (run_dir / "weights/last.pt").write_bytes(b"last")

        class FakeModel:
            callbacks = {}
            def __init__(self, _path): pass
            def add_callback(self, name, callback): self.callbacks[name] = callback
            def train(self, **kwargs):
                self.kwargs = kwargs
                trainer = SimpleNamespace(epoch=0, epochs=kwargs["epochs"], metrics={"metrics/mAP50(B)": .75},
                                          tloss=None, batch_i=0, nb=1, stop=False)
                self.callbacks["on_pretrain_routine_start"](trainer)
                self.callbacks["on_train_start"](trainer)
                self.callbacks["on_train_batch_end"](trainer)
                self.callbacks["on_fit_epoch_end"](trainer)
                return SimpleNamespace(save_dir=str(run_dir))

        with patch("ultralytics.YOLO", FakeModel):
            worker.run()
        self.assertTrue(manager.flush_metadata())
        metadata = json.loads(manager.metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["status"], "completed")
        self.assertEqual(metadata["metrics"][0]["metrics/mAP50(B)"], .75)
        self.assertTrue(Path(metadata["checkpoints"]["last"]).is_file())
        self.assertIn("running", states); self.assertIn("validating", states); self.assertEqual(states[-1], "completed")
        self.assertTrue(manager.metadata_path.is_file())

    def test_training_manager_runs_validation_and_training_off_ui_thread(self):
        config = self.config()
        manager = TrainingManager()
        manager.hardware_snapshot = TrainingManager._hardware_metadata()
        class FakeModel:
            def __init__(self, _path): self.callbacks = {}
            def add_callback(self, name, callback): self.callbacks[name] = callback
            def train(self, **kwargs):
                time.sleep(0.15)
                trainer = SimpleNamespace(epoch=0, epochs=1, metrics={"metrics/mAP50(B)": .2}, tloss=None,
                                          batch_i=0, nb=1, stop=False)
                self.callbacks["on_pretrain_routine_start"](trainer)
                self.callbacks["on_train_start"](trainer)
                self.callbacks["on_fit_epoch_end"](trainer)
                return SimpleNamespace(save_dir=str(Path(kwargs["project"]) / kwargs["name"]))
        started = time.perf_counter()
        with patch("ultralytics.YOLO", FakeModel):
            worker = manager.start("ultralytics-yolo", config, autostart=True)
            elapsed = time.perf_counter() - started
            self.assertTrue(worker.isRunning())
            worker.wait(30000)
        self.assertLess(elapsed, 1.0)
        self.app.processEvents()
        self.assertTrue(manager.flush_metadata())
        metadata = json.loads(manager.metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["status"], "completed")
        self.assertEqual(metadata["dataset"]["classes"], ["person"])

    def test_yolo_resume_passes_ultralytics_resume_mode(self):
        checkpoint = self.root / "last.pt"; checkpoint.write_bytes(b"checkpoint")
        worker = YOLOAdapter().create_worker(self.config(initial_weights=str(checkpoint), resume=True))
        class FakeModel:
            def __init__(self, _path): self.kwargs = None
            def add_callback(self, _name, _callback): pass
            def train(self, **kwargs): self.kwargs = kwargs; return SimpleNamespace(save_dir="resume-run")
        fake = FakeModel(str(checkpoint)); completed = []
        worker.finished_ok.connect(completed.append)
        with patch("ultralytics.YOLO", return_value=fake): worker.run()
        self.assertEqual(fake.kwargs, {"resume": True})
        self.assertEqual(completed, ["resume-run"])

    def test_rtdetr_worker_uses_rtdetr_api_fp32_and_reports_metrics(self):
        from ultralytics import __file__ as ultralytics_file
        model_yaml = Path(ultralytics_file).parent / "cfg/models/rt-detr/rtdetr-resnet50.yaml"
        config = self.config(architecture="RT-DETR", model_version="RT-DETR-ResNet50",
                             initial_weights=str(model_yaml), precision="fp32")
        class FakeModel:
            callbacks = {}; kwargs = None
            def add_callback(self, name, callback): self.callbacks[name] = callback
            def train(self, **kwargs):
                self.kwargs = kwargs
                trainer = SimpleNamespace(epoch=0, epochs=1, metrics={"metrics/mAP50(B)": .5}, stop=False)
                self.callbacks["on_train_start"](trainer)
                self.callbacks["on_train_epoch_start"](trainer)
                self.callbacks["on_fit_epoch_end"](trainer)
                return SimpleNamespace(save_dir="rtdetr-run")
        fake = FakeModel(); metrics = []; completed = []
        worker = RTDETRWorker(config); worker.metrics_row.connect(lambda epoch, row: metrics.append((epoch, row)))
        worker.finished_ok.connect(completed.append)
        with patch("ultralytics.RTDETR", return_value=fake): worker.run()
        self.assertFalse(fake.kwargs["amp"])
        self.assertEqual(fake.kwargs["data"], str(self.root / "data.yaml"))
        self.assertEqual(metrics[0], (1, {"metrics/mAP50(B)": .5}))
        self.assertEqual(completed, ["rtdetr-run"])

    def test_rtdetr_resume_uses_native_resume_argument(self):
        checkpoint = self.root / "last.pt"; checkpoint.write_bytes(b"resume")
        config = self.config(architecture="RT-DETR", initial_weights=str(checkpoint), resume=True, precision="fp32")
        class FakeModel:
            def __init__(self): self.kwargs = None
            def add_callback(self, _name, _callback): pass
            def train(self, **kwargs): self.kwargs = kwargs; return SimpleNamespace(save_dir="resume-rtdetr")
        fake = FakeModel(); worker = RTDETRWorker(config); completed=[]
        worker.finished_ok.connect(completed.append)
        with patch("ultralytics.RTDETR", return_value=fake): worker.run()
        self.assertEqual(fake.kwargs, {"resume": True, "data": str(self.root / "data.yaml")})
        self.assertEqual(completed, ["resume-rtdetr"])

    def test_metadata_metrics_extraction_and_cancel_before_training(self):
        adapter = YOLOAdapter()
        run = self.root / "metrics-run"; run.mkdir()
        with (run / "results.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["epoch", "metrics/mAP50(B)"]); writer.writeheader(); writer.writerow({"epoch": "1", "metrics/mAP50(B)": "0.75"})
        self.assertEqual(adapter.extract_metrics(run), [{"epoch": "1", "metrics/mAP50(B)": "0.75"}])
        manager = TrainingManager(); worker = manager.start("ultralytics-yolo", self.config(), autostart=False)
        worker.request_stop(); worker.run()
        self.assertTrue(manager.flush_metadata())
        metadata = json.loads(manager.metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["status"], "cancelled")
        self.assertIn("cancelled", metadata["lifecycle"])

    def test_rtdetr_cpu_training_smoke_from_installed_architecture_yaml(self):
        from ultralytics import __file__ as ultralytics_file
        model_yaml = Path(ultralytics_file).parent / "cfg/models/rt-detr/rtdetr-resnet50.yaml"
        output = self.root / "cpu-rtdetr-runs"
        config = self.config(architecture="RT-DETR", model_version="RT-DETR-ResNet50",
                             initial_weights=str(model_yaml), output_directory=str(output),
                             precision="fp32", workers=0, batch_size=1, image_size=64,
                             validate_each_epoch=False,
                             extra={"val": False, "plots": False, "verbose": False, "save": True})
        worker = RTDETRAdapter().create_worker(config)
        completed, failed = [], []
        worker.finished_ok.connect(completed.append); worker.failed.connect(failed.append)
        worker.run()
        self.assertFalse(failed, failed[0] if failed else "")
        self.assertEqual(len(completed), 1)
        self.assertTrue((Path(completed[0]) / "weights/last.pt").is_file())

    def test_yolo_cpu_training_smoke_through_isolated_child_and_metadata_manager(self):
        from ultralytics import __file__ as ultralytics_file
        model_yaml = Path(ultralytics_file).parent / "cfg/models/26/yolo26.yaml"
        if not model_yaml.is_file():
            self.skipTest("Installed Ultralytics package does not include the YOLO26 architecture YAML")
        output = self.root / "cpu-yolo-runs"
        config = self.config(initial_weights=str(model_yaml), output_directory=str(output), workers=0,
                             batch_size=1, image_size=64, validate_each_epoch=False,
                             extra={"plots": False, "verbose": False, "_isolated_process": True,
                                    "_cpu_threads": 2, "_process_timeout_seconds": 180})
        manager = TrainingManager(); worker = manager.start("ultralytics-yolo", config, autostart=False)
        deadline = time.monotonic() + 180
        worker.start()
        while worker.isRunning() and time.monotonic() < deadline:
            self.app.processEvents(); time.sleep(.02)
        if worker.isRunning():
            worker.request_stop(); worker.wait(10000)
            self.fail("Isolated CPU YOLO worker exceeded the integration-test deadline.")
        self.app.processEvents()
        self.assertTrue(manager.flush_metadata())
        metadata = json.loads(manager.metadata_path.read_text(encoding="utf-8"))
        run_dir = output / config.experiment_name
        self.assertEqual(metadata["status"], "completed")
        self.assertTrue((run_dir / "weights/last.pt").is_file())
        self.assertTrue(any("child_started" in item for item in metadata["logs"]))
        self.assertTrue(any("child_terminal_event" in item for item in metadata["logs"]))
        self.assertTrue(any("Epoch 1/1" in item for item in metadata["logs"]))
        self.assertTrue(any(str((run_dir / "weights/last.pt").resolve()) in item for item in metadata["logs"]))

    def test_isolated_child_failure_returns_to_parent_and_cleans_config(self):
        from training_framework import IsolatedTrainingWorker
        config = self.config(dataset_yaml=str(self.root / "missing.yaml"),
            extra={"_process_timeout_seconds": 90})
        worker = IsolatedTrainingWorker("ultralytics-yolo", config)
        failures, completions = [], []
        worker.failed.connect(failures.append); worker.finished_ok.connect(completions.append)
        worker.run()
        self.assertFalse(completions)
        self.assertEqual(len(failures), 1, failures)
        self.assertIn("does not exist", failures[0])
        self.assertIn("child exit code 1", failures[0])
        self.assertIsNone(worker._process)

    def test_rtdetr_cpu_training_smoke_through_isolated_child(self):
        from ultralytics import __file__ as ultralytics_file
        from training_framework import IsolatedTrainingWorker
        model_yaml = Path(ultralytics_file).parent / "cfg/models/rt-detr/rtdetr-resnet50.yaml"
        if not model_yaml.is_file(): self.skipTest("Installed Ultralytics package has no RT-DETR ResNet50 YAML")
        config = self.config(architecture="RT-DETR", model_version="RT-DETR-ResNet50",
            initial_weights=str(model_yaml), output_directory=str(self.root / "isolated-rtdetr"),
            precision="fp32", workers=0, batch_size=1, image_size=64, validate_each_epoch=False,
            extra={"val": False, "plots": False, "verbose": False, "save": True,
                   "_process_timeout_seconds": 180})
        worker = IsolatedTrainingWorker("ultralytics-rtdetr", config)
        completed, failures, progress = [], [], []
        worker.finished_ok.connect(completed.append); worker.failed.connect(failures.append)
        worker.progress.connect(progress.append)
        worker.run()
        self.assertFalse(failures, failures)
        self.assertEqual(len(completed), 1)
        self.assertTrue((Path(completed[0]) / "weights/last.pt").is_file())
        self.assertTrue(any("1/1" in item for item in progress))

    def test_isolated_child_safe_stop_and_control_file_cleanup(self):
        from ultralytics import __file__ as ultralytics_file
        from training_framework import IsolatedTrainingWorker
        model_yaml = Path(ultralytics_file).parent / "cfg/models/26/yolo26.yaml"
        if not model_yaml.is_file(): self.skipTest("Installed Ultralytics package does not include the YOLO26 YAML")
        config = self.config(initial_weights=str(model_yaml), epochs=5,
            output_directory=str(self.root / "isolated-cancel"), workers=0, batch_size=1,
            image_size=64, validate_each_epoch=False,
            extra={"plots": False, "verbose": False, "_process_timeout_seconds": 120})
        worker = IsolatedTrainingWorker("ultralytics-yolo", config)
        launched = threading.Event(); failures = []; completions = []
        worker.message.connect(lambda text: (launched.set(), worker.request_stop()) if "child_started" in text else None)
        worker.failed.connect(failures.append); worker.finished_ok.connect(completions.append)
        worker.start(); deadline = time.monotonic() + 120
        while worker.isRunning() and time.monotonic() < deadline:
            self.app.processEvents(); time.sleep(.02)
        if worker.isRunning():
            worker.request_stop(); worker.wait(10000)
            self.fail("Isolated child did not stop within the cancellation-test deadline.")
        self.app.processEvents()
        self.assertTrue(launched.is_set())
        self.assertTrue(completions or failures)
        self.assertIsNone(worker._process)
        self.assertIsNone(worker._control_path)

    def test_isolated_child_timeout_terminates_process_and_cleans_config(self):
        from training_framework import IsolatedTrainingWorker
        config = self.config(extra={"_process_timeout_seconds": 0.001})
        worker = IsolatedTrainingWorker("ultralytics-yolo", config)
        failures = []
        worker.failed.connect(failures.append)
        worker.run()
        self.assertEqual(len(failures), 1)
        self.assertIn("timeout", failures[0].lower())
        self.assertIsNone(worker._process)


if __name__ == "__main__":
    unittest.main()
