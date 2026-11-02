import tempfile
import unittest
import json
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from dataset_manager import DatasetManager
from dataset_browser import DatasetBrowserDialog
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from studio import DrillAnnotationStudio
from yolo_trainer import TrainingWorker
from project_views import ModelLoadWorker


class DatasetOperationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent.parent)
        self.root = Path(self.temp.name) / "data"
        (self.root / "images" / "train").mkdir(parents=True)
        (self.root / "labels" / "train").mkdir(parents=True)
        image = np.zeros((100, 120, 3), dtype=np.uint8)
        for name in ("one", "two"):
            cv2.imwrite(str(self.root / "images" / "train" / f"{name}.jpg"), image)
        (self.root / "labels" / "train" / "one.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n1 0.4 0.4 0.1 0.1\n", encoding="utf-8")
        (self.root / "labels" / "train" / "two.txt").write_text(
            "1 0.6 0.6 0.2 0.2\n", encoding="utf-8")
        (self.root / "classes.txt").write_text("person\nhelmet\n", encoding="utf-8")
        (self.root / "data.yaml").write_text(
            "path: .\ntrain: images/train\nval: images/val\nnames:\n  0: person\n  1: helmet\n", encoding="utf-8")
        self.manager = DatasetManager()
        self.manager.open(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_remove_one_class_preserves_other_boxes_and_undo_restores(self):
        result = self.manager.remove_class_annotations(1)
        self.assertEqual(result, {"images": 2, "annotations": 2, "deleted_images": 0})
        self.assertEqual((self.root / "labels/train/one.txt").read_text(encoding="utf-8"),
                         "0 0.5 0.5 0.2 0.2\n")
        self.manager.undo_last_operation()
        self.assertIn("1 0.4 0.4", (self.root / "labels/train/one.txt").read_text(encoding="utf-8"))

    def test_delete_class_shifts_ids_and_undo_restores_schema(self):
        result = self.manager.delete_class(0)
        self.assertEqual(result["annotations"], 1)
        self.assertEqual(self.manager.classes, ["helmet"])
        self.assertTrue((self.root / "labels/train/one.txt").read_text(encoding="utf-8").startswith("0 "))
        self.manager = DatasetManager(); self.manager.open(self.root)
        self.manager.undo_last_operation()
        self.manager.open(self.root)
        self.assertEqual(self.manager.classes, ["person", "helmet"])
        self.assertTrue((self.root / "labels/train/one.txt").read_text(encoding="utf-8").startswith("0 "))

    def test_replace_class_changes_only_matching_annotations(self):
        result = self.manager.replace_class(0, 1)
        self.assertEqual(result["annotations"], 1)
        self.assertTrue((self.root / "labels/train/one.txt").read_text(encoding="utf-8").startswith("1 "))
        self.assertEqual(len((self.root / "labels/train/two.txt").read_text(encoding="utf-8").splitlines()), 1)

    def test_rename_class_updates_names_and_is_recoverable(self):
        self.manager.rename_class(1, "hard_hat")
        self.assertEqual(self.manager.classes[1], "hard_hat")
        self.assertIn("1: hard_hat", (self.root / "data.yaml").read_text(encoding="utf-8"))
        self.manager = DatasetManager()
        self.manager.open(self.root)  # Recovery is available after reopening the dataset.
        self.manager.undo_last_operation()
        self.manager.open(self.root)
        self.assertEqual(self.manager.classes[1], "helmet")

    def test_class_statistics_counts_images_and_boxes(self):
        stats = self.manager.class_statistics()
        self.assertEqual(stats["images"], 2)
        self.assertEqual(stats["annotations"], 3)
        self.assertEqual(stats["classes"][0]["images"], 1)
        self.assertEqual(stats["classes"][1]["images"], 2)

    def test_delete_images_and_undo_restores_image_and_label(self):
        image = self.root / "images/train/one.jpg"
        label = self.root / "labels/train/one.txt"
        result = self.manager.delete_images([image])
        self.assertEqual(result, {"images": 1, "annotations": 2})
        self.assertFalse(image.exists())
        self.assertFalse(label.exists())
        self.manager = DatasetManager(); self.manager.open(self.root)
        self.manager.undo_last_operation()
        self.assertTrue(image.exists())
        self.assertTrue(label.exists())

    def test_reorder_classes_remaps_ids_and_undo_restores(self):
        self.manager.reorder_classes([1, 0])
        self.assertEqual(self.manager.classes, ["helmet", "person"])
        content = (self.root / "labels/train/one.txt").read_text(encoding="utf-8").splitlines()
        self.assertTrue(content[0].startswith("1 "))
        self.assertTrue(content[1].startswith("0 "))
        self.manager.undo_last_operation()
        self.manager.open(self.root)
        self.assertEqual(self.manager.classes, ["person", "helmet"])

    def test_replace_class_undo_after_restart(self):
        self.manager.replace_class(0, 1)
        self.manager = DatasetManager(); self.manager.open(self.root)
        self.manager.undo_last_operation()
        self.assertTrue((self.root / "labels/train/one.txt").read_text(encoding="utf-8").startswith("0 "))

    def test_multiple_snapshots_are_preserved_and_undo_is_a_stack(self):
        self.manager.rename_class(0, "worker")
        self.manager.replace_class(0, 1)
        recovery = self.root / ".dataset_studio_recovery"
        snapshots = [p for p in recovery.iterdir() if p.is_dir()]
        self.assertEqual(len(snapshots), 2)
        self.manager.undo_last_operation()
        self.assertTrue((self.root / "labels/train/one.txt").read_text(encoding="utf-8").startswith("0 "))
        self.manager.undo_last_operation()
        self.manager.open(self.root)
        self.assertEqual(self.manager.classes[0], "person")

    def test_partial_operation_failure_can_be_undone(self):
        def write_then_fail(updates):
            for index, (_image, label, content) in enumerate(updates):
                label.write_text(content, encoding="utf-8")
                if index == 0:
                    raise OSError("simulated disk failure after first write")
        with patch.object(self.manager, "_write_staged_yolo", side_effect=write_then_fail):
            with self.assertRaises(OSError):
                self.manager.remove_class_annotations(1)
        self.manager = DatasetManager(); self.manager.open(self.root)
        self.manager.undo_last_operation()
        self.assertEqual(len((self.root / "labels/train/one.txt").read_text(encoding="utf-8").splitlines()), 2)
        self.assertEqual(len((self.root / "labels/train/two.txt").read_text(encoding="utf-8").splitlines()), 1)

    def test_remove_class_can_delete_newly_empty_images(self):
        result = self.manager.remove_class_annotations(1, delete_empty_images=[self.root / "images/train/two.jpg"])
        self.assertEqual(result["deleted_images"], 1)
        self.assertFalse((self.root / "images/train/two.jpg").exists())
        self.manager.undo_last_operation()
        self.assertTrue((self.root / "images/train/two.jpg").exists())

    def test_batch_annotation_reassignment_and_deletion_preserve_unselected_boxes(self):
        result = self.manager.edit_annotations(self.root / "images/train/one.jpg", [0], class_id=1)
        self.assertEqual(result["annotations"], 1)
        content = (self.root / "labels/train/one.txt").read_text(encoding="utf-8").splitlines()
        self.assertEqual([line.split()[0] for line in content], ["1", "1"])
        self.manager.edit_annotations(self.root / "images/train/one.jpg", [0], delete=True)
        content = (self.root / "labels/train/one.txt").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(content), 1)

    def test_prepare_yolo_dataset_regression(self):
        output = self.root.parent / "prepared"
        result = self.manager.prepare_yolo(output, val_fraction=0.5, seed=42)
        self.assertTrue((output / "data.yaml").is_file())
        self.assertTrue((output / "images/train").is_dir())
        self.assertTrue((output / "labels/train").is_dir())
        self.assertEqual(result, output)

    def test_composable_browser_filters_and_class_combinations(self):
        cv2.imwrite(str(self.root / "images/train/third.png"), np.zeros((90, 140, 3), dtype=np.uint8))
        browser = DatasetBrowserDialog(self.manager)
        browser.annotations.item(0).setSelected(True)
        browser.annotations.item(1).setSelected(True)
        self.assertEqual(browser.affected.text(), "2 annotations selected")
        browser.include_classes.item(0).setCheckState(Qt.Checked)
        browser.include_classes.item(1).setCheckState(Qt.Checked)
        self.assertEqual(sum(browser._matches(row) for row in browser.index), 1)
        browser.exclude_classes.item(0).setCheckState(Qt.Checked)
        self.assertEqual(sum(browser._matches(row) for row in browser.index), 0)
        browser.include_classes.item(0).setCheckState(Qt.Unchecked)
        self.assertEqual(sum(browser._matches(row) for row in browser.index), 1)  # helmet, excluding person
        browser.exclude_classes.item(0).setCheckState(Qt.Unchecked)
        browser.only_selected_classes.setChecked(True)
        self.assertEqual(sum(browser._matches(row) for row in browser.index), 1)
        browser.min_w.setValue(121)
        self.assertEqual(sum(browser._matches(row) for row in browser.index), 0)
        browser.min_w.setValue(0)
        browser.only_selected_classes.setChecked(False)
        browser.include_classes.item(1).setCheckState(Qt.Unchecked)
        browser.include_classes.item(0).setCheckState(Qt.Checked)
        browser.include_classes.item(1).setCheckState(Qt.Checked)
        browser.min_count.setValue(2)
        self.assertEqual(sum(browser._matches(row) for row in browser.index), 1)
        browser.min_count.setValue(0)
        browser.min_h.setValue(95)
        self.assertEqual(sum(browser._matches(row) for row in browser.index), 1)
        browser.min_h.setValue(0)
        browser.include_classes.item(0).setCheckState(Qt.Unchecked)
        browser.include_classes.item(1).setCheckState(Qt.Unchecked)
        browser.only_selected_classes.setChecked(False)
        browser.max_count.setValue(1)
        self.assertEqual(sum(browser._matches(row) for row in browser.index), 2)
        browser.max_count.setValue(999999)
        browser.format_filter.setCurrentIndex(browser.format_filter.findData(".png"))
        self.assertEqual(sum(browser._matches(row) for row in browser.index), 1)
        browser.annotation_filter.setCurrentIndex(2)
        self.assertEqual(sum(browser._matches(row) for row in browser.index), 1)
        browser.annotation_filter.setCurrentIndex(0)
        browser.format_filter.setCurrentIndex(0)
        browser.max_w.setValue(119)
        self.assertEqual(sum(browser._matches(row) for row in browser.index), 0)
        browser.close()

    def test_validation_reports_coordinate_errors_unused_classes_and_duplicates(self):
        label = self.root / "labels/train/one.txt"
        label.write_text("5 0.5 0.5 1.2 0.2\n5 0.95 0.5 0.2 0.2\n", encoding="utf-8")
        cv2.imwrite(str(self.root / "images/train/copy.jpg"), cv2.imread(str(self.root / "images/train/two.jpg")))
        (self.root / "labels/train/copy.txt").write_text("", encoding="utf-8")
        cv2.imwrite(str(self.root / "images/train/missing.jpg"), np.zeros((20, 20, 3), dtype=np.uint8))
        (self.root / "images/train/corrupt.jpg").write_bytes(b"not an image")
        (self.root / "labels/train/orphan.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        (self.root / "data.yaml").write_text("names:\n  0: other\n", encoding="utf-8")
        report = self.manager.validate_dataset()
        codes = {issue["code"] for issue in report["issues"]}
        self.assertIn("invalid_class_id", codes)
        self.assertIn("invalid_coordinates", codes)
        self.assertIn("box_outside_image", codes)
        self.assertIn("duplicate_image", codes)
        self.assertIn("empty_annotation", codes)
        self.assertIn("unused_class", codes)
        self.assertIn("missing_annotation", codes)
        self.assertIn("corrupt_image", codes)
        self.assertIn("missing_image", codes)
        self.assertIn("class_definition_mismatch", codes)
        self.assertGreater(report["errors"], 0)

    def test_create_missing_empty_labels_is_confirmable_and_recoverable(self):
        label = self.root / "labels/train/two.txt"
        label.unlink()
        self.assertEqual(self.manager.create_empty_yolo_labels([self.root / "images/train/two.jpg"]), 1)
        self.assertTrue(label.is_file())
        self.manager.undo_last_operation()
        self.assertFalse(label.exists())

    def test_coco_delete_removes_only_selected_image_record_and_undo_restores(self):
        root = self.root.parent / "coco"
        (root / "images").mkdir(parents=True)
        (root / "annotations").mkdir()
        for name in ("one", "two"):
            cv2.imwrite(str(root / f"images/{name}.jpg"), np.zeros((50, 60, 3), dtype=np.uint8))
        coco = {"images": [{"id": 1, "file_name": "one.jpg", "width": 60, "height": 50},
                           {"id": 2, "file_name": "two.jpg", "width": 60, "height": 50}],
                "categories": [{"id": 1, "name": "person"}],
                "annotations": [{"id": 4, "image_id": 1, "category_id": 1,
                                 "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0}]}
        source = root / "annotations/instances_default.json"
        source.write_text(json.dumps(coco), encoding="utf-8")
        manager = DatasetManager(); manager.open(root)
        image = root / "images/two.jpg"
        manager.delete_images([image])
        data = json.loads(source.read_text(encoding="utf-8"))
        self.assertEqual([row["id"] for row in data["images"]], [1])
        self.assertEqual(len(data["annotations"]), 1)
        manager = DatasetManager(); manager.open(root); manager.undo_last_operation()
        restored = json.loads(source.read_text(encoding="utf-8"))
        self.assertEqual(len(restored["images"]), 2)
        self.assertEqual(image.exists(), True)

    def test_training_worker_flow_with_stubbed_model(self):
        class FakeModel:
            def __init__(self, _name): self.callbacks = {}; self.arguments = None
            def add_callback(self, name, callback): self.callbacks[name] = callback
            def train(self, **kwargs):
                self.arguments = kwargs
                return SimpleNamespace(save_dir="mock-run")
        fake = FakeModel("weights.pt")
        completed, failed = [], []
        worker = TrainingWorker("weights.pt", self.root / "data.yaml", "cpu", 1, 64, 1, self.root / "runs")
        worker.finished_ok.connect(completed.append); worker.failed.connect(failed.append)
        with patch("ultralytics.YOLO", return_value=fake):
            worker.run()
        self.assertFalse(failed)
        self.assertEqual(completed, ["mock-run"])
        self.assertEqual(fake.arguments["device"], "cpu")
        self.assertEqual(fake.arguments["epochs"], 1)

    def test_inference_model_loader_flow_with_stubbed_model(self):
        model_path = self.root / "mock.pt"; model_path.write_bytes(b"test checkpoint")
        fake = SimpleNamespace(names={0: "person"}, model=None, task="detect", ckpt={})
        loaded, failed = [], []
        worker = ModelLoadWorker(model_path)
        worker.loaded.connect(lambda model, names, details: loaded.append((model, names, details)))
        worker.failed.connect(failed.append)
        with patch("ultralytics.YOLO", return_value=fake):
            worker.run()
        self.assertFalse(failed)
        self.assertIs(loaded[0][0], fake)
        self.assertEqual(loaded[0][1], ["person"])

    def test_application_startup_smoke_without_restoring_user_session(self):
        with patch("studio.load_state", return_value={}):
            window = DrillAnnotationStudio()
        self.assertIsNotNone(window.manager)
        self.assertEqual(window.windowTitle(), "YOLO Dataset & Training Studio")
        window.session_save_timer.stop()
        window.inference_panel.stop_and_wait()
        window.close()


if __name__ == "__main__":
    unittest.main()
