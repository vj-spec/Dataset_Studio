import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from unittest.mock import patch
from PySide6.QtWidgets import QApplication, QListWidgetItem
from PySide6.QtCore import Qt

from dataset_merge import analyze_merge, execute_merge, finalize_merge, inspect_merge_source
from merge_studio import DatasetMergeDialog


class DatasetMergeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent.parent)
        self.base = Path(self.temp.name)
        self.a = self._dataset("alpha", ["Person", "Car"], [("same.jpg", (0, 0, 0)), ("only-a.jpg", (20, 30, 40))], [0, 1])
        self.b = self._dataset("beta", ["Human", "Helmet"], [("same.jpg", (0, 0, 0)), ("only-b.jpg", (50, 60, 70))], [0, 1])

    def tearDown(self):
        self.temp.cleanup()

    def _dataset(self, name, classes, images, class_ids):
        root = self.base / name
        (root / "images" / "train").mkdir(parents=True)
        (root / "labels" / "train").mkdir(parents=True)
        for (filename, color), cid in zip(images, class_ids):
            image = np.full((48, 64, 3), color, dtype=np.uint8)
            cv2.imwrite(str(root / "images" / "train" / filename), image)
            (root / "labels" / "train" / Path(filename).with_suffix(".txt").name).write_text(
                f"{cid} 0.5 0.5 0.25 0.25\n", encoding="utf-8")
        (root / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
        (root / "data.yaml").write_text("path: .\ntrain: images/train\nnames:\n" +
                                        "".join(f"  {i}: {name}\n" for i, name in enumerate(classes)), encoding="utf-8")
        return root

    def _analyze(self):
        sources = [inspect_merge_source(self.a), inspect_merge_source(self.b)]
        for source in sources:
            source["enabled"] = True
            source["selected_images"] = [row["path"] for row in source["records"]]
        # Deliberately different numeric positions and names map explicitly.
        return analyze_merge(sources, ["Person", "Car", "Helmet"], {
            str(self.a.resolve()): {0: 0, 1: 1},
            str(self.b.resolve()): {0: 0, 1: 2},
        })

    def test_explicit_mapping_duplicate_policy_filename_safety_and_source_preservation(self):
        analysis = self._analyze()
        self.assertEqual(len(analysis["duplicate_groups"]), 1)
        original = [(p, p.read_bytes()) for p in self.base.rglob("*.jpg")]
        plan = finalize_merge(analysis, self.base / "merged", "keep_first", {},
                              {"train": 80, "val": 10, "test": 10}, 42)
        result = execute_merge(plan)
        out = Path(result["output"])
        self.assertEqual(len(list((out / "images").rglob("*.jpg"))), 3)
        labels = [p.read_text(encoding="utf-8") for p in (out / "labels").rglob("*.txt")]
        self.assertIn("2 0.5", "".join(labels))
        self.assertTrue((out / "merge_metadata.json").is_file())
        meta = __import__("json").loads((out / "merge_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(set(item["output_split"] for item in meta["images"]), {"train", "val", "test"})
        split_by_hash = {}
        for item in meta["images"]:
            prior = split_by_hash.setdefault(item["image_sha256"], item["output_split"])
            self.assertEqual(prior, item["output_split"])
        self.assertEqual([(p, p.read_bytes()) for p, _ in original], original)
        self.assertEqual(len({p.relative_to(out).as_posix().casefold() for p in (out / "images").rglob("*.jpg")}), 3)

    def test_requires_explicit_mapping_and_duplicate_decision(self):
        sources = [inspect_merge_source(self.a)]
        sources[0].update(enabled=True, selected_images=[r["path"] for r in sources[0]["records"]])
        with self.assertRaisesRegex(Exception, "Map or explicitly exclude"):
            analyze_merge(sources, ["Person"], {str(self.a.resolve()): {0: 0}})
        analysis = self._analyze()
        with self.assertRaisesRegex(Exception, "Choose which occurrence"):
            finalize_merge(analysis, self.base / "manual", "manual", {}, {"train": 100, "val": 0, "test": 0}, 2)
        group = analysis["duplicate_groups"][0]
        first = analysis["records"][group["records"][0]]["key"]
        manual = finalize_merge(analysis, self.base / "manual-ok", "manual", {group["hash"]: first},
                                {"train": 100, "val": 0, "test": 0}, 2)
        self.assertEqual(len(manual["records"]), 3)

    def test_class_exclusion_is_explicit_and_remaps_remaining_boxes(self):
        sources = [inspect_merge_source(self.a), inspect_merge_source(self.b)]
        for source in sources:
            source.update(enabled=True, selected_images=[row["path"] for row in source["records"]])
        analysis = analyze_merge(sources, ["Person", "Car"], {
            str(self.a.resolve()): {0: 0, 1: 1}, str(self.b.resolve()): {0: 0, 1: None}})
        plan = finalize_merge(analysis, self.base / "excluded", "keep_first", {},
                              {"train": 100, "val": 0, "test": 0}, 5)
        self.assertEqual(sum(row["annotation_count"] for row in plan["records"]), 2)
        result = execute_merge(plan)
        self.assertEqual(result["validation"]["classes"], 2)
        self.assertEqual(sum(len(p.read_text(encoding="utf-8").splitlines()) for p in (Path(result["output"]) / "labels").rglob("*.txt")), 2)

    def test_failed_merge_cleans_staging_and_keeps_sources(self):
        analysis = self._analyze()
        plan = finalize_merge(analysis, self.base / "broken", "keep_all", {},
                              {"train": 80, "val": 10, "test": 10}, 9)
        def fail_copy(_source, _target):
            raise OSError("simulated copy failure")
        with self.assertRaisesRegex(OSError, "simulated"):
            execute_merge(plan, copy_file=fail_copy)
        self.assertFalse((self.base / "broken").exists())
        self.assertFalse(list(self.base.glob(".broken.merge-staging-*")))
        self.assertTrue((self.a / "images/train/same.jpg").exists())
        self.assertTrue((self.b / "images/train/same.jpg").exists())

    def test_same_stem_different_extensions_get_distinct_image_label_pairs(self):
        png = self.a / "images/train/only-a.png"
        cv2.imwrite(str(png), np.full((48, 64, 3), (90, 80, 70), dtype=np.uint8))
        analysis = self._analyze()
        plan = finalize_merge(analysis, self.base / "stem-conflict", "keep_first", {},
                              {"train": 100, "val": 0, "test": 0}, 8)
        result = execute_merge(plan)
        out = Path(result["output"])
        images = list((out / "images").rglob("*.*"))
        labels = list((out / "labels").rglob("*.txt"))
        self.assertEqual(len(images), 4)
        self.assertEqual(len(labels), 4)
        from dataset_manager import DatasetManager
        manager = DatasetManager(); manager.open(out)
        self.assertEqual(len(manager.image_records()), 4)

    def test_merge_dialog_individual_image_selection_and_preview_validation(self):
        source = inspect_merge_source(self.a)
        source["enabled"] = True
        source["selected_images"] = [record["path"] for record in source["records"]]
        dialog = DatasetMergeDialog()
        dialog.sources = [source]
        item = QListWidgetItem("alpha"); item.setCheckState(Qt.Checked); dialog.source_list.addItem(item)
        dialog.source_list.setCurrentRow(0)
        dialog.targets = ["Person", "Car"]
        dialog.mappings[str(self.a.resolve())] = {0: 0, 1: 1}
        dialog._rebuild_mappings(); dialog._source_changed(0)
        dialog.image_list.item(0).setCheckState(Qt.Unchecked)
        self.assertEqual(len(source["selected_images"]), 1)
        dialog.train_pct.setValue(100); dialog.val_pct.setValue(0); dialog.test_pct.setValue(0)
        with patch("merge_studio.QMessageBox.information"):
            dialog._preview()
        self.assertEqual(len(dialog.plan["records"]), 1)
        self.assertTrue(dialog.merge_btn.isEnabled())


if __name__ == "__main__":
    unittest.main()
