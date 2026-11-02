"""User interface for explicitly reconciling and merging YOLO datasets."""
from __future__ import annotations

import difflib
import threading
from pathlib import Path

from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget, QListWidgetItem,
    QLabel, QFileDialog, QMessageBox, QTableWidget, QTableWidgetItem, QComboBox,
    QLineEdit, QSpinBox, QProgressBar, QCheckBox, QGroupBox, QFormLayout,
    QSplitter, QTextEdit, QInputDialog, QHeaderView
    , QRadioButton, QButtonGroup, QDialogButtonBox
)

from dataset_merge import (MergeError, analyze_merge, execute_merge, finalize_merge,
                           inspect_merge_source)


class _MergeWorker(QThread):
    progress = Signal(int, int, str)
    finished_result = Signal(object)
    failed = Signal(str)

    def __init__(self, callback, parent=None):
        super().__init__(parent)
        self.callback = callback

    def run(self):
        try:
            self.finished_result.emit(self.callback(lambda done, total, message: self.progress.emit(done, total, message)))
        except Exception as exc:
            self.failed.emit(str(exc))


class DatasetMergeDialog(QDialog):
    """Merge only supported YOLO datasets; unsupported formats are preview-only."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Dataset Merge Studio")
        self.resize(1100, 760)
        self.sources = []
        self.targets = []
        self.mappings = {}
        self.analysis = None
        self.plan = None
        self.duplicate_decisions = {}
        self.cancel_event = threading.Event()
        self.worker = None
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.addWidget(QLabel("Add YOLO datasets, explicitly map every source class, preview the result, then confirm merge."))
        split = QSplitter(Qt.Horizontal)
        left = QVBoxLayout()
        source_box = QGroupBox("Source datasets")
        source_layout = QVBoxLayout(source_box)
        self.source_list = QListWidget()
        self.source_list.currentRowChanged.connect(self._source_changed)
        source_layout.addWidget(self.source_list, 1)
        buttons = QHBoxLayout()
        add = QPushButton("Add dataset…"); add.clicked.connect(self._add_source)
        remove = QPushButton("Remove"); remove.clicked.connect(self._remove_source)
        buttons.addWidget(add); buttons.addWidget(remove)
        source_layout.addLayout(buttons)
        left.addWidget(source_box, 1)
        self.details = QTextEdit(); self.details.setReadOnly(True); self.details.setMaximumHeight(150)
        left.addWidget(self.details)
        left.addWidget(QLabel("Images in selected dataset (check to include)"))
        self.image_list = QListWidget()
        self.image_list.itemChanged.connect(self._image_selection_changed)
        self.image_list.currentRowChanged.connect(self._show_image_preview)
        left.addWidget(self.image_list, 2)
        self.image_preview = QLabel("Select an image to preview")
        self.image_preview.setAlignment(Qt.AlignCenter); self.image_preview.setMinimumHeight(120)
        left.addWidget(self.image_preview)
        left_widget = QWidget(); left_widget.setLayout(left); split.addWidget(left_widget)

        right = QVBoxLayout()
        classes_box = QGroupBox("Unified classes and source mappings")
        classes_layout = QVBoxLayout(classes_box)
        self.class_table = QTableWidget(0, 4)
        self.class_table.setHorizontalHeaderLabels(["Dataset", "Source ID", "Source class", "Map explicitly to"])
        self.class_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.class_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.class_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.class_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        classes_layout.addWidget(self.class_table, 1)
        class_buttons = QHBoxLayout()
        create = QPushButton("Create unified class…"); create.clicked.connect(self._create_class)
        rename = QPushButton("Rename selected class…"); rename.clicked.connect(self._rename_class)
        merge = QPushButton("Merge selected classes…"); merge.clicked.connect(self._merge_classes)
        class_buttons.addWidget(create); class_buttons.addWidget(rename); class_buttons.addWidget(merge)
        classes_layout.addLayout(class_buttons)
        right.addWidget(classes_box, 1)

        opts = QGroupBox("Duplicates and split")
        form = QFormLayout(opts)
        self.duplicate_policy = QComboBox()
        self.duplicate_policy.addItem("Keep all duplicates", "keep_all")
        self.duplicate_policy.addItem("Keep first occurrence", "keep_first")
        self.duplicate_policy.addItem("Keep latest selected occurrence", "keep_latest")
        self.duplicate_policy.addItem("Review duplicate groups manually", "manual")
        form.addRow("Exact duplicates", self.duplicate_policy)
        self.train_pct = QSpinBox(); self.train_pct.setRange(1, 100); self.train_pct.setValue(80)
        self.val_pct = QSpinBox(); self.val_pct.setRange(0, 99); self.val_pct.setValue(10)
        self.test_pct = QSpinBox(); self.test_pct.setRange(0, 99); self.test_pct.setValue(10)
        self.seed = QSpinBox(); self.seed.setRange(0, 2**31 - 1); self.seed.setValue(42)
        self.group_source = QCheckBox("Keep each source dataset together")
        form.addRow("Train %", self.train_pct); form.addRow("Validation %", self.val_pct)
        form.addRow("Test %", self.test_pct); form.addRow("Random seed", self.seed)
        form.addRow(self.group_source)
        right.addWidget(opts)
        right_widget = QWidget(); right_widget.setLayout(right); split.addWidget(right_widget)
        split.setSizes([340, 740]); root.addWidget(split, 1)

        self.status = QLabel("No datasets selected.")
        root.addWidget(self.status)
        self.progress = QProgressBar(); self.progress.setRange(0, 100); root.addWidget(self.progress)
        actions = QHBoxLayout()
        self.preview_btn = QPushButton("Preview and validate"); self.preview_btn.clicked.connect(self._preview)
        self.merge_btn = QPushButton("Confirm and merge…"); self.merge_btn.clicked.connect(self._merge)
        self.merge_btn.setEnabled(False)
        self.cancel_btn = QPushButton("Cancel merge"); self.cancel_btn.clicked.connect(self.cancel_event.set); self.cancel_btn.setEnabled(False)
        close = QPushButton("Close"); close.clicked.connect(self.reject)
        actions.addWidget(self.preview_btn); actions.addWidget(self.merge_btn); actions.addWidget(self.cancel_btn); actions.addStretch(1); actions.addWidget(close)
        root.addLayout(actions)

    def _add_source(self):
        folder = QFileDialog.getExistingDirectory(self, "Select dataset folder")
        if not folder:
            return
        try:
            source = inspect_merge_source(Path(folder))
        except Exception as exc:
            QMessageBox.warning(self, "Dataset inspection failed", str(exc)); return
        if any(s["root"].casefold() == source["root"].casefold() for s in self.sources):
            QMessageBox.information(self, "Already added", "That dataset is already selected."); return
        source["enabled"] = source["mergeable"]
        source["selected_images"] = [r["path"] for r in source["records"]]
        source["image_checks"] = {}
        self.sources.append(source)
        label = f"{'✓' if source['mergeable'] else 'Preview only'}  {source['name']} — {source['images']} images, {source['annotations']} annotations"
        item = QListWidgetItem(label); item.setCheckState(Qt.Checked if source["enabled"] else Qt.Unchecked)
        self.source_list.addItem(item); self.source_list.setCurrentRow(len(self.sources)-1)
        self._rebuild_mappings(); self.merge_btn.setEnabled(False)

    def _remove_source(self):
        row = self.source_list.currentRow()
        if row < 0: return
        self.sources.pop(row); self.source_list.takeItem(row)
        self._rebuild_mappings(); self.merge_btn.setEnabled(False); self._source_changed(self.source_list.currentRow())

    def _source_changed(self, row):
        if row < 0 or row >= len(self.sources):
            self.details.clear(); return
        source = self.sources[row]
        lines = [f"Path: {source['root']}", f"Format: {source['format']} ({'mergeable' if source['mergeable'] else 'preview only'})",
                 f"Images: {source['images']}    Annotations: {source['annotations']}", "Classes:"]
        for cid, name in enumerate(source["classes"]):
            lines.append(f"  {cid}: {name} — {source['class_distribution'].get(cid, {}).get('count', 0)} annotations")
        self.details.setPlainText("\n".join(lines))
        self.image_list.blockSignals(True); self.image_list.clear()
        selected = set(source["selected_images"])
        for record in source["records"]:
            item = QListWidgetItem(f"{Path(record['path']).name} — {record['width']}×{record['height']}, {record['annotations']} annotations")
            item.setData(Qt.UserRole, record["path"]); item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if record["path"] in selected else Qt.Unchecked)
            self.image_list.addItem(item)
        self.image_list.blockSignals(False)
        self._show_image_preview(self.image_list.currentRow())

    def _image_selection_changed(self, _item):
        row = self.source_list.currentRow()
        if row < 0: return
        source = self.sources[row]
        source["selected_images"] = [self.image_list.item(i).data(Qt.UserRole) for i in range(self.image_list.count())
                                     if self.image_list.item(i).checkState() == Qt.Checked]
        self.merge_btn.setEnabled(False)

    def _show_image_preview(self, row):
        if row < 0 or row >= self.image_list.count():
            self.image_preview.setPixmap(QPixmap()); self.image_preview.setText("Select an image to preview"); return
        pixmap = QPixmap(self.image_list.item(row).data(Qt.UserRole))
        self.image_preview.setText("")
        self.image_preview.setPixmap(pixmap.scaled(self.image_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _rebuild_mappings(self):
        self.class_table.setRowCount(0)
        for source_index, source in enumerate(self.sources):
            if not source["mergeable"]: continue
            root = source["root"]
            mapping = self.mappings.setdefault(root, {})
            for cid, name in enumerate(source["classes"]):
                row = self.class_table.rowCount(); self.class_table.insertRow(row)
                self.class_table.setItem(row, 0, QTableWidgetItem(source["name"]))
                self.class_table.setItem(row, 1, QTableWidgetItem(str(cid)))
                self.class_table.setItem(row, 2, QTableWidgetItem(name))
                combo = QComboBox(); combo.addItem("— choose mapping —", "unset")
                for target_id, target_name in enumerate(self.targets): combo.addItem(f"{target_id}: {target_name}", target_id)
                combo.addItem("Exclude this source class", None)
                prior = mapping.get(cid, "unset")
                index = combo.findData(prior)
                combo.setCurrentIndex(max(0, index))
                combo.currentIndexChanged.connect(lambda _idx, r=root, c=cid, n=name, box=combo: self._mapping_changed(r, c, n, box.currentData()))
                self.class_table.setCellWidget(row, 3, combo)
        if not self.targets:
            self.status.setText("Create unified classes, then map every source class. No numeric or name-based auto-mapping is performed.")

    def _mapping_changed(self, root, cid, source_name, target):
        if target == "unset": self.mappings.setdefault(root, {}).pop(cid, None)
        elif isinstance(target, int):
            target_name = self.targets[target]
            similarity = difflib.SequenceMatcher(None, source_name.casefold(), target_name.casefold()).ratio()
            if source_name.casefold() != target_name.casefold() and similarity >= .65:
                answer = QMessageBox.question(self, "Potential semantic conflict",
                    f"'{source_name}' is similar to '{target_name}', but similar names can have different meanings. Confirm this explicit mapping?")
                if answer != QMessageBox.Yes:
                    self.mappings.setdefault(root, {}).pop(cid, None)
                    self._rebuild_mappings(); return
            self.mappings.setdefault(root, {})[cid] = target
        else: self.mappings.setdefault(root, {})[cid] = None
        self.merge_btn.setEnabled(False)

    def _create_class(self):
        name, ok = QInputDialog.getText(self, "New unified class", "Class name:")
        name = name.strip()
        if not ok or not name: return
        if any(name.casefold() == existing.casefold() for existing in self.targets):
            QMessageBox.warning(self, "Duplicate class", "Unified class names must be unique."); return
        similar = [existing for existing in self.targets if difflib.SequenceMatcher(None, name.casefold(), existing.casefold()).ratio() >= .72]
        if similar and QMessageBox.question(self, "Potential semantic conflict", f"Similar classes exist: {', '.join(similar)}. Create a separate class anyway?") != QMessageBox.Yes: return
        self.targets.append(name); self._rebuild_mappings()

    def _rename_class(self):
        row = self.class_table.currentRow()
        if row < 0: return
        combo = self.class_table.cellWidget(row, 3)
        value = combo.currentData()
        if not isinstance(value, int): QMessageBox.information(self, "Select a mapping", "Select a mapped unified class in the current row first."); return
        name, ok = QInputDialog.getText(self, "Rename unified class", "New name:", text=self.targets[value])
        name = name.strip()
        if not ok or not name: return
        if any(i != value and n.casefold() == name.casefold() for i, n in enumerate(self.targets)):
            QMessageBox.warning(self, "Duplicate class", "Unified class names must be unique."); return
        self.targets[value] = name; self._rebuild_mappings()

    def _merge_classes(self):
        selected = sorted({self.class_table.cellWidget(row, 3).currentData() for row in range(self.class_table.rowCount())
                           if self.class_table.item(row, 0).isSelected() and isinstance(self.class_table.cellWidget(row, 3).currentData(), int)})
        # Row selection varies by Qt selection mode; also allow choosing two mapped IDs via prompt.
        if len(selected) < 2:
            values = [str(i) for i in range(len(self.targets))]
            if len(values) < 2: QMessageBox.information(self, "Need two classes", "Create at least two unified classes first."); return
            first, ok = QInputDialog.getItem(self, "Merge classes", "Keep target class:", values, 0, False)
            if not ok: return
            second, ok = QInputDialog.getItem(self, "Merge classes", "Merge target class into it:", [v for v in values if v != first], 0, False)
            if not ok: return
            keep, remove = int(first), int(second)
        else: keep, remove = selected[:2]
        if QMessageBox.question(self, "Confirm class merge", f"Map all source annotations from '{self.targets[remove]}' to '{self.targets[keep]}'?") != QMessageBox.Yes: return
        for mapping in self.mappings.values():
            for cid, target in list(mapping.items()):
                if target == remove: mapping[cid] = keep
                elif isinstance(target, int) and target > remove: mapping[cid] = target - 1
        self.targets.pop(remove); self._rebuild_mappings()

    def _set_enabled_from_checks(self):
        for index, source in enumerate(self.sources):
            item = self.source_list.item(index)
            source["enabled"] = bool(item and item.checkState() == Qt.Checked and source["mergeable"])

    def _preview(self):
        self._set_enabled_from_checks()
        if not sum((self.train_pct.value(), self.val_pct.value(), self.test_pct.value())) == 100:
            QMessageBox.warning(self, "Invalid split", "Train, validation and test percentages must total 100."); return
        try:
            self.analysis = analyze_merge(self.sources, self.targets, self.mappings)
            self.duplicate_decisions = self._review_duplicates(self.analysis) if self.duplicate_policy.currentData() == "manual" else {}
            self.plan = finalize_merge(self.analysis, Path(self.sources[0]["root"]).parent / "merged_dataset",
                                       self.duplicate_policy.currentData(), self.duplicate_decisions,
                                       {"train": self.train_pct.value(), "val": self.val_pct.value(), "test": self.test_pct.value()},
                                       self.seed.value(), self.group_source.isChecked())
        except Exception as exc:
            self.plan = None; self.merge_btn.setEnabled(False)
            QMessageBox.warning(self, "Preview validation failed", str(exc)); return
        anns = sum(r["annotation_count"] for r in self.analysis["records"])
        source_images = sum(len(source["selected_images"]) for source in self.sources if source.get("enabled"))
        source_annotations = sum(record["annotations"] for source in self.sources if source.get("enabled")
                                 for record in source["records"] if record["path"] in set(source["selected_images"]))
        expected_annotations = sum(record["annotation_count"] for record in self.plan["records"])
        dup = len(self.analysis["duplicate_groups"])
        split_counts = {name: sum(1 for value in self.plan["split_map"].values() if value == name) for name in ("train", "val", "test")}
        mapping_text = []
        source_warnings = []
        for source in self.analysis["sources"]:
            mapping_text.append(f"{source['name']}: " + ", ".join(f"{source['classes'][int(cid)]} → {self.targets[target] if target is not None else 'EXCLUDED'}" for cid, target in source["mapping"].items()))
            warning_issues = [issue for issue in source.get("validation", {}).get("issues", []) if issue["severity"] == "warning"]
            source_warnings.extend(f"{source['name']}: {issue['message']}" for issue in warning_issues)
        text = (f"Sources: {len(self.analysis['sources'])}\nSelected source images: {source_images}\nSource annotations: {source_annotations}\n"
                f"Images before duplicate handling: {len(self.analysis['records'])}\n"
                f"Mapped annotations before duplicate handling: {anns}\nUnified classes: {', '.join(self.targets)}\n"
                f"Exact duplicate groups: {dup}\nDuplicate policy: {self.duplicate_policy.currentText()}\n"
                f"Expected output: {len(self.plan['records'])} images, {expected_annotations} annotations; split: {split_counts}\n"
                f"Potential source warnings: {len(source_warnings)}\n\nMappings:\n" + "\n".join(mapping_text) +
                ("\n\nWarnings:\n" + "\n".join(source_warnings[:20]) if source_warnings else ""))
        QMessageBox.information(self, "Merge preview and validation", text)
        self.status.setText(f"Validated preview: {len(self.plan['records'])} images, {anns} annotations, {dup} duplicate groups.")
        self.merge_btn.setEnabled(True)

    def _review_duplicates(self, analysis):
        decisions = {}
        records = analysis["records"]
        for number, group in enumerate(analysis["duplicate_groups"], 1):
            dialog = QDialog(self); dialog.setWindowTitle(f"Exact duplicate {number}/{len(analysis['duplicate_groups'])}")
            layout = QVBoxLayout(dialog)
            layout.addWidget(QLabel("These image files have identical SHA-256 hashes. Choose which occurrences to retain."))
            choices = QButtonGroup(dialog); rows = []
            for record_index in group["records"]:
                record = records[record_index]
                row = QHBoxLayout(); radio = QRadioButton(f"Keep {record['source_name']} — {Path(record['image']).relative_to(record['source_root'])}")
                choices.addButton(radio); row.addWidget(radio)
                preview = QLabel(); preview.setAlignment(Qt.AlignCenter); preview.setFixedSize(220, 140)
                pixmap = QPixmap(record["image"])
                preview.setPixmap(pixmap.scaled(preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                row.addWidget(preview); layout.addLayout(row); rows.append((radio, record))
            keep_all = QRadioButton("Keep all identical copies"); choices.addButton(keep_all); layout.addWidget(keep_all)
            rows[0][0].setChecked(True)
            buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            buttons.accepted.connect(dialog.accept); buttons.rejected.connect(dialog.reject); layout.addWidget(buttons)
            if dialog.exec() != QDialog.Accepted:
                raise MergeError("Duplicate review was cancelled; no merge was started.")
            selected = choices.checkedButton()
            decisions[group["hash"]] = "all" if selected is keep_all else next(record["key"] for radio, record in rows if radio is selected)
        return decisions

    def _merge(self):
        if not self.plan:
            QMessageBox.information(self, "Preview required", "Run preview and validation before merging."); return
        output, _ = QFileDialog.getSaveFileName(self, "Choose merged dataset folder name", str(Path(self.sources[0]["root"]).parent / "merged_dataset"))
        if not output: return
        try:
            self.plan = finalize_merge(self.analysis, Path(output), self.duplicate_policy.currentData(), self.duplicate_decisions,
                                       {"train": self.train_pct.value(), "val": self.val_pct.value(), "test": self.test_pct.value()},
                                       self.seed.value(), self.group_source.isChecked())
        except Exception as exc:
            QMessageBox.warning(self, "Cannot merge", str(exc)); return
        if QMessageBox.question(self, "Confirm dataset merge", f"Create a new merged dataset at:\n{output}\n\nSources will remain unchanged. Continue?") != QMessageBox.Yes: return
        self.cancel_event.clear(); self.cancel_btn.setEnabled(True); self.merge_btn.setEnabled(False); self.preview_btn.setEnabled(False)
        self.worker = _MergeWorker(lambda progress: execute_merge(self.plan, progress, self.cancel_event), self)
        self.worker.progress.connect(self._progress); self.worker.finished_result.connect(self._merge_done); self.worker.failed.connect(self._merge_failed)
        self.worker.start()

    def _progress(self, done, total, message):
        self.progress.setValue(int(done * 100 / max(total, 1))); self.status.setText(message)

    def _merge_done(self, result):
        self.cancel_btn.setEnabled(False); self.preview_btn.setEnabled(True); self.plan = None
        QMessageBox.information(self, "Merge complete", f"Merged dataset validated and saved:\n{result['output']}\n\nMetadata: {result['metadata']}")
        self.status.setText("Merge complete and validated.")

    def _merge_failed(self, message):
        self.cancel_btn.setEnabled(False); self.preview_btn.setEnabled(True); self.merge_btn.setEnabled(True)
        self.status.setText("Merge failed; staging output was cleaned up.")
        QMessageBox.warning(self, "Merge failed", message)

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            event.ignore(); QMessageBox.information(self, "Merge in progress", "Wait for the merge to stop or finish before closing."); return
        event.accept()
