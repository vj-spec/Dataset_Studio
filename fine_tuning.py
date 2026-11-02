"""Create a separate YOLO detection dataset reconciled to a pretrained model."""
from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QMessageBox, QLineEdit, QTableWidget, QTableWidgetItem, QComboBox,
    QHeaderView, QInputDialog, QSpinBox, QCheckBox, QProgressBar, QFormLayout)

from dataset_merge import MergeError, analyze_merge, execute_merge, finalize_merge, inspect_merge_source


class _ModelInspectWorker(QThread):
    loaded = Signal(object, str)
    failed = Signal(str)

    def __init__(self, path, parent=None):
        super().__init__(parent); self.path = path

    def run(self):
        try:
            from ultralytics import YOLO
            model = YOLO(self.path)
            task = getattr(model, "task", "") or getattr(getattr(model, "model", None), "task", "")
            names = model.names or {}
            names = [str(names[key]) for key in sorted(names)] if isinstance(names, dict) else list(names)
            if task and task != "detect":
                raise MergeError(f"This workflow currently supports detection models; selected model task is '{task}'.")
            self.loaded.emit(names, task or "detect")
        except Exception as exc:
            self.failed.emit(str(exc))


class _PrepareWorker(QThread):
    progress = Signal(int, int, str)
    complete = Signal(object)
    failed = Signal(str)

    def __init__(self, plan, parent=None):
        super().__init__(parent); self.plan = plan; self.cancel_event = threading.Event()

    def run(self):
        try:
            self.complete.emit(execute_merge(self.plan, lambda a, b, c: self.progress.emit(a, b, c), self.cancel_event))
        except Exception as exc:
            self.failed.emit(str(exc))


class FineTuningPreparationDialog(QDialog):
    """Inspect model classes and create a mapped dataset in a new folder."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Fine-Tuning Dataset Preparation")
        self.resize(880, 650)
        self.source = None; self.model_path = ""; self.model_classes = []; self.task = ""
        self.target_classes = []; self.mapping = {}; self.worker = None
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.addWidget(QLabel("Create a separate YOLO detection dataset. The pretrained model and source dataset are never modified."))
        row = QHBoxLayout()
        self.model_edit = QLineEdit(); self.model_edit.setReadOnly(True)
        model_btn = QPushButton("Select pretrained model…"); model_btn.clicked.connect(self._select_model)
        row.addWidget(self.model_edit, 1); row.addWidget(model_btn); root.addLayout(row)
        row = QHBoxLayout()
        self.dataset_edit = QLineEdit(); self.dataset_edit.setReadOnly(True)
        dataset_btn = QPushButton("Select merged YOLO dataset…"); dataset_btn.clicked.connect(self._select_dataset)
        row.addWidget(self.dataset_edit, 1); row.addWidget(dataset_btn); root.addLayout(row)
        self.model_info = QLabel("Select a detection model to inspect its class definitions."); self.model_info.setWordWrap(True)
        root.addWidget(self.model_info)
        self.table = QTableWidget(0, 3); self.table.setHorizontalHeaderLabels(["Dataset class ID", "Dataset class", "Map to model/target class"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        root.addWidget(self.table, 1)
        self.add_class_btn = QPushButton("Add fine-tuning class…"); self.add_class_btn.clicked.connect(self._add_class); self.add_class_btn.setEnabled(False)
        root.addWidget(self.add_class_btn)
        form = QFormLayout()
        self.train = QSpinBox(); self.train.setRange(1, 100); self.train.setValue(80)
        self.val = QSpinBox(); self.val.setRange(0, 99); self.val.setValue(10)
        self.test = QSpinBox(); self.test.setRange(0, 99); self.test.setValue(10)
        self.seed = QSpinBox(); self.seed.setRange(0, 2**31-1); self.seed.setValue(42)
        self.group = QCheckBox("Keep related images from each source dataset together")
        form.addRow("Train %", self.train); form.addRow("Validation %", self.val); form.addRow("Test %", self.test); form.addRow("Seed", self.seed); form.addRow(self.group)
        root.addLayout(form)
        self.note = QLabel("Existing model classes are retained in their original order. New target classes are appended. If class count/order differs from the pretrained model, the detection head must be adapted during fine-tuning; supported Ultralytics models initialize the new head while transferring compatible pretrained weights.")
        self.note.setWordWrap(True); root.addWidget(self.note)
        self.progress = QProgressBar(); root.addWidget(self.progress)
        self.status = QLabel("Ready"); root.addWidget(self.status)
        actions = QHBoxLayout()
        self.prepare_btn = QPushButton("Prepare separate dataset…"); self.prepare_btn.clicked.connect(self._prepare); self.prepare_btn.setEnabled(False)
        cancel = QPushButton("Cancel operation"); cancel.clicked.connect(self._cancel); cancel.setEnabled(False); self.cancel_btn = cancel
        close = QPushButton("Close"); close.clicked.connect(self.reject)
        actions.addWidget(self.prepare_btn); actions.addWidget(cancel); actions.addStretch(1); actions.addWidget(close); root.addLayout(actions)

    def _select_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select pretrained detection model", "", "Model weights (*.pt);;All files (*)")
        if not path: return
        self.model_path = path; self.model_edit.setText(path); self.model_info.setText("Loading model metadata in background…")
        self.model_worker = _ModelInspectWorker(path, self)
        self.model_worker.loaded.connect(self._model_loaded); self.model_worker.failed.connect(self._model_failed); self.model_worker.start()

    def _model_loaded(self, names, task):
        if not names:
            self.model_info.setText("The model did not provide class names. It cannot be used for explicit reconciliation."); return
        self.model_classes = list(names); self.task = task; self.target_classes = list(names)
        self.model_info.setText(f"Task: {task}. Pretrained model classes ({len(names)}): " + ", ".join(names))
        self.add_class_btn.setEnabled(True); self._rebuild_table()

    def _model_failed(self, message):
        self.model_info.setText("Model inspection failed."); QMessageBox.warning(self, "Model inspection failed", message)

    def _select_dataset(self):
        folder = QFileDialog.getExistingDirectory(self, "Select merged YOLO dataset")
        if not folder: return
        try:
            self.source = inspect_merge_source(Path(folder))
            if not self.source["mergeable"]: raise MergeError("Only YOLO TXT datasets are supported for fine-tuning preparation.")
            self.dataset_edit.setText(folder); self._rebuild_table()
        except Exception as exc:
            QMessageBox.warning(self, "Dataset inspection failed", str(exc)); self.source = None

    def _add_class(self):
        name, ok = QInputDialog.getText(self, "Add fine-tuning class", "Class name:")
        name = name.strip()
        if not ok or not name: return
        if any(name.casefold() == existing.casefold() for existing in self.target_classes):
            QMessageBox.warning(self, "Duplicate class", "Target class names must be unique."); return
        self.target_classes.append(name); self._rebuild_table()

    def _rebuild_table(self):
        self.table.setRowCount(0)
        if not self.source or not self.model_classes: return
        self.mapping = {}
        for cid, name in enumerate(self.source["classes"]):
            row = self.table.rowCount(); self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(str(cid))); self.table.setItem(row, 1, QTableWidgetItem(name))
            combo = QComboBox(); combo.addItem("— choose explicitly —", "unset")
            for target_id, target_name in enumerate(self.target_classes): combo.addItem(f"{target_id}: {target_name}", target_id)
            combo.addItem("Exclude this dataset class", None)
            combo.currentIndexChanged.connect(lambda _i, c=cid, box=combo: self.mapping.__setitem__(c, box.currentData()) if box.currentData() != "unset" else self.mapping.pop(c, None))
            self.table.setCellWidget(row, 2, combo)
        self.prepare_btn.setEnabled(True)

    def _prepare(self):
        if not self.source or not self.model_classes or self.task != "detect": return
        if self.train.value() + self.val.value() + self.test.value() != 100:
            QMessageBox.warning(self, "Invalid split", "Split percentages must total 100."); return
        if set(self.mapping) != set(range(len(self.source["classes"]))):
            QMessageBox.warning(self, "Mapping required", "Map or explicitly exclude every dataset class."); return
        output, _ = QFileDialog.getSaveFileName(self, "Choose separate fine-tuning dataset folder", str(Path(self.source["root"]).parent / "finetune_dataset"))
        if not output: return
        source = dict(self.source); source["enabled"] = True
        source["selected_images"] = [record["path"] for record in source["records"]]
        try:
            analysis = analyze_merge([source], self.target_classes, {str(Path(source["root"]).resolve()): self.mapping})
            plan = finalize_merge(analysis, Path(output), "keep_all", {},
                                 {"train": self.train.value(), "val": self.val.value(), "test": self.test.value()},
                                 self.seed.value(), self.group.isChecked())
        except Exception as exc:
            QMessageBox.warning(self, "Preparation validation failed", str(exc)); return
        changed = len(self.target_classes) != len(self.model_classes) or self.target_classes[:len(self.model_classes)] != self.model_classes
        impact = ("Class count/order changes: the model detection head needs adapting during fine-tuning; compatible pretrained weights will be transferred where supported."
                  if changed else "Class names/order match the inspected model.")
        if QMessageBox.question(self, "Confirm fine-tuning dataset", f"Create a separate dataset at:\n{output}\n\n{impact}\n\nThe source dataset and model weights remain unchanged. Continue?") != QMessageBox.Yes: return
        self.prepare_btn.setEnabled(False); self.cancel_btn.setEnabled(True)
        self.worker = _PrepareWorker(plan, self); self.worker.progress.connect(self._progress)
        self.worker.complete.connect(self._complete); self.worker.failed.connect(self._failed); self.worker.start()

    def _progress(self, done, total, message):
        self.progress.setValue(int(done * 100 / max(total, 1))); self.status.setText(message)

    def _complete(self, result):
        self.cancel_btn.setEnabled(False); self.prepare_btn.setEnabled(True)
        QMessageBox.information(self, "Fine-tuning dataset ready", f"Saved separate validated dataset:\n{result['output']}\n\nConfiguration: {Path(result['output']) / 'data.yaml'}")

    def _failed(self, message):
        self.cancel_btn.setEnabled(False); self.prepare_btn.setEnabled(True)
        QMessageBox.warning(self, "Preparation failed", message)

    def _cancel(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel_event.set(); self.status.setText("Cancellation requested; the worker stops safely between images.")

    def closeEvent(self, event):
        if (self.worker and self.worker.isRunning()) or (hasattr(self, "model_worker") and self.model_worker.isRunning()):
            event.ignore(); QMessageBox.information(self, "Operation in progress", "Wait for the background operation to finish or stop before closing."); return
        event.accept()
