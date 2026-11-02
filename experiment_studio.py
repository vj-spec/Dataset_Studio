"""PySide6 experiment history, detail dashboard, comparisons and model registry."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QUrl, QThread, Signal
from PySide6.QtGui import QColor, QDesktopServices, QPainter, QPen
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QComboBox, QTableWidget, QTableWidgetItem, QTextEdit, QSplitter, QMessageBox, QInputDialog,
    QFileDialog, QListWidget, QListWidgetItem, QCheckBox, QFormLayout, QSpinBox, QProgressBar)

from experiment_database import ExperimentDatabase, normalize_metrics
from experiment_manager import ExperimentManager, ModelEvaluationWorker, InferenceBenchmarkWorker
from training_framework import TrainingConfig


class MetricChart(QWidget):
    """Lightweight Qt line chart; no new plotting dependency."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows = []; self.setMinimumHeight(180)

    def set_rows(self, rows):
        self.rows = list(rows or []); self.update()

    def paintEvent(self, _event):
        painter = QPainter(self); painter.fillRect(self.rect(), QColor("#ffffff")); painter.setRenderHint(QPainter.Antialiasing)
        bounds = self.rect().adjusted(42, 18, -12, -28)
        painter.setPen(QPen(QColor("#cbd5e1"), 1)); painter.drawRect(bounds)
        numeric = []
        for row in self.rows:
            for key, value in row.items():
                if key == "epoch": continue
                try: numeric.append((str(key), int(row["epoch"]), float(value)))
                except (TypeError, ValueError, KeyError): continue
        if not numeric:
            painter.setPen(QColor("#64748b")); painter.drawText(bounds, Qt.AlignCenter, "No epoch metrics are available."); return
        keys = list(dict.fromkeys(k for k, _x, _y in numeric))[:6]
        values = [v for key, _x, v in numeric if key in keys]
        ymin, ymax = min(values), max(values)
        if ymax == ymin: ymax = ymin + 1
        epochs = [x for _k, x, _v in numeric]
        xmin, xmax = min(epochs), max(epochs)
        if xmax == xmin: xmax = xmin + 1
        palette = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#ea580c", "#0891b2"]
        for index, key in enumerate(keys):
            points = [(x, y) for k, x, y in numeric if k == key]
            pen = QPen(QColor(palette[index % len(palette)]), 2); painter.setPen(pen)
            previous = None
            for x, y in points:
                px = bounds.left() + (x - xmin) / (xmax - xmin) * bounds.width()
                py = bounds.bottom() - (y - ymin) / (ymax - ymin) * bounds.height()
                point = (int(px), int(py))
                if previous: painter.drawLine(previous[0], previous[1], point[0], point[1])
                painter.drawEllipse(point[0] - 2, point[1] - 2, 4, 4); previous = point
            painter.drawText(48 + index * 135, self.height() - 5, key[:20])


class ComparisonChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self.values = []; self.setMinimumHeight(210)
    def set_values(self, values): self.values = list(values); self.update()
    def paintEvent(self, _event):
        painter = QPainter(self); painter.fillRect(self.rect(), QColor("white")); painter.setRenderHint(QPainter.Antialiasing)
        valid = [(label, value) for label, value in self.values if isinstance(value, (float, int))]
        if not valid:
            painter.setPen(QColor("#64748b")); painter.drawText(self.rect(), Qt.AlignCenter, "No comparable values for this metric."); return
        area = self.rect().adjusted(50, 16, -12, -34); maximum = max(abs(value) for _, value in valid) or 1
        width = max(14, int(area.width() / max(len(valid) * 1.6, 1)))
        for index, (label, value) in enumerate(valid):
            x = area.left() + int((index + .5) * area.width() / len(valid)) - width // 2
            height = int(abs(value) / maximum * max(area.height() - 20, 1)); y = area.bottom() - height
            painter.fillRect(x, y, width, height, QColor("#2563eb"))
            painter.setPen(QColor("#1e293b")); painter.drawText(x - 35, area.bottom() + 16, width + 70, 16, Qt.AlignHCenter, label[:14])
            painter.drawText(x - 15, max(area.top(), y - 18), width + 30, 16, Qt.AlignHCenter, f"{value:.3g}")


class ExperimentManagerPanel(QWidget):
    def __init__(self, database: ExperimentDatabase, training_manager, parent=None):
        super().__init__(parent)
        self.database, self.manager = database, training_manager
        self.service = ExperimentManager(database); self._workers = []
        self._imported_legacy_roots = set()
        self._build()
        self.refresh()
        self.timer = QTimer(self); self.timer.timeout.connect(self.refresh); self.timer.start(3000)

    def import_legacy_root(self, root):
        root = Path(root).expanduser()
        key = str(root.resolve())
        if key in self._imported_legacy_roots or not root.is_dir(): return 0
        self._imported_legacy_roots.add(key)
        count = self.service.import_legacy_metadata([root])
        if count: self.refresh()
        return count

    def _build(self):
        root = QVBoxLayout(self)
        bar = QHBoxLayout(); self.search = QLineEdit(); self.search.setPlaceholderText("Search experiments…")
        self.status_filter = QComboBox(); self.status_filter.addItems(["All", "Created", "Queued", "Preparing", "Running", "Validating", "Completed", "Failed", "Cancelled"])
        refresh = QPushButton("Refresh"); refresh.clicked.connect(self.refresh)
        for widget in (self.search, self.status_filter, refresh): bar.addWidget(widget)
        root.addLayout(bar)
        split = QSplitter(Qt.Horizontal)
        self.table = QTableWidget(0, 6); self.table.setHorizontalHeaderLabels(["Experiment", "Architecture", "Status", "Started", "Dataset", "Output"])
        self.table.setSelectionBehavior(QTableWidget.SelectRows); self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self.show_details); split.addWidget(self.table)
        pane = QWidget(); layout = QVBoxLayout(pane)
        self.title = QLabel("Select an experiment"); self.title.setStyleSheet("font-weight:600;font-size:16px")
        self.info = QTextEdit(); self.info.setReadOnly(True)
        self.chart = MetricChart()
        actions = QHBoxLayout()
        self.open_button = QPushButton("Open output folder"); self.open_button.clicked.connect(self.open_output)
        self.resume_button = QPushButton("Resume from checkpoint"); self.resume_button.clicked.connect(self.resume_selected)
        self.evaluate_button = QPushButton("Evaluate…"); self.evaluate_button.clicked.connect(self.evaluate_selected)
        self.benchmark_button = QPushButton("Benchmark…"); self.benchmark_button.clicked.connect(self.benchmark_selected)
        edit = QPushButton("Edit notes…"); edit.clicked.connect(self.edit_notes)
        delete = QPushButton("Delete metadata…"); delete.clicked.connect(self.delete_metadata)
        for button in (self.open_button, self.resume_button, self.evaluate_button, self.benchmark_button, edit, delete): actions.addWidget(button)
        layout.addWidget(self.title); layout.addWidget(self.info, 1); layout.addWidget(QLabel("Epoch metric history")); layout.addWidget(self.chart)
        layout.addLayout(actions); split.addWidget(pane); split.setSizes([520, 760]); root.addWidget(split, 1)
        self.search.textChanged.connect(self.refresh); self.status_filter.currentTextChanged.connect(self.refresh)
        self.progress = QProgressBar(); self.progress.setVisible(False); root.addWidget(self.progress)
        self.eval_status = QLabel(""); root.addWidget(self.eval_status)

    def refresh(self):
        selected = self.selected_id()
        records = self.database.list_experiments(self.search.text(), self.status_filter.currentText())
        self.table.setRowCount(len(records))
        for row, record in enumerate(records):
            values = [record["name"], record["architecture"], record["status"], record.get("start_time") or "",
                      record.get("dataset_ref") or "", record.get("output_dir") or ""]
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value)); item.setData(Qt.UserRole, record["id"]); self.table.setItem(row, col, item)
        self.table.resizeColumnsToContents()
        if selected:
            for row in range(self.table.rowCount()):
                if self.table.item(row, 0).data(Qt.UserRole) == selected:
                    self.table.selectRow(row); break

    def selected_id(self):
        row = self.table.currentRow() if hasattr(self, "table") else -1
        item = self.table.item(row, 0) if row >= 0 else None
        return item.data(Qt.UserRole) if item else None

    def selected_record(self):
        eid = self.selected_id(); return self.database.get_experiment(eid) if eid else None

    def show_details(self):
        record = self.selected_record()
        if not record: return
        self.title.setText(f"{record['name']} · {record['architecture']} · {record['status']}")
        artifacts = self.database.artifacts(record["id"])
        evaluation = self.database.latest_evaluation(experiment_id=record["id"])
        benchmark = self.database.latest_benchmark(experiment_id=record["id"])
        details = {"id": record["id"], "description": record["description"], "model": record["model_name"],
                   "initial weights": record["initial_weights"], "dataset": record["dataset_ref"],
                   "classes": record["classes"], "class mapping": record["class_mapping"],
                   "training configuration": record["config"], "hardware": record["hardware"],
                   "framework versions": record["versions"], "start": record["start_time"],
                   "end": record["end_time"], "duration seconds": record["duration_seconds"],
                   "best checkpoint": record["best_checkpoint"], "last checkpoint": record["last_checkpoint"],
                   "remote execution": record.get("metadata", {}).get("remote_job"),
                   "artifacts": [{"kind": a["kind"], "path": a["path"], "exists": Path(a["path"]).is_file()} for a in artifacts],
                   "latest evaluation": evaluation, "latest inference benchmark": benchmark,
                   "error": record["error"]}
        self.info.setPlainText(json.dumps(details, indent=2, ensure_ascii=False, default=str))
        rows = []
        for item in self.database.metrics(record["id"]):
            row = {"epoch": item["epoch"]}
            for key, value in item["normalized"].items():
                if isinstance(value, (int, float)): row[key] = value
                elif isinstance(value, dict):
                    for component, number in value.items():
                        if isinstance(number, (int, float)): row[f"{key}/{component}"] = number
            rows.append(row)
        self.chart.set_rows(rows)
        self.resume_button.setEnabled(record["status"] in {"failed", "cancelled", "running", "validating", "preparing"} and Path(record["last_checkpoint"]).is_file())
        self.open_button.setEnabled(bool(record["output_dir"]))

    def open_output(self):
        record = self.selected_record()
        if record and Path(record["output_dir"]).is_dir(): QDesktopServices.openUrl(QUrl.fromLocalFile(record["output_dir"]))
        elif record: QMessageBox.warning(self, "Output unavailable", "The experiment output folder was moved or is missing.")

    def edit_notes(self):
        record = self.selected_record()
        if not record: return
        text, ok = QInputDialog.getMultiLineText(self, "Experiment notes", "Description / notes:", record["description"])
        if ok: self.database.set_description(record["id"], text); self.refresh()

    def delete_metadata(self):
        record = self.selected_record()
        if not record: return
        active = bool(self.manager.worker and self.manager.worker.isRunning() and self.manager.metadata
                      and self.manager.metadata.get("experiment_id") == record["id"])
        active = active or any(worker.isRunning() and manager.metadata and manager.metadata.get("experiment_id") == record["id"]
                               for manager, worker in (getattr(self, "active_training_managers", {}) or {}).values())
        if active:
            QMessageBox.information(self, "Experiment is active", "Wait for this training job to stop before deleting its metadata.")
            return
        answer = QMessageBox.question(self, "Delete experiment metadata?",
            "This removes only its database record. Output folders and model files will remain untouched.")
        if answer == QMessageBox.Yes:
            self.service.delete_metadata(record["id"], confirmed=True); self.refresh()

    def resume_selected(self):
        record = self.selected_record()
        if not record: return
        config_data = dict(record["config"])
        config_data.update(initial_weights=record["last_checkpoint"], resume=True)
        try:
            config = TrainingConfig(**config_data)
            worker = self.manager.start(record["adapter"], config)
            self.eval_status.setText(f"Resume job started ({worker.objectName() or record['id'][:8]}).")
        except Exception as exc:
            QMessageBox.warning(self, "Could not resume experiment", str(exc))

    def evaluate_selected(self):
        record = self.selected_record()
        if not record: return
        dataset, _ = QFileDialog.getOpenFileName(self, "Select validation dataset YAML", "", "Dataset YAML (*.yaml *.yml)")
        checkpoint = record["best_checkpoint"] if Path(record["best_checkpoint"]).is_file() else record["last_checkpoint"]
        if not dataset or not Path(checkpoint).is_file(): return
        config = record["config"]
        worker = ModelEvaluationWorker(architecture=record["architecture"], checkpoint=checkpoint,
            dataset_yaml=dataset, device=config.get("device", "cpu"), image_size=config.get("image_size", 640),
            database=self.database, experiment_id=record["id"], parent=self)
        worker.progress.connect(self.eval_status.setText); worker.completed.connect(self._evaluation_complete); worker.failed.connect(self._job_failed)
        self._keep_worker(worker)

    def benchmark_selected(self):
        record = self.selected_record()
        if not record: return
        checkpoint = record["best_checkpoint"] if Path(record["best_checkpoint"]).is_file() else record["last_checkpoint"]
        if not Path(checkpoint).is_file(): QMessageBox.warning(self, "Checkpoint unavailable", "No checkpoint file is present."); return
        config = record["config"]
        worker = InferenceBenchmarkWorker(architecture=record["architecture"], checkpoint=checkpoint,
            dataset_yaml=record["dataset_ref"], device=config.get("device", "cpu"), image_size=config.get("image_size", 640),
            database=self.database, experiment_id=record["id"], parent=self)
        worker.progress.connect(lambda done, total, msg: self.eval_status.setText(f"{msg} ({done}/{total})"))
        worker.completed.connect(self._benchmark_complete); worker.failed.connect(self._job_failed)
        self._keep_worker(worker)

    def _keep_worker(self, worker):
        self._workers.append(worker); worker.finished.connect(lambda: self._workers.remove(worker) if worker in self._workers else None)
        self.eval_status.setText("Background job started…"); worker.start()

    def _evaluation_complete(self, result):
        self.eval_status.setText("Evaluation saved: " + json.dumps(result.get("metrics", {}), ensure_ascii=False)); self.refresh()

    def _benchmark_complete(self, result):
        self.eval_status.setText(f"Benchmark saved: {result['average_latency_ms']:.2f} ms · {result['throughput_fps']:.2f} FPS")

    def _job_failed(self, error):
        self.eval_status.setText("Background job failed: " + error); QMessageBox.warning(self, "Experiment job failed", error)


class ModelComparisonPanel(QWidget):
    METRICS = ["precision", "recall", "f1", "map50", "map50_95", "duration_seconds", "checkpoint_size_bytes", "inference_latency_ms", "throughput_fps"]

    def __init__(self, database: ExperimentDatabase, parent=None):
        super().__init__(parent); self.database = database; self._build(); self.refresh()

    def _build(self):
        root = QVBoxLayout(self); row = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh experiments"); self.refresh_button.clicked.connect(self.refresh)
        row.addWidget(self.refresh_button); row.addWidget(QLabel("Select completed runs; metrics unavailable on a run are marked N/A.")); root.addLayout(row)
        self.runs = QListWidget(); self.runs.setMaximumHeight(150); root.addWidget(self.runs)
        choices = QHBoxLayout(); self.metric_checks = {}
        for name in self.METRICS:
            check = QCheckBox(name.replace("_", " ")); check.setChecked(name in {"precision", "recall", "f1", "map50", "map50_95", "duration_seconds"})
            check.stateChanged.connect(self.compare); self.metric_checks[name] = check; choices.addWidget(check)
        root.addLayout(choices)
        chart_row = QHBoxLayout(); chart_row.addWidget(QLabel("Chart metric:")); self.chart_metric = QComboBox(); self.chart_metric.addItems(self.METRICS)
        self.chart_metric.currentTextChanged.connect(self.compare); chart_row.addWidget(self.chart_metric); root.addLayout(chart_row)
        self.warning = QLabel(""); self.warning.setWordWrap(True); root.addWidget(self.warning)
        self.table = QTableWidget(); root.addWidget(self.table, 1)
        self.chart = ComparisonChart(); root.addWidget(self.chart)
        self.runs.itemChanged.connect(self.compare)

    def refresh(self):
        self.runs.clear()
        self._records = {}
        for record in self.database.list_experiments():
            if record["status"] != "completed": continue
            item = QListWidgetItem(f"{record['name']} · {record['architecture']} · {record['start_time'] or ''}")
            key = ("experiment", record["id"]); item.setData(Qt.UserRole, key); self._records[key] = record
            self.runs.addItem(item)
        for model in self.database.list_models():
            key = ("model", model["id"])
            item = QListWidgetItem(f"[Registry] {model['name']} v{model['version']} · {model['architecture']}")
            item.setData(Qt.UserRole, key); self._records[key] = model; self.runs.addItem(item)
        for i in range(self.runs.count()):
            item = self.runs.item(i); item.setFlags(item.flags() | Qt.ItemIsUserCheckable); item.setCheckState(Qt.Unchecked)
        self.compare()

    def compare(self, *_args):
        keys = [self.runs.item(i).data(Qt.UserRole) for i in range(self.runs.count()) if self.runs.item(i).checkState() == Qt.Checked]
        records = [self._records[key] for key in keys]
        metrics = [key for key, check in self.metric_checks.items() if check.isChecked()]
        self.table.setRowCount(len(metrics)); self.table.setColumnCount(len(records) + 1)
        self.table.setHorizontalHeaderLabels(["Metric"] + [f"{r['name']} ({r['architecture']})" for r in records])
        for row, metric in enumerate(metrics):
            self.table.setItem(row, 0, QTableWidgetItem(metric))
            for col, record in enumerate(records, 1):
                value = self._metric(record, metric, keys[col - 1][0])
                self.table.setItem(row, col, QTableWidgetItem("N/A" if value is None else str(value)))
        self.table.resizeColumnsToContents()
        dataset_set = {r.get("dataset_ref", "") for r in records}; arch_set = {r["architecture"] for r in records}
        notes = []
        if len(dataset_set) > 1: notes.append("Datasets differ; quality metrics may not be comparable.")
        if len(arch_set) > 1: notes.append("Architectures differ; metric definitions may differ. Interpret side-by-side values with care.")
        benchmarks = [self.database.latest_benchmark(experiment_id=r["id"]) if key[0] == "experiment" else self.database.latest_benchmark(model_id=r["id"]) for key, r in zip(keys, records)]
        signatures = {(b["device"], b["input_size"], b["batch_size"],
                       b["settings"].get("hardware", {}).get("torch_version"),
                       tuple(gpu.get("name", "") for gpu in b["settings"].get("hardware", {}).get("gpus", [])))
                      for b in benchmarks if b}
        if len(signatures) > 1: notes.append("Inference benchmark settings differ; latency and throughput are not directly comparable.")
        self.warning.setText(" ".join(notes) if notes else "Comparisons are descriptive; no universal best model is selected.")
        selected_chart_metric = self.chart_metric.currentText() if hasattr(self, "chart_metric") else "map50"
        self.chart.set_values([(r["name"], self._metric(r, selected_chart_metric, key[0])) for key, r in zip(keys, records)])

    def _metric(self, record, name, kind="experiment"):
        if kind == "model":
            evaluation = self.database.latest_evaluation(model_id=record["id"])
            benchmark = self.database.latest_benchmark(model_id=record["id"])
            metrics = normalize_metrics(evaluation["metrics"] if evaluation else record.get("metrics", {}))
            if name == "duration_seconds":
                source = self.database.get_experiment(record["source_experiment_id"]) if record.get("source_experiment_id") else None
                return source["duration_seconds"] if source else None
            if name == "inference_latency_ms": return benchmark["average_latency_ms"] if benchmark else None
            if name == "throughput_fps": return benchmark["throughput_fps"] if benchmark else None
            if name == "checkpoint_size_bytes":
                path = record["checkpoint_path"]; return Path(path).stat().st_size if Path(path).is_file() else None
            return metrics.get(name)
        if name == "duration_seconds": return record["duration_seconds"]
        if name in ("inference_latency_ms", "throughput_fps"):
            benchmark = self.database.latest_benchmark(experiment_id=record["id"])
            return benchmark["average_latency_ms"] if name == "inference_latency_ms" and benchmark else benchmark["throughput_fps"] if benchmark else None
        if name == "checkpoint_size_bytes":
            path = record["best_checkpoint"]
            return Path(path).stat().st_size if Path(path).is_file() else None
        rows = self.database.metrics(record["id"])
        candidates = [item["normalized"].get(name) for item in rows if name in item["normalized"]]
        return candidates[-1] if candidates else None


class _CheckpointCopyWorker(QThread):
    completed = Signal(str); failed = Signal(str)
    def __init__(self, source, destination, parent=None):
        super().__init__(parent); self.source, self.destination = Path(source), Path(destination)
    def run(self):
        try:
            if self.destination.exists(): raise FileExistsError("Destination already exists; choose a new name to avoid overwriting.")
            shutil.copy2(self.source, self.destination); self.completed.emit(str(self.destination))
        except Exception as exc: self.failed.emit(str(exc))


class ModelRegistryPanel(QWidget):
    def __init__(self, database: ExperimentDatabase, parent=None):
        super().__init__(parent); self.database = database; self._workers = []; self._build(); self.refresh()

    def _build(self):
        root = QVBoxLayout(self); bar = QHBoxLayout(); self.search = QLineEdit(); self.search.setPlaceholderText("Search registered models…")
        refresh = QPushButton("Refresh"); refresh.clicked.connect(self.refresh); self.preferred = QPushButton("Mark preferred"); self.preferred.clicked.connect(self.set_preferred)
        register = QPushButton("Register from experiment…"); register.clicked.connect(self.register_from_experiment)
        export = QPushButton("Copy model…"); export.clicked.connect(self.copy_model)
        evaluate = QPushButton("Evaluate…"); evaluate.clicked.connect(self.evaluate_model)
        benchmark = QPushButton("Benchmark…"); benchmark.clicked.connect(self.benchmark_model)
        for widget in (self.search, refresh, register, self.preferred, evaluate, benchmark, export): bar.addWidget(widget)
        root.addLayout(bar)
        self.table = QTableWidget(0, 8); self.table.setHorizontalHeaderLabels(["Model", "Version", "Architecture", "Checkpoint", "Source experiment", "Classes", "Preferred", "Created"])
        self.table.setSelectionBehavior(QTableWidget.SelectRows); self.table.setEditTriggers(QTableWidget.NoEditTriggers); root.addWidget(self.table, 1)
        self.details = QTextEdit(); self.details.setReadOnly(True); root.addWidget(self.details); self.table.itemSelectionChanged.connect(self.show_details)
        self.search.textChanged.connect(self.refresh); self.status = QLabel(""); root.addWidget(self.status)

    def refresh(self):
        records = self.database.list_models(self.search.text()); self.table.setRowCount(len(records))
        for row, record in enumerate(records):
            values = [record["name"], record["version"], record["architecture"], record["checkpoint_path"],
                      record["source_experiment_id"] or "", ", ".join(record["classes"]), "Yes" if record["preferred"] else "", record["created_at"]]
            for col, value in enumerate(values):
                cell = QTableWidgetItem(str(value)); cell.setData(Qt.UserRole, record["id"]); self.table.setItem(row, col, cell)
        self.table.resizeColumnsToContents()

    def selected_id(self):
        item = self.table.item(self.table.currentRow(), 0) if self.table.currentRow() >= 0 else None
        return item.data(Qt.UserRole) if item else None

    def show_details(self):
        model_id = self.selected_id(); record = self.database.get_model(model_id) if model_id else None
        if record: self.details.setPlainText(json.dumps(record, indent=2, ensure_ascii=False, default=str))

    def register_from_experiment(self):
        experiments = [r for r in self.database.list_experiments() if r["status"] == "completed"]
        if not experiments: QMessageBox.information(self, "No completed experiments", "Complete a training run before registering a model."); return
        labels = [f"{r['name']} · {r['architecture']} · {r['id'][:8]}" for r in experiments]
        label, ok = QInputDialog.getItem(self, "Choose source experiment", "Experiment:", labels, 0, False)
        if not ok: return
        record = experiments[labels.index(label)]; checkpoint = record["best_checkpoint"] if Path(record["best_checkpoint"]).is_file() else record["last_checkpoint"]
        if not Path(checkpoint).is_file(): QMessageBox.warning(self, "Checkpoint missing", "The experiment's checkpoint is not available on disk."); return
        name, ok = QInputDialog.getText(self, "Register model", "Model name:", text=record["name"])
        if not ok or not name.strip(): return
        version, ok = QInputDialog.getText(self, "Model version", "Version (must be unique for this name):", text="1.0")
        if not ok or not version.strip(): return
        try:
            self.database.register_model(name=name, version=version, architecture=record["architecture"], checkpoint_path=checkpoint,
                source_experiment_id=record["id"], dataset_ref=record["dataset_ref"], classes=record["classes"],
                metrics=(self.database.metrics(record["id"])[-1]["raw"] if self.database.metrics(record["id"]) else {}))
            self.refresh()
        except Exception as exc: QMessageBox.warning(self, "Could not register model", str(exc))

    def set_preferred(self):
        model_id = self.selected_id()
        if model_id:
            try: self.database.set_preferred_model(model_id); self.refresh()
            except Exception as exc: QMessageBox.warning(self, "Could not update preferred model", str(exc))

    def evaluate_model(self):
        model_id = self.selected_id(); model = self.database.get_model(model_id) if model_id else None
        if not model: return
        dataset, _ = QFileDialog.getOpenFileName(self, "Select validation dataset YAML", "", "Dataset YAML (*.yaml *.yml)")
        if not dataset: return
        device, ok = QInputDialog.getItem(self, "Evaluation device", "Device:", ["cpu", "auto", "cuda:0"], 0, False)
        if not ok: return
        worker = ModelEvaluationWorker(architecture=model["architecture"], checkpoint=model["checkpoint_path"],
            dataset_yaml=dataset, device=device, database=self.database, model_id=model_id, parent=self)
        worker.progress.connect(self.status.setText); worker.completed.connect(lambda result: self.status.setText(f"Evaluation saved: {result['metrics']}"))
        worker.failed.connect(lambda error: QMessageBox.warning(self, "Evaluation failed", error))
        self._workers.append(worker); worker.finished.connect(lambda: self._workers.remove(worker) if worker in self._workers else None); worker.start()

    def benchmark_model(self):
        model_id = self.selected_id(); model = self.database.get_model(model_id) if model_id else None
        if not model: return
        dataset, _ = QFileDialog.getOpenFileName(self, "Select benchmark dataset YAML", "", "Dataset YAML (*.yaml *.yml)")
        if not dataset: return
        device, ok = QInputDialog.getItem(self, "Benchmark device", "Device:", ["cpu", "auto", "cuda:0"], 0, False)
        if not ok: return
        worker = InferenceBenchmarkWorker(architecture=model["architecture"], checkpoint=model["checkpoint_path"],
            dataset_yaml=dataset, device=device, database=self.database, model_id=model_id, parent=self)
        worker.progress.connect(lambda done, total, message: self.status.setText(f"{message} ({done}/{total})"))
        worker.completed.connect(lambda result: self.status.setText(f"Benchmark saved: {result['average_latency_ms']:.2f} ms, {result['throughput_fps']:.2f} FPS"))
        worker.failed.connect(lambda error: QMessageBox.warning(self, "Benchmark failed", error))
        self._workers.append(worker); worker.finished.connect(lambda: self._workers.remove(worker) if worker in self._workers else None); worker.start()

    def copy_model(self):
        model_id = self.selected_id(); record = self.database.get_model(model_id) if model_id else None
        if not record: return
        source = Path(record["checkpoint_path"])
        if not source.is_file(): QMessageBox.warning(self, "Checkpoint unavailable", "The registered artifact has moved or is missing."); return
        target, _ = QFileDialog.getSaveFileName(self, "Copy registered model", str(source.name), "Model files (*.pt *.pth *.onnx *.engine);;All files (*.*)")
        if not target: return
        target = Path(target)
        if target.exists(): QMessageBox.warning(self, "Destination exists", "Choose a new filename; registered model export never overwrites files."); return
        worker = _CheckpointCopyWorker(source, target, self); worker.completed.connect(lambda path: self.status.setText(f"Model copied to {path}"))
        worker.failed.connect(lambda error: QMessageBox.warning(self, "Model copy failed", error)); self._workers.append(worker)
        worker.finished.connect(lambda: self._workers.remove(worker) if worker in self._workers else None); worker.start()
