from __future__ import annotations

import csv
import shutil
import urllib.request
from urllib.parse import urlparse
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QListWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QFileDialog, QLineEdit, QFormLayout, QComboBox, QDoubleSpinBox,
    QGroupBox, QCheckBox, QScrollArea, QTableWidget, QTableWidgetItem,
    QMessageBox, QAbstractItemView,
)


class TrainingResultsPanel(QWidget):
    """Browse Ultralytics run folders, metrics, and downloadable checkpoints."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.run_root = None
        layout = QVBoxLayout(self)
        bar = QHBoxLayout()
        self.root_label = QLabel("Open a dataset to browse its training runs.")
        self.root_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        bar.addWidget(self.root_label, 1)
        refresh = QPushButton("Refresh runs")
        refresh.clicked.connect(self.refresh_runs)
        bar.addWidget(refresh)
        layout.addLayout(bar)
        split = QSplitter(Qt.Horizontal)
        self.runs = QListWidget()
        self.runs.currentRowChanged.connect(self.show_run)
        split.addWidget(self.runs)
        right = QWidget()
        detail = QVBoxLayout(right)
        self.summary = QLabel("Select a run to inspect its metrics and checkpoints.")
        self.summary.setWordWrap(True)
        detail.addWidget(self.summary)
        self.metrics = QTableWidget(0, 0)
        self.metrics.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.metrics.setAlternatingRowColors(True)
        detail.addWidget(self.metrics, 1)
        actions = QHBoxLayout()
        self.best_button = QPushButton("Save best.pt…")
        self.best_button.clicked.connect(lambda: self.export_checkpoint("best.pt"))
        self.last_button = QPushButton("Save last.pt…")
        self.last_button.clicked.connect(lambda: self.export_checkpoint("last.pt"))
        self.open_model_button = QPushButton("Use best.pt for detection")
        self.open_model_button.clicked.connect(self.use_best_model)
        actions.addWidget(self.best_button)
        actions.addWidget(self.last_button)
        actions.addWidget(self.open_model_button)
        detail.addLayout(actions)
        split.addWidget(right)
        split.setSizes([260, 900])
        layout.addWidget(split, 1)

    def set_dataset_root(self, root):
        self.run_root = Path(root) / "runs" if root else None
        self.root_label.setText(str(self.run_root) if self.run_root else "Open a dataset to browse its training runs.")
        self.refresh_runs()

    def refresh_runs(self):
        self.runs.clear()
        if not self.run_root or not self.run_root.is_dir():
            self.show_empty()
            return
        found = sorted({p.parent for p in self.run_root.rglob("results.csv")}, key=lambda p: p.stat().st_mtime, reverse=True)
        for folder in found:
            self.runs.addItem(f"{folder.name}  —  {folder.parent.name}")
            self.runs.item(self.runs.count() - 1).setData(Qt.UserRole, str(folder))
        if self.runs.count():
            self.runs.setCurrentRow(0)
        else:
            self.show_empty()

    def show_empty(self):
        self.summary.setText("No training results.csv files found yet. Completed or in-progress Ultralytics runs appear here.")
        self.metrics.clear()
        self.metrics.setRowCount(0)
        self.metrics.setColumnCount(0)
        self.best_button.setEnabled(False)
        self.last_button.setEnabled(False)
        self.open_model_button.setEnabled(False)

    def show_run(self, row):
        if row < 0:
            self.show_empty()
            return
        folder = Path(self.runs.item(row).data(Qt.UserRole))
        csv_path = folder / "results.csv"
        try:
            with csv_path.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                raw_headers = list(reader.fieldnames or [])
                headers = [h.strip() for h in raw_headers]
                rows = [{str(k).strip(): v for k, v in row.items()} for row in rows]
        except OSError as exc:
            self.summary.setText(f"Could not read run metrics: {exc}")
            return
        headers = [h for h in headers if h]
        self.metrics.setColumnCount(len(headers))
        self.metrics.setHorizontalHeaderLabels(headers)
        self.metrics.setRowCount(len(rows))
        for r, record in enumerate(rows):
            for c, key in enumerate(headers):
                self.metrics.setItem(r, c, QTableWidgetItem(str(record.get(key, "")).strip()))
        self.metrics.resizeColumnsToContents()
        epoch = rows[-1].get("epoch", "?").strip() if rows else "0"
        best = folder / "weights" / "best.pt"
        last = folder / "weights" / "last.pt"
        status = "Run still in progress" if not best.exists() else "Best checkpoint available"
        latest = rows[-1] if rows else {}
        key_metrics = [(k, v) for k, v in latest.items()
                       if any(token in k.lower() for token in ("precision", "recall", "map50", "map50-95"))]
        metric_summary = " | ".join(f"{k.strip()}: {str(v).strip()}" for k, v in key_metrics)
        self.summary.setText(
            f"{folder}\n{status} | {len(rows)} recorded epochs (latest: {epoch})\n"
            f"Latest validation metrics: {metric_summary or 'not available yet'}\n"
            f"Metrics: {csv_path} | Best checkpoint: {'available' if best.exists() else 'not found'} | "
            f"Last checkpoint: {'available' if last.exists() else 'not found'}"
        )
        self.best_button.setEnabled(best.is_file())
        self.last_button.setEnabled(last.is_file())
        self.open_model_button.setEnabled(best.is_file())

    def selected_folder(self):
        item = self.runs.currentItem()
        return Path(item.data(Qt.UserRole)) if item else None

    def export_checkpoint(self, name):
        folder = self.selected_folder()
        source = folder / "weights" / name if folder else None
        if not source or not source.is_file():
            QMessageBox.warning(self, "Checkpoint missing", f"Could not find {name} for this run.")
            return
        target, _ = QFileDialog.getSaveFileName(self, "Save model checkpoint", name, "PyTorch checkpoint (*.pt)")
        if target:
            try:
                shutil.copy2(source, target)
                QMessageBox.information(self, "Checkpoint saved", f"Saved checkpoint to:\n{target}")
            except OSError as exc:
                QMessageBox.critical(self, "Save failed", str(exc))

    def use_best_model(self):
        folder = self.selected_folder()
        if folder:
            self.window().select_detection_model(folder / "weights" / "best.pt")


class ModelLoadWorker(QThread):
    loaded = Signal(object, object, str)
    failed = Signal(str)

    def __init__(self, path):
        super().__init__()
        self.path = str(path)

    def run(self):
        try:
            from ultralytics import YOLO
            model = YOLO(self.path)
            names = model.names or {}
            names = [str(names[k]) for k in sorted(names)] if isinstance(names, dict) else list(names)
            core = getattr(model, "model", None)
            params = sum(p.numel() for p in core.parameters()) if core is not None and hasattr(core, "parameters") else None
            task = getattr(model, "task", "unknown")
            stride = getattr(core, "stride", None)
            stride_text = str(stride.tolist() if hasattr(stride, "tolist") else stride)
            checkpoint = getattr(model, "ckpt", None) or {}
            train_args = checkpoint.get("train_args") or checkpoint.get("args") or {}
            epoch = checkpoint.get("epoch", "unknown")
            fitness = checkpoint.get("best_fitness", "unknown")
            architecture = getattr(core, "yaml", {})
            architecture = architecture.get("yaml_file", architecture.get("scale", "unknown")) if isinstance(architecture, dict) else str(architecture)
            file = Path(self.path)
            detail = (
                f"File: {file}\nSize: {file.stat().st_size / (1024 * 1024):.2f} MB\n"
                f"Task: {task}\nClasses: {len(names)}\nParameters: {params:,} ({params / 1e6:.2f} M)\n"
                f"Architecture: {architecture}\nStride: {stride_text}\n"
                f"Checkpoint epoch: {epoch}\nBest fitness: {fitness}\n"
                f"Training args: {train_args}\nUltralytics: {__import__('ultralytics').__version__}"
                if params is not None else f"File: {file}\nTask: {task}\nClasses: {len(names)}\nStride: {stride_text}\nTraining args: {train_args}"
            )
            if names:
                detail += "\nClass names: " + ", ".join(names)
            self.loaded.emit(model, names, detail)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class DetectWorker(QThread):
    frame_ready = Signal(object)
    message = Signal(str)
    done = Signal()
    failed = Signal(str)

    def __init__(self, model, source_kind, source, class_ids, confidence, device, source_mode="File", output_video=None):
        super().__init__()
        self.model = model
        self.source_kind = source_kind
        self.source = source
        self.source_mode = source_mode
        self.class_ids = class_ids
        self.confidence = confidence
        self.device = device
        self.output_video = output_video
        self._stop_requested = False
        self.writer = None

    def stop(self):
        self._stop_requested = True

    def run(self):
        cap = None
        try:
            if self.source_kind == "Image":
                if self.source_mode == "URL":
                    self.message.emit("Loading photo from URL…")
                    frame = self.read_remote_image(self.source)
                else:
                    frame = cv2.imread(str(self.source))
                if frame is None:
                    raise RuntimeError(f"Could not read image: {self.source}")
                self.process_frame(frame)
                self.done.emit()
                return
            video_source = int(self.source) if self.source_kind == "Camera" else str(self.source)
            if self.source_kind == "Video" and self.source_mode == "URL":
                self.message.emit("Resolving video link…")
                video_source = self.resolve_video_url(self.source)
            cap = cv2.VideoCapture(video_source)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open {self.source_kind.lower()}: {self.source}")
            if self.source_kind == "Video" and self.output_video:
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                self.video_fps = fps if fps > 0 else 25.0
            self.message.emit(f"Detecting from {self.source_kind.lower()}…")
            while not self._stop_requested:
                ok, frame = cap.read()
                if not ok:
                    break
                self.process_frame(frame)
            self.done.emit()
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            if cap is not None:
                cap.release()
            if self.writer is not None:
                self.writer.release()

    @staticmethod
    def validate_web_url(url):
        parsed = urlparse(url.strip())
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("Enter a complete http:// or https:// link.")
        return url.strip()

    @classmethod
    def read_remote_image(cls, url):
        url = cls.validate_web_url(url)
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 YOLO-Dataset-Studio"})
        with urllib.request.urlopen(request, timeout=30) as response:
            length = response.headers.get("Content-Length")
            if length and int(length) > 50 * 1024 * 1024:
                raise ValueError("Photo is larger than the 50 MB URL limit.")
            data = response.read(50 * 1024 * 1024 + 1)
        if len(data) > 50 * 1024 * 1024:
            raise ValueError("Photo is larger than the 50 MB URL limit.")
        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("The URL did not return a supported image. Use a direct image link.")
        return frame

    @classmethod
    def resolve_video_url(cls, url):
        url = cls.validate_web_url(url)
        try:
            import yt_dlp
        except ImportError as exc:
            raise RuntimeError("Video links need yt-dlp. Install project dependencies with: python -m pip install -r requirements.txt") from exc
        options = {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "format": "best[protocol^=http][vcodec!=none]/best[protocol^=http]/best",
        }
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=False)
        if info and info.get("entries"):
            info = next((entry for entry in info["entries"] if entry), None)
        if not info or not info.get("url"):
            raise RuntimeError("Could not find a playable video stream at that link.")
        headers = info.get("http_headers") or {}
        if headers:
            header_block = "\r\n".join(f"{key}: {value}" for key, value in headers.items()) + "\r\n"
            return f"{info['url']}|{header_block}"
        return info["url"]

    def process_frame(self, frame):
        result = self.model.predict(
            source=frame, classes=self.class_ids or None,
            conf=self.confidence, device=self.device, verbose=False,
        )[0]
        annotated = result.plot()
        if self.source_kind == "Video" and self.output_video:
            if self.writer is None:
                h, w = annotated.shape[:2]
                self.writer = cv2.VideoWriter(
                    self.output_video, cv2.VideoWriter_fourcc(*"mp4v"),
                    self.video_fps, (w, h),
                )
                if not self.writer.isOpened():
                    raise RuntimeError(f"Could not create annotated video: {self.output_video}")
            self.writer.write(annotated)
        self.frame_ready.emit(annotated)


class InferencePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = None
        self.names = []
        self.loader = None
        self.worker = None
        self.last_frame = None
        layout = QVBoxLayout(self)
        model_group = QGroupBox("Detection model")
        mf = QFormLayout(model_group)
        row = QHBoxLayout()
        self.model_path = QLineEdit()
        self.model_path.setPlaceholderText("Choose a trained or external .pt checkpoint")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self.browse_model)
        row.addWidget(self.model_path, 1)
        row.addWidget(browse)
        self.load_button = QPushButton("Load model and inspect")
        self.load_button.clicked.connect(self.load_model)
        row.addWidget(self.load_button)
        mf.addRow("Checkpoint:", row)
        self.model_details = QLabel("No model loaded")
        self.model_details.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.model_details.setWordWrap(True)
        mf.addRow(self.model_details)
        layout.addWidget(model_group)

        split = QSplitter(Qt.Horizontal)
        controls = QWidget()
        cl = QVBoxLayout(controls)
        input_group = QGroupBox("Input")
        inf = QFormLayout(input_group)
        self.input_type = QComboBox()
        self.input_type.addItems(["Image", "Video", "Camera"])
        self.input_type.currentTextChanged.connect(self.update_input_controls)
        inf.addRow("Source type:", self.input_type)
        self.source_mode = QComboBox()
        self.source_mode.addItems(["File", "URL"])
        self.source_mode.currentTextChanged.connect(self.update_input_controls)
        inf.addRow("Source location:", self.source_mode)
        self.source_row = QWidget()
        sr = QHBoxLayout(self.source_row)
        sr.setContentsMargins(0, 0, 0, 0)
        self.source_path = QLineEdit()
        self.source_path.setPlaceholderText("Select an image or video")
        self.source_browse = QPushButton("Browse…")
        self.source_browse.clicked.connect(self.browse_source)
        self.camera_combo = QComboBox()
        self.camera_combo.addItems([f"Camera {i}" for i in range(8)])
        sr.addWidget(self.source_path, 1)
        sr.addWidget(self.source_browse)
        sr.addWidget(self.camera_combo)
        inf.addRow("Source:", self.source_row)
        self.video_output = QLineEdit()
        self.video_output.setPlaceholderText("Optional: save annotated video")
        self.video_output_browse = QPushButton("Save as…")
        self.video_output_browse.clicked.connect(self.browse_video_output)
        video_row = QHBoxLayout()
        video_row.addWidget(self.video_output, 1)
        video_row.addWidget(self.video_output_browse)
        inf.addRow("Video output:", video_row)
        self.confidence = QDoubleSpinBox()
        self.confidence.setRange(0.01, 0.99)
        self.confidence.setSingleStep(0.05)
        self.confidence.setValue(0.25)
        inf.addRow("Confidence:", self.confidence)
        self.device_combo = QComboBox()
        self.device_combo.addItem("Auto", "")
        self.device_combo.addItem("CPU", "cpu")
        for i in range(8):
            self.device_combo.addItem(f"CUDA {i}", str(i))
        inf.addRow("Device:", self.device_combo)
        cl.addWidget(input_group)
        cl.addWidget(QLabel("Detect these classes (select none for all):"))
        self.class_scroll = QScrollArea()
        self.class_scroll.setWidgetResizable(True)
        self.class_container = QWidget()
        self.class_layout = QVBoxLayout(self.class_container)
        self.class_scroll.setWidget(self.class_container)
        cl.addWidget(self.class_scroll, 1)
        self.run_button = QPushButton("Run detection")
        self.run_button.clicked.connect(self.start_detection)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_detection)
        self.stop_button.setEnabled(False)
        self.save_button = QPushButton("Save annotated image…")
        self.save_button.clicked.connect(self.save_image)
        cl.addWidget(self.run_button)
        cl.addWidget(self.stop_button)
        cl.addWidget(self.save_button)
        split.addWidget(controls)
        display = QWidget()
        dl = QVBoxLayout(display)
        self.preview = QLabel("Detection output appears here")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(500, 360)
        self.preview.setStyleSheet("background: #111; color: #ddd;")
        dl.addWidget(self.preview, 1)
        self.status = QLabel("Ready")
        dl.addWidget(self.status)
        split.addWidget(display)
        split.setSizes([390, 900])
        layout.addWidget(split, 1)
        self.update_input_controls()
        self.update_controls()

    def browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select YOLO checkpoint", "", "PyTorch model (*.pt);;All files (*.*)")
        if path:
            self.model_path.setText(path)
            self.load_model()

    def load_model(self):
        path = Path(self.model_path.text().strip())
        if not path.is_file():
            QMessageBox.warning(self, "Model required", "Choose an existing .pt model file first.")
            return
        self.load_button.setEnabled(False)
        self.model_details.setText("Loading checkpoint and reading model metadata…")
        self.loader = ModelLoadWorker(path)
        self.loader.loaded.connect(self.model_loaded)
        self.loader.failed.connect(self.model_failed)
        self.loader.start()

    def model_loaded(self, model, names, details):
        self.model = model
        self.names = names
        self.model_details.setText(details)
        while self.class_layout.count():
            item = self.class_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for name in self.names:
            check = QCheckBox(name)
            check.setChecked(True)
            self.class_layout.addWidget(check)
        self.class_layout.addStretch(1)
        self.load_button.setEnabled(True)
        self.status.setText(f"Loaded {len(names)} class(es).")
        self.update_controls()

    def model_failed(self, error):
        self.load_button.setEnabled(True)
        self.model = None
        self.model_details.setText("Could not load model.")
        self.status.setText(error)
        QMessageBox.critical(self, "Model load failed", error)

    def update_input_controls(self, *_):
        camera = self.input_type.currentText() == "Camera"
        is_url = self.source_mode.currentText() == "URL"
        self.source_path.setVisible(not camera)
        self.source_path.setPlaceholderText("Paste a direct photo/video URL" if is_url else "Select an image or video")
        self.source_browse.setVisible(not camera and not is_url)
        self.source_mode.setVisible(not camera)
        self.camera_combo.setVisible(camera)
        show_video_output = self.input_type.currentText() == "Video"
        self.video_output.setVisible(show_video_output)
        self.video_output_browse.setVisible(show_video_output)

    def browse_source(self):
        kind = self.input_type.currentText()
        filters = "Images (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff)" if kind == "Image" else "Videos (*.mp4 *.avi *.mov *.mkv *.webm)"
        path, _ = QFileDialog.getOpenFileName(self, f"Select {kind.lower()}", "", filters)
        if path:
            self.source_path.setText(path)

    def browse_video_output(self):
        default = "annotated_video.mp4"
        if self.source_mode.currentText() == "File" and self.source_path.text().strip():
            source = Path(self.source_path.text().strip())
            default = str(source.with_name(source.stem + "_detected.mp4"))
        path, _ = QFileDialog.getSaveFileName(self, "Save annotated video", default, "MP4 video (*.mp4)")
        if path:
            self.video_output.setText(path)

    def selected_classes(self):
        ids = []
        for i in range(self.class_layout.count()):
            widget = self.class_layout.itemAt(i).widget()
            if isinstance(widget, QCheckBox) and widget.isChecked():
                ids.append(i)
        return ids

    def start_detection(self):
        if self.model is None:
            QMessageBox.warning(self, "No model", "Load a model checkpoint first.")
            return
        kind = self.input_type.currentText()
        source_mode = "File" if kind == "Camera" else self.source_mode.currentText()
        source = self.camera_combo.currentIndex() if kind == "Camera" else self.source_path.text().strip()
        if not source and kind != "Camera":
            QMessageBox.warning(self, "No input", "Choose a file or paste a URL first.")
            return
        if source_mode == "URL":
            try:
                source = DetectWorker.validate_web_url(source)
            except ValueError as exc:
                QMessageBox.warning(self, "Invalid URL", str(exc))
                return
        self.worker = DetectWorker(
            self.model, kind, source, self.selected_classes(),
            self.confidence.value(), self.device_combo.currentData(), source_mode,
            self.video_output.text().strip() if kind == "Video" and self.video_output.text().strip() else None,
        )
        self.worker.frame_ready.connect(self.show_frame)
        self.worker.message.connect(self.status.setText)
        self.worker.failed.connect(self.detection_failed)
        self.worker.done.connect(self.detection_done)
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(kind != "Image")
        self.worker.start()

    def stop_detection(self):
        if self.worker:
            self.worker.stop()
            self.stop_button.setEnabled(False)

    def show_frame(self, frame):
        self.last_frame = frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        self.preview.setPixmap(QPixmap.fromImage(image).scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def detection_failed(self, error):
        self.status.setText(error)
        QMessageBox.critical(self, "Detection failed", error)

    def detection_done(self):
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status.setText("Detection finished.")

    def save_image(self):
        if self.last_frame is None:
            QMessageBox.information(self, "No output", "Run image detection first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save annotated image", "detection.jpg", "JPEG image (*.jpg);;PNG image (*.png)")
        if path and not cv2.imwrite(path, self.last_frame):
            QMessageBox.critical(self, "Save failed", f"Could not save image to:\n{path}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.last_frame is not None:
            self.show_frame(self.last_frame)

    def update_controls(self):
        self.run_button.setEnabled(self.model is not None)

    def closeEvent(self, event):
        self.stop_and_wait()
        super().closeEvent(event)

    def stop_and_wait(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        if self.loader and self.loader.isRunning():
            self.loader.wait()
