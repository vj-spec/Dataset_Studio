from __future__ import annotations

import sys
from pathlib import Path

import cv2
from PySide6.QtCore import Qt, Signal, QRectF
from PySide6.QtGui import QImage, QPainter, QPen, QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QListWidget,
    QListWidgetItem, QVBoxLayout, QHBoxLayout, QSplitter, QGroupBox,
    QComboBox, QInputDialog, QMessageBox, QFileDialog, QSizePolicy,
    QSpinBox, QDoubleSpinBox, QLineEdit, QFormLayout, QTabWidget, QProgressBar
)

from config import MODEL_PRESETS, YOLO_FAMILIES
from dataset_manager import DatasetManager, DatasetError
from hardware import detect_hardware, cuda_install_command
from yolo_trainer import TrainingWorker, install_cuda_pytorch
from project_views import TrainingResultsPanel, InferencePanel

STYLESHEET = """
QLabel#datasetLabel { font-weight: 600; padding: 4px 2px; }
QLabel#sectionTitle { font-weight: 600; padding: 2px 0; }
QLabel#hint { color: palette(mid); }
QGroupBox { font-weight: 600; margin-top: 10px; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px; }
QPushButton { padding: 5px 10px; }
QPushButton#primary {
    background: #2563eb; color: white; font-weight: 600;
    border: none; border-radius: 4px;
}
QPushButton#primary:hover { background: #1d4ed8; }
QPushButton#primary:disabled { background: #94a3b8; }
"""

class AnnotationCanvas(QWidget):
    boxesChanged = Signal()
    selectionChanged = Signal(int)
    classNumberPressed = Signal(int)

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.frame = None
        self.qimage = None
        self.boxes = []
        self.selected_box = -1
        self.class_names = []
        self.zoom = 1.0
        self.pan_x = self.pan_y = 0.0
        self.action = None
        self.action_box = -1
        self.action_handle = None
        self.action_start = None
        self.original_bbox = None
        self.new_box_start = None
        self.last_mouse = None

    def set_class_names(self, names):
        self.class_names = list(names)
        self.update()

    def set_image(self, image, name=""):
        self.frame = image.copy() if image is not None else None
        self.boxes = []
        self.selected_box = -1
        self.action = None
        self.new_box_start = None
        if self.frame is not None:
            rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            self.qimage = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        else:
            self.qimage = None
        self.reset_view()
        self.selectionChanged.emit(-1)
        self.update()

    def set_boxes(self, boxes):
        self.boxes = [{"bbox": tuple(b["bbox"]), "class_id": b.get("class_id")} for b in boxes]
        self.selected_box = -1
        self.selectionChanged.emit(-1)
        self.update()

    def reset_view(self):
        self.zoom = 1.0
        self.pan_x = self.pan_y = 0.0
        self.update()

    def current_scale(self):
        if self.qimage is None:
            return 1.0
        fit = min(self.width() / max(1, self.qimage.width()), self.height() / max(1, self.qimage.height()))
        return max(0.001, fit * self.zoom)

    def image_offset(self, scale=None):
        if self.qimage is None:
            return 0.0, 0.0
        scale = self.current_scale() if scale is None else scale
        return (
            (self.width() - self.qimage.width() * scale) / 2 + self.pan_x,
            (self.height() - self.qimage.height() * scale) / 2 + self.pan_y,
        )

    def zoom_at(self, factor, x, y):
        if self.qimage is None:
            return
        old = self.current_scale()
        ox, oy = self.image_offset(old)
        ix, iy = (x - ox) / old, (y - oy) / old
        self.zoom = max(0.25, min(8.0, self.zoom * factor))
        new = self.current_scale()
        self.pan_x = x - (self.width() - self.qimage.width() * new) / 2 - ix * new
        self.pan_y = y - (self.height() - self.qimage.height() * new) / 2 - iy * new
        self.update()

    def wheelEvent(self, event):
        p = event.position()
        self.zoom_at(1.15 if event.angleDelta().y() > 0 else 1 / 1.15, p.x(), p.y())

    def widget_to_image(self, x, y):
        if self.qimage is None:
            return None
        scale = self.current_scale()
        ox, oy = self.image_offset(scale)
        return (
            max(0, min(self.qimage.width() - 1, (x - ox) / scale)),
            max(0, min(self.qimage.height() - 1, (y - oy) / scale)),
        )

    def image_to_widget(self, x, y):
        scale = self.current_scale()
        ox, oy = self.image_offset(scale)
        return ox + x * scale, oy + y * scale

    def hit_test(self, p):
        if self.frame is None:
            return -1, None
        for i in range(len(self.boxes) - 1, -1, -1):
            x1, y1, x2, y2 = self.boxes[i]["bbox"]
            x, y = p
            tol = max(6.0, 10.0 / max(0.001, self.current_scale()))
            handles = {
                "nw": (x1, y1), "n": ((x1+x2)/2, y1), "ne": (x2, y1),
                "e": (x2, (y1+y2)/2), "se": (x2, y2), "s": ((x1+x2)/2, y2),
                "sw": (x1, y2), "w": (x1, (y1+y2)/2),
            }
            for name, (hx, hy) in handles.items():
                if abs(x-hx) <= tol and abs(y-hy) <= tol:
                    return i, name
            if x1 <= x <= x2 and y1 <= y <= y2:
                return i, "inside"
        return -1, None

    def mousePressEvent(self, event):
        if self.qimage is None:
            return
        self.setFocus()
        if event.button() == Qt.MiddleButton:
            self.action = "pan"
            self.last_mouse = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            return
        if event.button() != Qt.LeftButton:
            return
        p = self.widget_to_image(event.position().x(), event.position().y())
        idx, hit = self.hit_test(p)
        if idx >= 0:
            self.selected_box = idx
            self.selectionChanged.emit(idx)
            self.action_box = idx
            self.action_start = p
            self.original_bbox = tuple(self.boxes[idx]["bbox"])
            self.action = "move" if hit == "inside" else "resize"
            self.action_handle = None if hit == "inside" else hit
        else:
            self.selected_box = -1
            self.selectionChanged.emit(-1)
            self.action = "create"
            self.new_box_start = p
            self.last_mouse = p
        self.update()

    def mouseMoveEvent(self, event):
        if self.qimage is None:
            return
        if self.action == "pan":
            cur = event.position()
            if self.last_mouse is not None:
                d = cur - self.last_mouse
                self.pan_x += d.x()
                self.pan_y += d.y()
            self.last_mouse = cur
            self.update()
            return
        p = self.widget_to_image(event.position().x(), event.position().y())
        if p is None:
            return
        if self.action == "create":
            self.last_mouse = p
            self.update()
            return
        if self.action == "move" and self.action_box >= 0:
            x1, y1, x2, y2 = self.original_bbox
            dx, dy = p[0]-self.action_start[0], p[1]-self.action_start[1]
            bw, bh = x2-x1, y2-y1
            nx1 = max(0, min(self.qimage.width()-bw, x1+dx))
            ny1 = max(0, min(self.qimage.height()-bh, y1+dy))
            self.boxes[self.action_box]["bbox"] = (nx1, ny1, nx1+bw, ny1+bh)
            self.boxesChanged.emit()
            self.update()
            return
        if self.action == "resize" and self.action_box >= 0:
            x1, y1, x2, y2 = self.original_bbox
            h = self.action_handle
            if "w" in h: x1 = p[0]
            if "e" in h: x2 = p[0]
            if "n" in h: y1 = p[1]
            if "s" in h: y2 = p[1]
            x1 = max(0, min(self.qimage.width()-1, x1))
            x2 = max(0, min(self.qimage.width()-1, x2))
            y1 = max(0, min(self.qimage.height()-1, y1))
            y2 = max(0, min(self.qimage.height()-1, y2))
            if abs(x2-x1) >= 4 and abs(y2-y1) >= 4:
                self.boxes[self.action_box]["bbox"] = (min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2))
                self.boxesChanged.emit()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.action = None
            self.last_mouse = None
            return
        if event.button() != Qt.LeftButton:
            return
        if self.action == "create" and self.new_box_start is not None:
            p = self.widget_to_image(event.position().x(), event.position().y())
            if p is not None:
                x1, y1 = min(self.new_box_start[0], p[0]), min(self.new_box_start[1], p[1])
                x2, y2 = max(self.new_box_start[0], p[0]), max(self.new_box_start[1], p[1])
                if x2-x1 >= 8 and y2-y1 >= 8:
                    self.boxes.append({"bbox": (x1,y1,x2,y2), "class_id": None})
                    self.selected_box = len(self.boxes)-1
                    self.selectionChanged.emit(self.selected_box)
                    self.boxesChanged.emit()
        elif self.action in ("move", "resize"):
            self.boxesChanged.emit()
        self.action = None
        self.action_box = -1
        self.action_handle = None
        self.action_start = None
        self.original_bbox = None
        self.new_box_start = None
        self.update()

    def keyPressEvent(self, event):
        if Qt.Key_1 <= event.key() <= Qt.Key_9:
            self.classNumberPressed.emit(event.key() - Qt.Key_1)
            return
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            if 0 <= self.selected_box < len(self.boxes):
                self.boxes.pop(self.selected_box)
                self.selected_box = min(self.selected_box, len(self.boxes)-1)
                self.selectionChanged.emit(self.selected_box)
                self.boxesChanged.emit()
                self.update()
                return
        if event.key() == Qt.Key_0:
            self.reset_view()
            return
        super().keyPressEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(17,17,17))
        if self.qimage is None:
            painter.setPen(QColor(180,180,180))
            painter.drawText(self.rect(), Qt.AlignCenter, "No image selected")
            return
        scale = self.current_scale()
        ox, oy = self.image_offset(scale)
        painter.drawImage(QRectF(ox, oy, self.qimage.width()*scale, self.qimage.height()*scale), self.qimage)
        colors = [
            QColor(0,255,0), QColor(255,200,0), QColor(255,0,255),
            QColor(0,128,255), QColor(255,0,0), QColor(0,255,255)
        ]
        for i, box in enumerate(self.boxes):
            x1,y1,x2,y2 = box["bbox"]
            cid = box.get("class_id")
            color = QColor(255,165,0) if i == self.selected_box else (
                QColor(255,255,255) if cid is None else colors[cid % len(colors)]
            )
            sx1,sy1 = self.image_to_widget(x1,y1)
            sx2,sy2 = self.image_to_widget(x2,y2)
            painter.setPen(QPen(color, 3 if i == self.selected_box else 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(QRectF(sx1,sy1,sx2-sx1,sy2-sy1))
            if cid is None:
                label = f"BOX {i+1} | UNASSIGNED"
            elif 0 <= cid < len(self.class_names):
                label = f"BOX {i+1} | {self.class_names[cid]}"
            else:
                label = f"BOX {i+1} | CLASS {cid}"
            fm = painter.fontMetrics()
            tw,th = fm.horizontalAdvance(label)+8, fm.height()+4
            ly = max(th, sy1)
            painter.setPen(Qt.NoPen); painter.setBrush(color)
            painter.drawRect(QRectF(sx1, ly-th, tw, th))
            painter.setPen(QColor(0,0,0))
            painter.drawText(QRectF(sx1+4,ly-th+2,tw-6,th-2), Qt.AlignLeft|Qt.AlignVCenter, label)
            if i == self.selected_box:
                painter.setBrush(color)
                for hx,hy in [(x1,y1),((x1+x2)/2,y1),(x2,y1),(x2,(y1+y2)/2),
                              (x2,y2),((x1+x2)/2,y2),(x1,y2),(x1,(y1+y2)/2)]:
                    sx,sy = self.image_to_widget(hx,hy)
                    painter.drawRect(QRectF(sx-4,sy-4,8,8))
        if self.action == "create" and self.new_box_start and self.last_mouse:
            x1,y1=min(self.new_box_start[0],self.last_mouse[0]),min(self.new_box_start[1],self.last_mouse[1])
            x2,y2=max(self.new_box_start[0],self.last_mouse[0]),max(self.new_box_start[1],self.last_mouse[1])
            sx1,sy1=self.image_to_widget(x1,y1); sx2,sy2=self.image_to_widget(x2,y2)
            painter.setPen(QPen(QColor(0,255,255),2,Qt.DashLine)); painter.setBrush(Qt.NoBrush)
            painter.drawRect(QRectF(sx1,sy1,sx2-sx1,sy2-sy1))

class DrillAnnotationStudio(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Dataset & Training Studio")
        self.resize(1650, 950)
        self.manager = DatasetManager()
        self.current_image = None
        self.unsaved = False
        self.worker = None
        self.build_ui()
        self.detect_hardware_ui()

    @staticmethod
    def _button(text, slot, tooltip=None, primary=False):
        b = QPushButton(text)
        b.clicked.connect(slot)
        if tooltip:
            b.setToolTip(tooltip)
        if primary:
            b.setObjectName("primary")
            b.setMinimumHeight(38)
        return b

    def build_ui(self):
        self.setStyleSheet(STYLESHEET)
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 0)

        # Top bar: current dataset + dataset actions
        top = QHBoxLayout()
        self.dataset_label = QLabel("No dataset open")
        self.dataset_label.setObjectName("datasetLabel")
        self.dataset_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        top.addWidget(self.dataset_label, 1)
        top.addWidget(self._button("Open Dataset…", self.open_existing))
        top.addWidget(self._button("New Dataset…", self.create_new))
        top.addWidget(self._button("Refresh", self.refresh_dataset, "Reload the dataset from disk (F5)"))
        root.addLayout(top)

        self.workspace_tabs = QTabWidget()
        workspace = QWidget()
        workspace_layout = QVBoxLayout(workspace)
        workspace_layout.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_image_panel())
        splitter.addWidget(self._build_canvas_panel())
        splitter.addWidget(self._build_side_tabs())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([280, 1000, 380])
        workspace_layout.addWidget(splitter, 1)
        self.results_panel = TrainingResultsPanel()
        self.inference_panel = InferencePanel()
        self.workspace_tabs.addTab(workspace, "Dataset & Training")
        self.workspace_tabs.addTab(self.results_panel, "Training Results")
        self.workspace_tabs.addTab(self.inference_panel, "Detect")
        root.addWidget(self.workspace_tabs, 1)

        self.status = QLabel("Ready")
        self.statusBar().addWidget(self.status, 1)

        for keys, slot in [
            (QKeySequence.Save, self.save_annotation),
            ("D", self.next_image),
            ("A", self.previous_image),
            ("F5", self.refresh_dataset),
        ]:
            QShortcut(QKeySequence(keys), self, activated=slot)

        self.refresh_model_names()

    def _build_image_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        self.images_header = QLabel("Images")
        self.images_header.setObjectName("sectionTitle")
        layout.addWidget(self.images_header)
        self.image_list = QListWidget()
        self.image_list.currentRowChanged.connect(self.select_image)
        layout.addWidget(self.image_list, 1)
        return panel

    def _build_canvas_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = AnnotationCanvas()
        self.canvas.boxesChanged.connect(self.annotation_changed)
        self.canvas.selectionChanged.connect(self.refresh_box_list)
        self.canvas.classNumberPressed.connect(self.assign_class_by_number)
        layout.addWidget(self.canvas, 1)

        bar = QHBoxLayout()
        bar.addWidget(self._button("◀ Prev", self.previous_image, "Previous image (A)"))
        bar.addWidget(self._button("Next ▶", self.next_image, "Next image (D)"))
        bar.addWidget(self._button("Fit", self.canvas.reset_view, "Reset zoom and pan (0)"))
        self.file_label = QLabel("No image selected")
        self.file_label.setObjectName("hint")
        self.file_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        bar.addWidget(self.file_label, 1)
        layout.addLayout(bar)

        hint = QLabel(
            "Drag: new box  ·  Drag box/handle: move/resize  ·  1–9: assign class  ·  "
            "Del: delete box  ·  Wheel: zoom  ·  Middle-drag: pan  ·  Ctrl+S: save"
        )
        hint.setObjectName("hint")
        layout.addWidget(hint)
        return panel

    def _build_side_tabs(self):
        tabs = QTabWidget()
        tabs.addTab(self._build_annotate_tab(), "Annotate")
        tabs.addTab(self._build_prepare_tab(), "Prepare")
        tabs.addTab(self._build_train_tab(), "Train")
        tabs.setMinimumWidth(320)
        return tabs

    def _build_annotate_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        classes = QGroupBox("Classes")
        cl = QVBoxLayout(classes)
        self.class_combo = QComboBox()
        cl.addWidget(self.class_combo)
        row = QHBoxLayout()
        row.addWidget(self._button("Add…", self.add_class))
        row.addWidget(self._button("Delete", self.delete_class, "Delete class and rewrite all label IDs"))
        cl.addLayout(row)
        layout.addWidget(classes)

        boxes = QGroupBox("Boxes")
        bl = QVBoxLayout(boxes)
        self.box_list = QListWidget()
        self.box_list.currentRowChanged.connect(self.select_box)
        bl.addWidget(self.box_list, 1)
        bl.addWidget(self._button("Assign Class to Box", self.assign_class,
                                  "Assign the class above to the selected box (or press 1–9)"))
        row = QHBoxLayout()
        row.addWidget(self._button("Delete Box", self.delete_box, "Delete selected box (Del)"))
        row.addWidget(self._button("Clear All", self.clear_boxes))
        bl.addLayout(row)
        layout.addWidget(boxes, 1)

        layout.addWidget(self._button("Save Annotation", self.save_annotation, "Ctrl+S", primary=True))
        return tab

    def _build_prepare_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        group = QGroupBox("YOLO Dataset Preparation")
        form = QFormLayout(group)

        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Choose an empty output folder")
        row = QHBoxLayout()
        row.addWidget(self.output_edit)
        row.addWidget(self._button("Browse…", self.browse_output))
        form.addRow("Output:", row)

        self.val_spin = QDoubleSpinBox()
        self.val_spin.setRange(0.05, 0.5)
        self.val_spin.setSingleStep(0.05)
        self.val_spin.setValue(0.2)
        form.addRow("Validation split:", self.val_spin)

        form.addRow(self._button("Prepare YOLO Dataset", self.prepare_dataset, primary=True))
        layout.addWidget(group)
        layout.addStretch(1)
        return tab

    def _build_train_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        model = QGroupBox("Model")
        mf = QFormLayout(model)
        self.family_combo = QComboBox()
        self.family_combo.addItems(YOLO_FAMILIES)
        self.family_combo.currentTextChanged.connect(self.refresh_model_names)
        mf.addRow("Family:", self.family_combo)
        self.model_combo = QComboBox()
        mf.addRow("Preset:", self.model_combo)
        self.model_path = QLineEdit()
        self.model_path.setPlaceholderText("Optional .pt/.yaml (overrides preset)")
        row = QHBoxLayout()
        row.addWidget(self.model_path)
        row.addWidget(self._button("Browse…", self.browse_model))
        mf.addRow("Local file:", row)
        layout.addWidget(model)

        hw = QGroupBox("Hardware")
        hf = QFormLayout(hw)
        self.device_combo = QComboBox()
        hf.addRow("Device:", self.device_combo)
        self.device_info = QLabel("Detecting hardware…")
        self.device_info.setObjectName("hint")
        self.device_info.setWordWrap(True)
        hf.addRow(self.device_info)
        hf.addRow(self._button("Check / Install CUDA PyTorch", self.install_cuda))
        layout.addWidget(hw)

        params = QGroupBox("Parameters")
        pf = QFormLayout(params)
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 10000)
        self.epochs.setValue(100)
        pf.addRow("Epochs:", self.epochs)
        self.imgsz = QSpinBox()
        self.imgsz.setRange(32, 4096)
        self.imgsz.setSingleStep(32)
        self.imgsz.setValue(640)
        pf.addRow("Image size:", self.imgsz)
        self.batch = QSpinBox()
        self.batch.setRange(-1, 1024)
        self.batch.setValue(16)
        self.batch.setToolTip("-1 = auto batch size")
        pf.addRow("Batch:", self.batch)
        layout.addWidget(params)

        self.train_btn = self._button("Train YOLO", self.train, primary=True)
        layout.addWidget(self.train_btn)
        self.train_progress = QProgressBar()
        self.train_progress.setRange(0, 100)
        self.train_progress.setValue(0)
        self.train_progress.setFormat("Training progress: %p%")
        layout.addWidget(self.train_progress)
        layout.addStretch(1)
        return tab

    def detect_hardware_ui(self):
        self.hw=detect_hardware()
        self.device_combo.clear()
        self.device_combo.addItem("Auto", "auto")
        self.device_combo.addItem("CPU", "cpu")
        for g in self.hw["gpus"]:
            text=f"GPU {g.index}: {g.name}"
            if g.memory_mb: text += f" ({g.memory_mb/1024:.1f} GB)"
            self.device_combo.addItem(text, f"cuda:{g.index}")
        if self.hw["torch_cuda_available"] and self.hw["gpus"]:
            self.device_combo.setCurrentIndex(2 if self.device_combo.count()>2 else 0)
        if self.hw["nvidia_gpu_detected"] and not self.hw["torch_cuda_available"]:
            self.device_info.setText(
                "NVIDIA GPU detected, but PyTorch CUDA is unavailable. "
                "Use 'Check / Install CUDA PyTorch'. A system NVIDIA driver is still required."
            )
        elif self.hw["torch_cuda_available"]:
            self.device_info.setText(
                f"CUDA is ready. PyTorch {self.hw['torch_version']} | "
                f"CUDA runtime {self.hw['torch_cuda_version']}"
            )
        else:
            self.device_info.setText("No usable CUDA GPU detected. CPU training is available.")

    def open_existing(self):
        path=QFileDialog.getExistingDirectory(self,"Open Existing Dataset")
        if path: self.load_dataset(Path(path))

    def create_new(self):
        parent=QFileDialog.getExistingDirectory(self,"Choose Parent Folder")
        if not parent: return
        name,ok=QInputDialog.getText(self,"Create New Dataset","Dataset name:")
        if not ok or not name.strip(): return
        root=Path(parent)/name.strip()
        if root.exists():
            QMessageBox.warning(self,"Already Exists",f"Folder already exists:\n{root}"); return
        try:
            (root/"images").mkdir(parents=True)
            (root/"labels").mkdir()
            (root/"classes.txt").write_text("",encoding="utf-8")
            self.load_dataset(root)
        except OSError as exc:
            QMessageBox.critical(self,"Create Failed",str(exc))

    def load_dataset(self, path):
        try:
            self.manager.open(path)
            self.results_panel.set_dataset_root(self.manager.info.root)
            if not self.manager.classes:
                ids=self.manager.detect_class_ids()
                if ids:
                    names=[]
                    for cid in range(max(ids)+1):
                        name,ok=QInputDialog.getText(self,"Class Name",f"Enter name for class ID {cid}:")
                        if not ok or not name.strip():
                            raise DatasetError("Class setup cancelled.")
                        names.append(name)
                    self.manager.prepare_missing_classes(names)
                else:
                    QMessageBox.information(self,"No Classes Yet","No class names or labeled class IDs were found. Add classes before annotation/training.")
            self.dataset_label.setText(f"{self.manager.info.kind.upper()} | {self.manager.info.root}")
            self.canvas.set_class_names(self.manager.classes)
            self.refresh_class_combo()
            self.refresh_images()
            self.status.setText(f"Opened {self.manager.info.kind} dataset: {self.manager.info.root}")
        except Exception as exc:
            QMessageBox.critical(self,"Open Failed",str(exc))

    def refresh_dataset(self):
        if self.manager.info:
            self.load_dataset(self.manager.info.root)

    def refresh_images(self):
        self.image_list.clear()
        if not self.manager.info: return
        records=self.manager.image_records()
        labeled=sum(1 for _,label,_ in records if label and label.is_file())
        self.images_header.setText(f"Images  ({labeled}/{len(records)} labeled)")
        for image,label,split in records:
            mark="✓" if label and label.is_file() else "○"
            item=QListWidgetItem(f"{mark} [{split}] {image.name}")
            item.setData(Qt.UserRole,str(image))
            self.image_list.addItem(item)
        if self.image_list.count(): self.image_list.setCurrentRow(0)
        else: self.canvas.set_image(None)

    def select_image(self,row):
        if row<0: return
        item=self.image_list.item(row)
        if not item: return
        if self.unsaved:
            ans=QMessageBox.question(self,"Unsaved","Save current annotation?",QMessageBox.Save|QMessageBox.Discard|QMessageBox.Cancel,QMessageBox.Save)
            if ans==QMessageBox.Save:
                if not self.save_annotation(): return
            elif ans==QMessageBox.Cancel:
                return
        self.open_image(Path(item.data(Qt.UserRole)))

    def open_image(self,path):
        image=cv2.imread(str(path))
        if image is None:
            QMessageBox.warning(self,"Image Error",f"Could not read:\n{path}"); return
        self.current_image=path
        self.canvas.set_image(image,path.name)
        self.canvas.set_boxes(self.load_labels(path))
        self.file_label.setText(str(path))
        self.unsaved=False
        self.refresh_box_list()

    def load_labels(self,image_path):
        label=self.manager.label_path_for(image_path)
        if not label or not label.is_file(): return []
        image=cv2.imread(str(image_path))
        if image is None: return []
        h,w=image.shape[:2]; boxes=[]
        try: lines=label.read_text(encoding="utf-8").splitlines()
        except OSError: return []
        for line in lines:
            p=line.split()
            if len(p)!=5: continue
            try: cid=int(p[0]); cx,cy,bw,bh=map(float,p[1:])
            except ValueError: continue
            x1=max(0,min(w-1,int((cx-bw/2)*w))); y1=max(0,min(h-1,int((cy-bh/2)*h)))
            x2=max(0,min(w-1,int((cx+bw/2)*w))); y2=max(0,min(h-1,int((cy+bh/2)*h)))
            if x2>x1 and y2>y1:
                boxes.append({"bbox":(x1,y1,x2,y2),"class_id":cid if 0<=cid<len(self.manager.classes) else None})
        return boxes

    def save_annotation(self):
        if not self.current_image or not self.manager.info: return False
        if not self.manager.classes:
            QMessageBox.warning(self,"No Classes","Add a class before saving."); return False
        for b in self.canvas.boxes:
            if b.get("class_id") is None:
                QMessageBox.warning(self,"Unassigned Box","Assign a class to every box."); return False
        label=self.manager.label_path_for(self.current_image)
        if label is None:
            QMessageBox.warning(self,"Cannot Save","This image does not map to a writable labels/annotations folder.")
            return False
        label.parent.mkdir(parents=True,exist_ok=True)
        image=cv2.imread(str(self.current_image))
        h,w=image.shape[:2]; lines=[]
        for b in self.canvas.boxes:
            cid=int(b["class_id"]); x1,y1,x2,y2=b["bbox"]
            cx=((x1+x2)/2)/w; cy=((y1+y2)/2)/h; bw=(x2-x1)/w; bh=(y2-y1)/h
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        label.write_text("\n".join(lines)+("\n" if lines else ""),encoding="utf-8")
        self.unsaved=False; self.refresh_images()
        self.status.setText(f"Saved: {label}")
        return True

    def annotation_changed(self):
        self.unsaved=True; self.refresh_box_list()

    def refresh_box_list(self,index=-1):
        self.box_list.blockSignals(True); self.box_list.clear()
        for i,b in enumerate(self.canvas.boxes,1):
            cid=b.get("class_id"); name=self.manager.classes[cid] if cid is not None and cid<len(self.manager.classes) else "UNASSIGNED"
            self.box_list.addItem(f"BOX {i} | {name}")
        if 0<=self.canvas.selected_box<self.box_list.count(): self.box_list.setCurrentRow(self.canvas.selected_box)
        self.box_list.blockSignals(False)

    def select_box(self,row):
        if 0<=row<len(self.canvas.boxes):
            self.canvas.selected_box=row; self.canvas.selectionChanged.emit(row); self.canvas.update()

    def assign_class_by_number(self,cid):
        if 0<=cid<len(self.manager.classes) and self.canvas.boxes:
            idx=self.canvas.selected_box if self.canvas.selected_box>=0 else len(self.canvas.boxes)-1
            self.canvas.boxes[idx]["class_id"]=cid; self.canvas.selected_box=idx; self.canvas.boxesChanged.emit(); self.canvas.update()

    def assign_class(self):
        if not self.canvas.boxes: return
        cid=self.class_combo.currentIndex()
        if cid<0: return
        idx=self.canvas.selected_box if self.canvas.selected_box>=0 else len(self.canvas.boxes)-1
        self.canvas.boxes[idx]["class_id"]=cid; self.canvas.selected_box=idx; self.canvas.boxesChanged.emit(); self.canvas.update()

    def delete_box(self):
        i=self.canvas.selected_box
        if 0<=i<len(self.canvas.boxes):
            self.canvas.boxes.pop(i); self.canvas.selected_box=min(i,len(self.canvas.boxes)-1)
            self.canvas.boxesChanged.emit(); self.canvas.update()

    def clear_boxes(self):
        if self.canvas.boxes and QMessageBox.question(self,"Clear","Delete all boxes?",QMessageBox.Yes|QMessageBox.No)==QMessageBox.Yes:
            self.canvas.boxes.clear(); self.canvas.selected_box=-1; self.canvas.boxesChanged.emit(); self.canvas.update()

    def refresh_class_combo(self):
        self.class_combo.clear()
        for i,n in enumerate(self.manager.classes): self.class_combo.addItem(f"{i} | {n}",i)
        self.canvas.set_class_names(self.manager.classes)

    def add_class(self):
        name,ok=QInputDialog.getText(self,"Add Class","Class name:")
        if not ok or not name.strip(): return
        classes=self.manager.classes+[name]
        try: self.manager.set_classes(classes); self.refresh_class_combo()
        except Exception as exc: QMessageBox.warning(self,"Class Error",str(exc))

    def delete_class(self):
        row=self.class_combo.currentIndex()
        if row<0: return
        if QMessageBox.question(self,"Delete Class","Delete this class and rewrite label IDs?",QMessageBox.Yes|QMessageBox.No)==QMessageBox.No: return
        for label in self.manager._label_files():
            try:
                lines=[]
                for line in label.read_text(encoding="utf-8").splitlines():
                    p=line.split()
                    if len(p)!=5: continue
                    cid=int(p[0])
                    if cid==row: continue
                    if cid>row: cid-=1
                    lines.append(" ".join([str(cid)]+p[1:]))
                label.write_text("\n".join(lines)+("\n" if lines else ""),encoding="utf-8")
            except Exception as exc:
                QMessageBox.warning(self,"Label Error",f"{label}\n{exc}")
        classes=self.manager.classes[:row]+self.manager.classes[row+1:]
        self.manager.set_classes(classes); self.refresh_class_combo()
        if self.current_image: self.canvas.set_boxes(self.load_labels(self.current_image))

    def previous_image(self):
        r=self.image_list.currentRow()
        if r>0: self.image_list.setCurrentRow(r-1)

    def next_image(self):
        r=self.image_list.currentRow()
        if r>=0 and r<self.image_list.count()-1: self.image_list.setCurrentRow(r+1)

    def browse_output(self):
        path=QFileDialog.getExistingDirectory(self,"Choose YOLO Output Folder")
        if path: self.output_edit.setText(path)

    def prepare_dataset(self):
        if not self.manager.info:
            QMessageBox.warning(self,"No Dataset","Open or create a dataset first."); return
        if not self.output_edit.text().strip():
            QMessageBox.warning(self,"Output Required","Choose the YOLO output folder first."); return
        try:
            out=self.manager.prepare_yolo(Path(self.output_edit.text().strip()), self.val_spin.value())
            self.load_dataset(out)
            QMessageBox.information(self,"Preparation Complete",f"YOLO dataset created at:\n{out}")
        except Exception as exc:
            QMessageBox.critical(self,"Preparation Failed",str(exc))

    def refresh_model_names(self):
        family=self.family_combo.currentText()
        self.model_combo.clear()
        self.model_combo.addItems(MODEL_PRESETS.get(family, []))

    def browse_model(self):
        path,_=QFileDialog.getOpenFileName(self,"Select Model","", "Model files (*.pt *.yaml *.yml);;All files (*.*)")
        if path: self.model_path.setText(path)

    def selected_model(self):
        return self.model_path.text().strip() or self.model_combo.currentText()

    def install_cuda(self):
        hw=detect_hardware()
        if not hw["nvidia_gpu_detected"]:
            QMessageBox.information(self,"No NVIDIA GPU","No NVIDIA GPU was detected by nvidia-smi. CUDA PyTorch installation is not appropriate.")
            return
        if hw["torch_cuda_available"]:
            QMessageBox.information(self,"CUDA Ready",f"CUDA is already available.\n\nPyTorch: {hw['torch_version']}\nCUDA runtime: {hw['torch_cuda_version']}")
            return
        reply=QMessageBox.question(
            self,"Install CUDA PyTorch",
            "An NVIDIA GPU was detected, but this Python environment cannot use CUDA.\n\n"
            "The studio can install a CUDA-enabled PyTorch wheel into the CURRENT venv.\n"
            "This does not install the NVIDIA driver or system CUDA Toolkit.\n\n"
            f"Command:\n{cuda_install_command()}\n\nContinue?",
            QMessageBox.Yes|QMessageBox.No,QMessageBox.Yes)
        if reply!=QMessageBox.Yes: return
        self.train_btn.setEnabled(False)
        self.status.setText("Installing CUDA-enabled PyTorch in the current venv...")
        proc=install_cuda_pytorch()
        output=[]
        while True:
            line=proc.stdout.readline() if proc.stdout else ""
            if line: output.append(line.rstrip()); self.status.setText(line.rstrip()[-300:])
            elif proc.poll() is not None: break
        code=proc.wait()
        self.train_btn.setEnabled(True)
        if code==0:
            QMessageBox.information(self,"CUDA Installation Complete","PyTorch CUDA packages were installed.\n\nRestart the application so the new PyTorch runtime is loaded.")
        else:
            QMessageBox.critical(self,"CUDA Installation Failed","\n".join(output[-30:]))
        self.detect_hardware_ui()

    def train(self):
        if not self.manager.info:
            QMessageBox.warning(self,"No Dataset","Open a dataset first."); return
        data_yaml=self.manager.info.root/"data.yaml"
        if not data_yaml.is_file():
            QMessageBox.warning(self,"No data.yaml","Prepare a YOLO dataset first and select its output folder.")
            return
        model=self.selected_model()
        if not model:
            QMessageBox.warning(self,"Model Required","Choose a model."); return
        device=self.device_combo.currentData() or "auto"
        # Put training runs beside the selected YOLO dataset, not inside the source.
        run_root=data_yaml.parent/"runs"
        self.worker=TrainingWorker(model,data_yaml,device,self.epochs.value(),self.imgsz.value(),self.batch.value(),run_root)
        self.train_btn.setEnabled(False)
        self.worker.message.connect(lambda s:self.status.setText(s))
        self.worker.progress.connect(lambda s:self.status.setText(f"Training: {s}"))
        self.worker.epoch_progress.connect(self.update_training_progress)
        self.worker.finished_ok.connect(self.training_finished)
        self.worker.failed.connect(self.training_failed)
        self.worker.finished.connect(lambda:self.train_btn.setEnabled(True))
        self.worker.start()

    def training_finished(self,path):
        self.train_progress.setValue(100)
        self.results_panel.refresh_runs()
        self.status.setText(f"Training finished: {path}")
        QMessageBox.information(self,"Training Complete",f"Training finished.\n\n{path}")

    def training_failed(self,msg):
        self.results_panel.refresh_runs()
        self.status.setText("Training failed.")
        QMessageBox.critical(self,"Training Failed",msg)

    def select_detection_model(self, path):
        self.workspace_tabs.setCurrentWidget(self.inference_panel)
        self.inference_panel.model_path.setText(str(path))
        self.inference_panel.load_model()

    def update_training_progress(self, epoch, total, detail):
        self.train_progress.setRange(0, max(1, total))
        self.train_progress.setValue(epoch)
        self.status.setText(f"Training: epoch {epoch}/{total}" + (f" | {detail}" if detail else ""))

    def closeEvent(self,event):
        if self.unsaved:
            ans=QMessageBox.question(self,"Unsaved","Save current annotation before closing?",QMessageBox.Save|QMessageBox.Discard|QMessageBox.Cancel,QMessageBox.Save)
            if ans==QMessageBox.Save and not self.save_annotation(): event.ignore(); return
            if ans==QMessageBox.Cancel: event.ignore(); return
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "Training in progress", "Wait for training to finish before closing the studio.")
            event.ignore(); return
        self.inference_panel.stop_and_wait()
        event.accept()

def main():
    app=QApplication(sys.argv)
    win=DrillAnnotationStudio()
    win.show()
    sys.exit(app.exec())

if __name__=="__main__":
    main()
