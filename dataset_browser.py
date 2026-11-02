"""Searchable dataset browser with composable filters and annotation batch edits."""
from pathlib import Path

import cv2
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QIcon
from PySide6.QtWidgets import (
    QDialog, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QLabel, QListWidget,
    QListWidgetItem, QSplitter, QLineEdit, QComboBox, QPushButton, QMessageBox,
    QAbstractItemView, QSpinBox, QCheckBox, QFormLayout,
)

from class_manager import remove_class_with_empty_decision


class DatasetBrowserDialog(QDialog):
    def __init__(self, manager, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.root = Path(manager.info.root).resolve()
        self.records = manager.image_records()
        self.index = []
        self._index_dataset()
        self.setWindowTitle(f"Dataset Browser — {self.root.name}")
        self.resize(1360, 850)
        layout = QVBoxLayout(self)
        filters = QGridLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Filename contains…")
        self.format_filter = QComboBox()
        self.format_filter.addItem("All formats", "")
        for ext in sorted({row["image"].suffix.lower() for row in self.index}):
            self.format_filter.addItem(ext or "(no extension)", ext)
        filters.addWidget(QLabel("Filename"), 0, 0); filters.addWidget(self.search, 0, 1, 1, 3)
        filters.addWidget(QLabel("Format"), 0, 4); filters.addWidget(self.format_filter, 0, 5)
        self.min_w, self.max_w = self._range_spin(), self._range_spin(999999)
        self.min_h, self.max_h = self._range_spin(), self._range_spin(999999)
        self.min_count, self.max_count = self._range_spin(), self._range_spin(999999)
        for label, low, high, row, column in (
            ("Width min / max", self.min_w, self.max_w, 1, 0),
            ("Height min / max", self.min_h, self.max_h, 1, 2),
            ("Annotations min / max", self.min_count, self.max_count, 1, 4),
        ):
            pair = QHBoxLayout(); pair.addWidget(low); pair.addWidget(QLabel("to")); pair.addWidget(high)
            filters.addWidget(QLabel(label), row, column)
            filters.addLayout(pair, row, column + 1)
        layout.addLayout(filters)

        classes = QHBoxLayout()
        self.include_container, self.include_classes = self._class_selector("Require all selected classes")
        self.exclude_container, self.exclude_classes = self._class_selector("Exclude selected classes")
        classes.addWidget(self.include_container, 1)
        classes.addWidget(self.exclude_container, 1)
        class_tools = QVBoxLayout()
        self.only_selected_classes = QCheckBox("Only these classes (no others)")
        self.only_selected_classes.setToolTip("When classes are required above, exclude any image containing additional classes.")
        class_tools.addWidget(self.only_selected_classes)
        self.annotation_filter = QComboBox()
        self.annotation_filter.addItems(["Any annotation state", "Has annotation file", "No annotation file", "Has boxes", "No boxes"])
        class_tools.addWidget(self.annotation_filter)
        self.class_match_count = QLabel("Class filters require every checked class; exclusions override includes.")
        self.class_match_count.setWordWrap(True)
        class_tools.addWidget(self.class_match_count)
        class_tools.addStretch(1)
        classes.addLayout(class_tools, 1)
        layout.addLayout(classes)

        split = QSplitter(Qt.Horizontal)
        self.images = QListWidget()
        self.images.setViewMode(QListWidget.IconMode)
        self.images.setResizeMode(QListWidget.Adjust)
        self.images.setMovement(QListWidget.Static)
        self.images.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.images.setIconSize(QSize(132, 96)); self.images.setGridSize(QSize(170, 142))
        self.images.currentItemChanged.connect(self._selection_changed)
        split.addWidget(self.images)
        panel = QVBoxLayout()
        self.preview = QLabel("Select an image")
        self.preview.setAlignment(Qt.AlignCenter); self.preview.setMinimumSize(440, 340)
        self.preview.setStyleSheet("background: #111; color: #ddd;")
        panel.addWidget(self.preview, 1)
        self.details = QLabel(""); self.details.setWordWrap(True)
        self.details.setTextInteractionFlags(Qt.TextSelectableByMouse)
        panel.addWidget(self.details)
        self.annotations = QListWidget()
        self.annotations.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.annotations.itemSelectionChanged.connect(self._render_current)
        panel.addWidget(QLabel("Annotations — select one or more boxes for a batch action"))
        panel.addWidget(self.annotations, 1)
        right = QLabel(); right.setLayout(panel)
        split.addWidget(right); split.setSizes([650, 600])
        layout.addWidget(split, 1)

        actions = QGridLayout()
        self.target_class = QComboBox()
        for cid, name in enumerate(manager.classes):
            self.target_class.addItem(f"{cid} | {name}", cid)
        self.affected = QLabel("0 annotations selected")
        assign = QPushButton("Reassign selected boxes…")
        assign.clicked.connect(self.reassign_annotations)
        delete_boxes = QPushButton("Delete selected boxes…")
        delete_boxes.clicked.connect(self.delete_annotations)
        delete_images = QPushButton("Delete selected images…")
        delete_images.clicked.connect(self.delete_selected_images)
        remove_class = QPushButton("Remove class from selected images…")
        remove_class.clicked.connect(self.remove_filtered_class)
        actions.addWidget(QLabel("Set selected boxes to"), 0, 0)
        actions.addWidget(self.target_class, 0, 1)
        actions.addWidget(assign, 0, 2)
        actions.addWidget(delete_boxes, 0, 3)
        actions.addWidget(self.affected, 0, 4)
        actions.addWidget(delete_images, 1, 2)
        actions.addWidget(remove_class, 1, 3)
        layout.addLayout(actions)
        self.count = QLabel(); layout.addWidget(self.count)

        for widget in (self.search, self.format_filter, self.min_w, self.max_w, self.min_h,
                       self.max_h, self.min_count, self.max_count, self.only_selected_classes,
                       self.annotation_filter):
            signal = widget.textChanged if isinstance(widget, QLineEdit) else (
                widget.valueChanged if isinstance(widget, QSpinBox) else
                widget.stateChanged if isinstance(widget, QCheckBox) else widget.currentIndexChanged)
            signal.connect(self.refresh)
        for selector in (self.include_classes, self.exclude_classes):
            selector.itemChanged.connect(self.refresh)
        self.refresh()

    @staticmethod
    def _range_spin(value=0):
        spin = QSpinBox(); spin.setRange(0, 999999); spin.setMaximumWidth(110)
        spin.setValue(value)
        return spin

    def _class_selector(self, title):
        widget = QWidget(); holder = QVBoxLayout(widget); holder.addWidget(QLabel(title))
        selector = QListWidget(); selector.setMaximumHeight(105)
        for cid, name in enumerate(self.manager.classes):
            item = QListWidgetItem(f"{cid} | {name}")
            item.setData(Qt.UserRole, cid); item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked); selector.addItem(item)
        holder.addWidget(selector)
        return widget, selector

    def _index_dataset(self):
        for image, label, split in self.records:
            frame = cv2.imread(str(image))
            width = height = 0
            boxes = []
            if frame is not None:
                height, width = frame.shape[:2]
                try:
                    boxes = self.manager.boxes_for_image(image, width, height)
                except Exception:
                    pass
            self.index.append({"image": image, "label": label, "split": split,
                               "width": width, "height": height, "boxes": boxes,
                               "class_ids": {int(b.get("class_id", -1)) for b in boxes},
                               "has_file": self.manager.has_annotations(image),
                               "format": image.suffix.lower()})

    @staticmethod
    def _checked(selector):
        return {selector.item(i).data(Qt.UserRole) for i in range(selector.count())
                if selector.item(i).checkState() == Qt.Checked}

    def _matches(self, row):
        image = row["image"]
        query = self.search.text().casefold().strip()
        if query and query not in image.name.casefold(): return False
        if self.format_filter.currentData() and row["format"] != self.format_filter.currentData(): return False
        if row["width"] < self.min_w.value() or row["width"] > self.max_w.value(): return False
        if row["height"] < self.min_h.value() or row["height"] > self.max_h.value(): return False
        count = len(row["boxes"])
        if count < self.min_count.value() or count > self.max_count.value(): return False
        include, exclude = self._checked(self.include_classes), self._checked(self.exclude_classes)
        ids = row["class_ids"]
        if not include.issubset(ids) or ids.intersection(exclude): return False
        if self.only_selected_classes.isChecked() and include and ids != include: return False
        mode = self.annotation_filter.currentIndex()
        if mode == 1 and not row["has_file"]: return False
        if mode == 2 and row["has_file"]: return False
        if mode == 3 and count == 0: return False
        if mode == 4 and count != 0: return False
        return True

    def refresh(self, *_):
        current_path = self.images.currentItem().data(Qt.UserRole) if self.images.currentItem() else None
        self.images.clear()
        matches = [row for row in self.index if self._matches(row)]
        for row in matches:
            item = QListWidgetItem(f"{row['image'].name}\n{row['width']}×{row['height']} · {len(row['boxes'])} boxes")
            item.setData(Qt.UserRole, str(row["image"]))
            frame = cv2.imread(str(row["image"]))
            if frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                q = QImage(rgb.data, row["width"], row["height"], rgb.strides[0], QImage.Format_RGB888).copy()
                item.setIcon(QIcon(QPixmap.fromImage(q).scaled(132, 96, Qt.KeepAspectRatio, Qt.SmoothTransformation)))
            self.images.addItem(item)
            if str(row["image"]) == current_path:
                self.images.setCurrentItem(item)
        if not self.images.currentItem() and self.images.count(): self.images.setCurrentRow(0)
        if not matches:
            self.preview.setPixmap(QPixmap()); self.preview.setText("No images match these filters.")
            self.details.clear(); self.annotations.clear()
        includes = self._checked(self.include_classes)
        excludes = self._checked(self.exclude_classes)
        self.count.setText(f"{len(matches)} matching image(s) of {len(self.index)}")
        self.class_match_count.setText(f"Require all: {', '.join(self.manager.classes[i] for i in sorted(includes)) or 'none'} · "
                                       f"Exclude: {', '.join(self.manager.classes[i] for i in sorted(excludes)) or 'none'}")

    def _row_for(self, path):
        return next((row for row in self.index if str(row["image"]) == str(path)), None)

    def _selection_changed(self, current, _previous):
        if not current: return
        row = self._row_for(current.data(Qt.UserRole))
        if not row: return
        self.annotations.blockSignals(True); self.annotations.clear()
        for index, box in enumerate(row["boxes"]):
            cid = int(box.get("class_id", -1))
            name = self.manager.classes[cid] if 0 <= cid < len(self.manager.classes) else f"Invalid class {cid}"
            x1, y1, x2, y2 = map(int, box["bbox"])
            item = QListWidgetItem(f"Box {index + 1} · {name} · ({x1},{y1})–({x2},{y2})")
            item.setData(Qt.UserRole, index)
            self.annotations.addItem(item)
        self.annotations.blockSignals(False)
        self._render_current()

    def _render_current(self):
        item = self.images.currentItem()
        if not item: return
        path = Path(item.data(Qt.UserRole)); row = self._row_for(path)
        frame = cv2.imread(str(path))
        if frame is None or not row:
            self.preview.setText("Image could not be read."); return
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimage = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy()
        selected = {int(i.data(Qt.UserRole)) for i in self.annotations.selectedItems()}
        painter = QPainter(qimage)
        for index, box in enumerate(row["boxes"]):
            x1, y1, x2, y2 = map(int, box["bbox"])
            painter.setPen(QPen(QColor(255, 150, 30) if index in selected else QColor(40, 220, 100), max(2, int(min(w, h) / 300))) )
            painter.drawRect(x1, y1, x2-x1, y2-y1)
            cid = int(box.get("class_id", -1)); name = self.manager.classes[cid] if 0 <= cid < len(self.manager.classes) else f"Invalid {cid}"
            painter.drawText(x1, max(16, y1-4), name)
        painter.end()
        self.preview.setPixmap(QPixmap.fromImage(qimage).scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        size = path.stat().st_size if path.is_file() else 0
        names = sorted(self.manager.classes[c] for c in row["class_ids"] if 0 <= c < len(self.manager.classes))
        self.details.setText(f"{path}\n{w} × {h} px · {size / (1024 * 1024):.2f} MB · {len(row['boxes'])} annotations · "
                             f"classes: {', '.join(names) if names else 'none'}")
        self.affected.setText(f"{len(self.annotations.selectedItems())} annotations selected")

    def _selected_annotation_indexes(self):
        return sorted(int(item.data(Qt.UserRole)) for item in self.annotations.selectedItems())

    def _edit_selected(self, delete=False):
        image_item = self.images.currentItem(); indexes = self._selected_annotation_indexes()
        if not image_item or not indexes:
            QMessageBox.information(self, "Select annotations", "Select one or more annotation rows first."); return
        action = "Delete" if delete else "Reassign"
        target = self.target_class.currentData()
        target_name = self.manager.classes[target] if target is not None else ""
        answer = QMessageBox.question(self, f"{action} annotations",
            f"{action} {len(indexes)} selected box(es)" + (f" to {target_name}?" if not delete else "?") +
            "\n\nAll unselected boxes will be preserved. A recovery snapshot will be saved.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes: return
        try:
            self.manager.edit_annotations(Path(image_item.data(Qt.UserRole)), indexes,
                                          class_id=None if delete else target, delete=delete)
            self._index_dataset(); self.refresh()
        except Exception as exc:
            QMessageBox.critical(self, "Annotation edit failed", str(exc))

    def reassign_annotations(self): self._edit_selected(False)
    def delete_annotations(self): self._edit_selected(True)

    def delete_selected_images(self):
        paths = [Path(item.data(Qt.UserRole)) for item in self.images.selectedItems()]
        if not paths:
            QMessageBox.information(self, "No images selected", "Select one or more images first."); return
        total = sum(len((self._row_for(path) or {}).get("boxes", [])) for path in paths)
        answer = QMessageBox.question(self, "Delete selected images",
            f"Delete {len(paths)} images and their {total} annotation(s)?\nA recovery snapshot will be saved.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes: return
        try:
            self.manager.delete_images(paths)
            self.records = self.manager.image_records(); self._index_dataset(); self.refresh()
        except Exception as exc:
            QMessageBox.critical(self, "Delete failed", str(exc))

    def remove_filtered_class(self):
        includes = self._checked(self.include_classes)
        paths = [Path(item.data(Qt.UserRole)) for item in self.images.selectedItems()]
        if len(includes) != 1 or not paths:
            QMessageBox.information(self, "Choose one class", "Check exactly one required class and select images."); return
        cid = next(iter(includes))
        try:
            result = remove_class_with_empty_decision(self.manager, cid, paths, self)
            if result is not None:
                QMessageBox.information(self, "Annotations removed",
                    f"Removed {result['annotations']} annotation(s); deleted {result['deleted_images']} selected image(s).")
                self.records = self.manager.image_records(); self._index_dataset(); self.refresh()
        except Exception as exc:
            QMessageBox.critical(self, "Removal failed", str(exc))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render_current()
