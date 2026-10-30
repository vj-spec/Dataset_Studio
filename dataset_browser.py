"""A small file-browser window with image previews for the open dataset."""
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog, QFileSystemModel, QHBoxLayout, QLabel, QSplitter, QTreeView, QVBoxLayout,
)

from config import IMAGE_EXTENSIONS


class DatasetBrowserDialog(QDialog):
    def __init__(self, root: Path, parent=None):
        super().__init__(parent)
        self.root = Path(root).resolve()
        self.setWindowTitle(f"Dataset Files — {self.root.name}")
        self.resize(1050, 680)
        layout = QVBoxLayout(self)
        self.location = QLabel(str(self.root))
        self.location.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.location)
        split = QSplitter(Qt.Horizontal)
        self.model = QFileSystemModel(self)
        self.model.setRootPath(str(self.root))
        self.tree = QTreeView()
        self.tree.setModel(self.model)
        self.tree.setRootIndex(self.model.index(str(self.root)))
        self.tree.setAlternatingRowColors(True)
        self.tree.setColumnWidth(0, 330)
        for column in (1, 2, 3):
            self.tree.hideColumn(column)
        self.tree.selectionModel().currentChanged.connect(self._selection_changed)
        split.addWidget(self.tree)
        preview_panel = QVBoxLayout()
        self.preview = QLabel("Select an image to preview it")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(360, 300)
        self.preview.setStyleSheet("background: #111; color: #ddd;")
        preview_panel.addWidget(self.preview, 1)
        self.details = QLabel("")
        self.details.setWordWrap(True)
        self.details.setTextInteractionFlags(Qt.TextSelectableByMouse)
        preview_panel.addWidget(self.details)
        panel = QLabel()
        panel.setLayout(preview_panel)
        split.addWidget(panel)
        split.setSizes([520, 480])
        layout.addWidget(split, 1)

    def _selection_changed(self, current, _previous):
        path = Path(self.model.filePath(current))
        self.location.setText(str(path))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            pixmap = QPixmap(str(path))
            if not pixmap.isNull():
                self.preview.setPixmap(pixmap.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.details.setText(f"{path.name}\n{pixmap.width()} × {pixmap.height()} pixels\n{path.stat().st_size / (1024 * 1024):.2f} MB")
                return
        self.preview.setPixmap(QPixmap())
        if path.is_dir():
            self.preview.setText("Folder selected\nExpand it to browse files.")
            self.details.setText(str(path))
        else:
            self.preview.setText("Preview is available for supported image files.")
            self.details.setText(f"{path.name}\n{path.stat().st_size / (1024 * 1024):.2f} MB" if path.is_file() else "")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        index = self.tree.currentIndex()
        if index.isValid() and Path(self.model.filePath(index)).suffix.lower() in IMAGE_EXTENSIONS:
            self._selection_changed(index, index)
