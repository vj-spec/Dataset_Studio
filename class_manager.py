"""Dataset class tools and safe, recoverable annotation operations."""
from pathlib import Path
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem, QPushButton,
    QMessageBox, QInputDialog, QLabel, QTableWidget, QTableWidgetItem,
    QAbstractItemView, QHeaderView,
)


def remove_class_with_empty_decision(manager, class_id, images=None, parent=None):
    """Ask before changing labels and make the keep/delete/review choice explicit."""
    candidates = manager.images_that_become_unannotated(class_id, images)
    selected = {manager._resolve(path) for path in images} if images is not None else None
    import cv2
    affected_images = annotation_count = 0
    for image, _, _ in manager.image_records():
        if selected is not None and manager._resolve(image) not in selected:
            continue
        frame = cv2.imread(str(image))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        count = sum(int(box.get("class_id", -1)) == class_id
                    for box in manager.boxes_for_image(image, w, h))
        if count:
            affected_images += 1
            annotation_count += count
    prompt = QMessageBox(parent)
    prompt.setWindowTitle("Remove class annotations")
    prompt.setText(f"Remove {annotation_count} {manager.classes[class_id]} annotation(s) from {affected_images} image(s)?")
    if candidates:
        prompt.setInformativeText(f"{len(candidates)} image(s) will become unannotated. Choose how to handle them; no images are deleted automatically.")
    else:
        prompt.setInformativeText("The affected images will remain in the dataset.")
    prompt.setStandardButtons(QMessageBox.NoButton)
    keep = prompt.addButton("Keep images" if candidates else "Proceed", QMessageBox.AcceptRole)
    delete = prompt.addButton("Delete images", QMessageBox.DestructiveRole) if candidates else None
    review = prompt.addButton("Review individually", QMessageBox.ActionRole) if candidates else None
    cancel = prompt.addButton("Cancel", QMessageBox.RejectRole)
    prompt.exec()
    clicked = prompt.clickedButton()
    if clicked == cancel or clicked is None:
        return None
    delete_paths = []
    if clicked == delete and delete is not None:
        delete_paths = candidates
    elif clicked == review and review is not None:
        for path in candidates:
            choice = QMessageBox(parent)
            choice.setWindowTitle("Review unannotated image")
            choice.setText(str(path))
            choice.setInformativeText("Delete this image and its annotation file?")
            choice.setStandardButtons(QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            choice.setDefaultButton(QMessageBox.No)
            answer = choice.exec()
            if answer == QMessageBox.Yes:
                delete_paths.append(path)
            elif answer == QMessageBox.Cancel:
                return None
    result = manager.remove_class_annotations(class_id, images, delete_paths)
    result["empty_candidates"] = len(candidates)
    return result


class ClassManagerDialog(QDialog):
    def __init__(self, manager, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.setWindowTitle("Class Manager")
        self.resize(720, 540)
        layout = QVBoxLayout(self)
        self.summary = QLabel()
        layout.addWidget(self.summary)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["ID", "Class", "Images", "Annotations"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table, 1)
        row = QHBoxLayout()
        for title, callback in (("Add", self.add_class), ("Rename", self.rename_class),
                                ("Delete class…", self.delete_class),
                                ("Move up", lambda: self.move_class(-1)),
                                ("Move down", lambda: self.move_class(1)),
                                ("Replace annotations…", self.replace_class),
                                ("Remove annotations…", self.remove_annotations),
                                ("Validate dataset", self.validate), ("Undo last operation", self.undo)):
            button = QPushButton(title)
            button.clicked.connect(callback)
            row.addWidget(button)
        layout.addLayout(row)
        self.refresh()

    def selected_id(self):
        row = self.table.currentRow()
        return int(self.table.item(row, 0).text()) if row >= 0 else None

    def refresh(self):
        try:
            stats = self.manager.class_statistics()
        except Exception as exc:
            QMessageBox.warning(self, "Dataset summary", str(exc))
            return
        self.summary.setText(
            f"{stats['images']} images · {stats['annotations']} annotations · "
            f"{stats['labeled']} with boxes · {stats['missing']} without annotation files · "
            f"{len(stats['problems'])} image/annotation read issue(s)"
        )
        self.table.setRowCount(len(stats['classes']))
        for row, item in enumerate(stats['classes']):
            for column, value in enumerate((item['id'], item['name'], item['images'], item['annotations'])):
                self.table.setItem(row, column, QTableWidgetItem(str(value)))

    def add_class(self):
        name, ok = QInputDialog.getText(self, "Add class", "Class name:")
        if ok and name.strip():
            try:
                self.manager.set_classes(self.manager.classes + [name])
                self.refresh()
            except Exception as exc:
                QMessageBox.warning(self, "Add class", str(exc))

    def rename_class(self):
        cid = self.selected_id()
        if cid is None:
            return
        name, ok = QInputDialog.getText(self, "Rename class", "New class name:", text=self.manager.classes[cid])
        if not ok or not name.strip():
            return
        try:
            self.manager.rename_class(cid, name)
            self.refresh()
        except Exception as exc:
            QMessageBox.warning(self, "Rename class", str(exc))

    def replace_class(self):
        source = self.selected_id()
        if source is None or len(self.manager.classes) < 2:
            return
        options = [f"{i} | {name}" for i, name in enumerate(self.manager.classes) if i != source]
        selected, ok = QInputDialog.getItem(self, "Replace class annotations", "Replace with:", options, 0, False)
        if not ok:
            return
        target = int(selected.split(" | ", 1)[0])
        answer = QMessageBox.question(self, "Replace annotations",
            f"Replace every {self.manager.classes[source]} annotation with {self.manager.classes[target]}?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            return
        try:
            result = self.manager.replace_class(source, target)
            QMessageBox.information(self, "Replace complete", f"Updated {result['annotations']} annotations in {result['images']} images.")
            self.refresh()
        except Exception as exc:
            QMessageBox.critical(self, "Replace failed", str(exc))

    def delete_class(self):
        cid = self.selected_id()
        if cid is None:
            return
        if len(self.manager.classes) == 1:
            QMessageBox.information(self, "Keep one class", "A dataset must retain at least one class definition.")
            return
        name = self.manager.classes[cid]
        answer = QMessageBox.question(self, "Delete class",
            f"Delete class {name}, remove its annotations, and shift later class IDs?\n\n"
            "Images and other annotations will be preserved. A recovery snapshot will be saved.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            return
        try:
            removal = remove_class_with_empty_decision(self.manager, cid, parent=self)
            if removal is None:
                return
            self.manager.delete_class(cid)
            QMessageBox.information(self, "Class deleted",
                f"Deleted {name}. Removed {removal['annotations']} annotations across {removal['images']} images; "
                f"deleted {removal['deleted_images']} newly unannotated image(s).")
            self.refresh()
        except Exception as exc:
            QMessageBox.critical(self, "Delete failed", str(exc))

    def move_class(self, offset):
        cid = self.selected_id()
        if cid is None or not 0 <= cid + offset < len(self.manager.classes):
            return
        order = list(range(len(self.manager.classes)))
        other = cid + offset
        order[cid], order[other] = order[other], order[cid]
        try:
            self.manager.reorder_classes(order)
            self.refresh()
            self.table.selectRow(other)
        except Exception as exc:
            QMessageBox.critical(self, "Reorder failed", str(exc))

    def remove_annotations(self):
        cid = self.selected_id()
        if cid is None:
            return
        try:
            stats = self.manager.class_statistics()['classes'][cid]
        except Exception as exc:
            QMessageBox.warning(self, "Remove annotations", str(exc)); return
        answer = QMessageBox.question(self, "Remove class annotations",
            f"Remove {stats['annotations']} {self.manager.classes[cid]} annotations from {stats['images']} images?\n\n"
            "Images and all other class annotations will be preserved. A recovery snapshot will be saved.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            return
        try:
            result = remove_class_with_empty_decision(self.manager, cid, parent=self)
            if result is None:
                return
            QMessageBox.information(self, "Removal complete", f"Removed {result['annotations']} annotations from {result['images']} images; deleted {result['deleted_images']} images.")
            self.refresh()
        except Exception as exc:
            QMessageBox.critical(self, "Removal failed", str(exc))

    def validate(self):
        try:
            report = self.manager.validate_dataset()
            dialog = QDialog(self)
            dialog.setWindowTitle("Dataset Validation Report")
            dialog.resize(1050, 560)
            layout = QVBoxLayout(dialog)
            layout.addWidget(QLabel(f"{report['errors']} error(s) · {report['warnings']} warning(s) · "
                                    f"{len(report['affected_images'])} affected image(s) · "
                                    f"{len(report['affected_annotations'])} annotation file(s)"))
            table = QTableWidget(len(report['issues']), 5)
            table.setHorizontalHeaderLabels(["Severity", "Check", "Image", "Annotation", "Suggested correction"])
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
            table.horizontalHeader().setStretchLastSection(True)
            for row, issue in enumerate(report['issues']):
                for column, key in enumerate(("severity", "message", "image", "annotation", "suggestion")):
                    table.setItem(row, column, QTableWidgetItem(issue[key]))
            layout.addWidget(table, 1)
            buttons = QHBoxLayout()
            buttons.addStretch(1)
            missing = [Path(issue["image"]) for issue in report['issues']
                       if issue["code"] == "missing_annotation" and issue["image"]]
            if self.manager.info.annotation_format == "yolo" and missing:
                correction = QPushButton(f"Create {len(missing)} empty YOLO label(s)…")
                correction.clicked.connect(lambda: self._create_empty_labels(missing, dialog))
                buttons.addWidget(correction)
            close = QPushButton("Close"); close.clicked.connect(dialog.accept); buttons.addWidget(close)
            layout.addLayout(buttons)
            dialog.exec()
        except Exception as exc:
            QMessageBox.warning(self, "Dataset validation", str(exc))

    def _create_empty_labels(self, images, parent):
        answer = QMessageBox.question(parent, "Create empty labels",
            f"Create empty YOLO label files for {len(images)} images? This marks them as background images.\n\n"
            "A recovery snapshot will be saved. Review each image first if it may contain unlabeled objects.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            return
        try:
            count = self.manager.create_empty_yolo_labels(images)
            QMessageBox.information(parent, "Correction complete", f"Created {count} empty label file(s). Use Undo last operation to reverse this correction.")
            parent.accept()
            self.refresh()
        except Exception as exc:
            QMessageBox.critical(parent, "Correction failed", str(exc))

    def undo(self):
        try:
            message = self.manager.undo_last_operation()
            QMessageBox.information(self, "Undo complete", message)
            self.refresh()
        except Exception as exc:
            QMessageBox.warning(self, "Undo unavailable", str(exc))
