"""Temporal video annotation window for sawing-machine safety datasets."""
from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QListWidget, QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QSlider, QSplitter, QTextEdit, QVBoxLayout, QWidget, QInputDialog,
)
from app_state import load_state, save_state
from config import app_state_file


LEGACY_CLASSES = [
    "machine_with_guard",
    "machine_without_guard",
    "damaged_wire",
    "damaged_insulation",
]
COLORS = [(0, 255, 0), (0, 0, 255), (0, 165, 255), (0, 255, 255)]


def boxes_for_frame(tracks, frame_index):
    boxes = []
    for track_id, track in tracks.items():
        start, end = int(track["start"]), int(track["end"])
        if frame_index < start or frame_index > end:
            continue
        keys = sorted((int(k), box) for k, box in track["keyframes"].items())
        if not keys:
            continue
        if frame_index <= keys[0][0]:
            box = dict(keys[0][1])
        elif frame_index >= keys[-1][0]:
            box = dict(keys[-1][1])
        else:
            box = dict(keys[-1][1])
            for (f1, b1), (f2, b2) in zip(keys, keys[1:]):
                if f1 <= frame_index <= f2:
                    ratio = (frame_index - f1) / max(1, f2 - f1)
                    box = {key: b1[key] + (b2[key] - b1[key]) * ratio
                           for key in ("x1", "y1", "x2", "y2")}
                    break
        boxes.append({**box, "id": int(track["id"]), "class_id": int(track["class_id"]),
                      "class_name": track["class_name"]})
    return boxes


def render_annotated(frame, tracks, frame_index):
    output = frame.copy()
    for box in boxes_for_frame(tracks, frame_index):
        x1, y1, x2, y2 = (int(box[k]) for k in ("x1", "y1", "x2", "y2"))
        color = COLORS[box["class_id"] % len(COLORS)]
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = f'{box["class_name"]} ID:{box["id"]}'
        cv2.putText(output, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2, cv2.LINE_AA)
    return output


class ExportWorker(QThread):
    progress = Signal(int, str)
    completed = Signal(str)
    failed = Signal(str)

    def __init__(self, action, video_path, output_path, tracks, classes, fps, width, height):
        super().__init__()
        self.action = action
        self.video_path = str(video_path)
        self.output_path = str(output_path)
        self.tracks = copy.deepcopy(tracks)
        self.classes = list(classes)
        self.fps, self.width, self.height = fps, width, height

    def run(self):
        cap = None
        writer = None
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise RuntimeError("Could not reopen the source video for export.")
            total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            if self.action == "video":
                output = Path(self.output_path)
                writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), self.fps,
                                         (self.width, self.height))
                if not writer.isOpened():
                    raise RuntimeError(f"Could not create output video: {output}")
                frame_index = 0
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    writer.write(render_annotated(frame, self.tracks, frame_index))
                    frame_index += 1
                    if frame_index % 15 == 0:
                        self.progress.emit(int(frame_index * 100 / total), f"Exporting video: {frame_index}/{total} frames")
                writer.release()
                writer = None
            else:
                root = Path(self.output_path)
                images, labels = root / "images", root / "labels"
                images.mkdir(parents=True, exist_ok=True)
                labels.mkdir(parents=True, exist_ok=True)
                (root / "data.yaml").write_text(
                    "path: .\ntrain: images\nval: images\n\n"
                    f"nc: {len(self.classes)}\nnames:\n" + "".join(f"  {i}: {name}\n" for i, name in enumerate(self.classes)),
                    encoding="utf-8",
                )
                frame_index = 0
                saved = 0
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frame_boxes = boxes_for_frame(self.tracks, frame_index)
                    if frame_boxes:
                        stem = f"frame_{frame_index:06d}"
                        if not cv2.imwrite(str(images / f"{stem}.jpg"), frame):
                            raise RuntimeError(f"Could not save image frame {frame_index}.")
                        rows = []
                        h, w = frame.shape[:2]
                        for box in frame_boxes:
                            x1, y1, x2, y2 = (float(box[k]) for k in ("x1", "y1", "x2", "y2"))
                            cx, cy = (x1 + x2) / (2 * w), (y1 + y2) / (2 * h)
                            bw, bh = (x2 - x1) / w, (y2 - y1) / h
                            rows.append(f"{box['class_id']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                        (labels / f"{stem}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
                        saved += 1
                    frame_index += 1
                    if frame_index % 15 == 0:
                        self.progress.emit(int(frame_index * 100 / total), f"Exporting YOLO frames: {frame_index}/{total}")
                self.output_path = f"{root} ({saved} annotated frames)"
            self.progress.emit(100, "Export complete")
            self.completed.emit(self.output_path)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            if writer is not None:
                writer.release()
            if cap is not None:
                cap.release()


class VideoCanvas(QWidget):
    box_completed = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(500, 360)
        self.setMouseTracking(True)
        self.frame = None
        self.boxes = []
        self.selected_id = None
        self.drag_start = None
        self.drag_end = None
        self.pending_box = None

    def set_view(self, frame, boxes, selected_id, pending_box=None):
        self.frame = frame
        self.boxes = boxes
        self.selected_id = selected_id
        self.pending_box = pending_box
        self.update()

    def _mapping(self):
        if self.frame is None:
            return 1.0, 0.0, 0.0
        h, w = self.frame.shape[:2]
        scale = min(self.width() / w, self.height() / h)
        return scale, (self.width() - w * scale) / 2, (self.height() - h * scale) / 2

    def _to_video(self, point):
        scale, ox, oy = self._mapping()
        h, w = self.frame.shape[:2]
        x = max(0, min(w - 1, (point.x() - ox) / scale))
        y = max(0, min(h - 1, (point.y() - oy) / scale))
        return int(x), int(y)

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(12, 12, 12))
        if self.frame is None:
            painter.setPen(QColor(205, 205, 205))
            painter.drawText(self.rect(), Qt.AlignCenter, "Open a video to begin annotating")
            return
        h, w = self.frame.shape[:2]
        image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        qimage = QImage(image.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        scale, ox, oy = self._mapping()
        target = (int(ox), int(oy), int(w * scale), int(h * scale))
        painter.drawImage(*target, qimage)
        for box in self.boxes:
            color = QColor(*reversed(COLORS[box["class_id"] % len(COLORS)]))
            painter.setPen(QPen(color, 3 if box["id"] == self.selected_id else 2))
            x1, y1, x2, y2 = (box[k] for k in ("x1", "y1", "x2", "y2"))
            painter.drawRect(ox + x1 * scale, oy + y1 * scale, (x2 - x1) * scale, (y2 - y1) * scale)
            painter.drawText(ox + x1 * scale + 3, max(oy + y1 * scale - 5, 14), f'{box["class_name"]} ID:{box["id"]}')
        if self.pending_box:
            x1, y1, x2, y2 = (self.pending_box[k] for k in ("x1", "y1", "x2", "y2"))
            painter.setPen(QPen(QColor(0, 255, 255), 2, Qt.DashLine))
            painter.drawRect(ox + x1 * scale, oy + y1 * scale, (x2 - x1) * scale, (y2 - y1) * scale)
        if self.drag_start and self.drag_end:
            x1, y1 = self.drag_start
            x2, y2 = self.drag_end
            painter.setPen(QPen(QColor(0, 255, 255), 2, Qt.DashLine))
            painter.drawRect(ox + min(x1, x2) * scale, oy + min(y1, y2) * scale,
                             abs(x2 - x1) * scale, abs(y2 - y1) * scale)

    def mousePressEvent(self, event):
        if self.frame is not None and event.button() == Qt.LeftButton:
            self.drag_start = self._to_video(event.position())
            self.drag_end = self.drag_start
            self.update()

    def mouseMoveEvent(self, event):
        if self.drag_start:
            self.drag_end = self._to_video(event.position())
            self.update()

    def mouseReleaseEvent(self, event):
        if self.drag_start and event.button() == Qt.LeftButton:
            self.drag_end = self._to_video(event.position())
            x1, y1 = self.drag_start
            x2, y2 = self.drag_end
            self.drag_start = self.drag_end = None
            box = {"x1": min(x1, x2), "y1": min(y1, y2), "x2": max(x1, x2), "y2": max(y1, y2)}
            if box["x2"] - box["x1"] >= 5 and box["y2"] - box["y1"] >= 5:
                self.pending_box = box
                self.box_completed.emit(box)
            self.update()


class VideoAnnotatorWindow(QMainWindow):
    """Standalone temporal track and keyframe video annotation tool."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Sawing Machine Safety Video Annotator")
        self.resize(1450, 900)
        self.setMinimumSize(1100, 720)
        self.video_path = None
        self.cap = None
        self.fps = 25.0
        self.total_frames = 0
        self.width_px = self.height_px = 0
        self.current_frame_idx = 0
        self.current_frame = None
        self.playing = False
        self.tracks = {}
        self.next_track_id = 1
        self.selected_track_id = None
        self.classes = []
        self.current_class = -1
        self.pending_box = None
        self.dirty = False
        self.export_worker = None
        self.autosave_timer = QTimer(self)
        self.autosave_timer.setSingleShot(True)
        self.autosave_timer.timeout.connect(self._write_autosave)
        self._build_ui()
        self._refresh_tracks()
        self._update_controls()
        self._restore_autosave()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        top = QHBoxLayout()
        for label, slot in [("Open Video…", self.open_video), ("Save Project…", self.save_project),
                            ("Load Project…", self.load_project), ("Export Annotated Video…", self.export_video),
                            ("Export YOLO Dataset…", self.export_yolo)]:
            button = QPushButton(label)
            button.clicked.connect(slot)
            top.addWidget(button)
        top.addStretch(1)
        root.addLayout(top)
        split = QSplitter(Qt.Horizontal)
        self.canvas = VideoCanvas()
        self.canvas.box_completed.connect(self._box_drawn)
        split.addWidget(self.canvas)
        side = QWidget()
        side.setMinimumWidth(300)
        side.setMaximumWidth(400)
        sl = QVBoxLayout(side)
        class_group = QGroupBox("User-defined classes")
        form = QFormLayout(class_group)
        self.class_combo = QComboBox()
        self.class_combo.setPlaceholderText("Add a class to start")
        self.class_combo.currentIndexChanged.connect(self._class_changed)
        form.addRow("Class:", self.class_combo)
        class_buttons = QHBoxLayout()
        add_class = QPushButton("Add class…")
        add_class.clicked.connect(self.add_class)
        self.remove_class_button = QPushButton("Remove class")
        self.remove_class_button.clicked.connect(self.remove_class)
        class_buttons.addWidget(add_class)
        class_buttons.addWidget(self.remove_class_button)
        form.addRow(class_buttons)
        sl.addWidget(class_group)
        sl.addWidget(QLabel("Tracks / object occurrences"))
        self.track_list = QListWidget()
        self.track_list.currentRowChanged.connect(self._track_selected)
        sl.addWidget(self.track_list, 1)
        for label, slot in [
            ("Create new track from box", self.create_track),
            ("Update selected track at this frame", self.update_track),
            ("Add keyframe at this frame", self.add_keyframe),
            ("Set selected track start here", self.set_track_start),
            ("Set selected track end here", self.set_track_end),
            ("Change selected track class", self.change_track_class),
            ("Delete selected track", self.delete_track),
        ]:
            button = QPushButton(label)
            button.clicked.connect(slot)
            sl.addWidget(button)
        self.track_info = QLabel("No track selected.")
        self.track_info.setWordWrap(True)
        sl.addWidget(self.track_info)
        sl.addWidget(QLabel("Drag on the video to draw a box. Create a track for each separate object occurrence. Add keyframes as the object moves; intermediate boxes are interpolated."))
        split.addWidget(side)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 0)
        split.setSizes([1100, 350])
        root.addWidget(split, 1)
        playback = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_play)
        playback.addWidget(self.play_button)
        for label, delta in [("|<", None), ("−10", -10), ("−1", -1), ("+1", 1), ("+10", 10), (">|", "last")]:
            button = QPushButton(label)
            button.clicked.connect(lambda _checked=False, d=delta: self.goto_last() if d == "last" else self.show_frame(0) if d is None else self.step_frames(d))
            playback.addWidget(button)
        self.frame_label = QLabel("Frame 0 / 0")
        playback.addWidget(self.frame_label)
        playback.addStretch(1)
        self.time_label = QLabel("00:00.000 / 00:00.000")
        playback.addWidget(self.time_label)
        root.addLayout(playback)
        self.timeline = QSlider(Qt.Horizontal)
        self.timeline.setRange(0, 0)
        self.timeline.sliderReleased.connect(lambda: self.show_frame(self.timeline.value()))
        root.addWidget(self.timeline)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.hide()
        root.addWidget(self.progress)
        self.status = QLabel("Open a video to start. Space toggles playback; arrow keys move one frame.")
        root.addWidget(self.status)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._play_next)

    def open_video(self):
        if not self._confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open video", "", "Video files (*.mp4 *.avi *.mov *.mkv *.m4v);;All files (*.*)")
        if path:
            self._open_video_path(path)

    def _open_video_path(self, path, preserve_tracks=False):
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            QMessageBox.critical(self, "Open video", "Could not open the selected video.")
            return False
        self.stop_playback()
        if self.cap:
            self.cap.release()
        self.cap = cap
        self.video_path = os.path.abspath(path)
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if self.fps <= 0: self.fps = 25.0
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.width_px = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        self.height_px = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        self.current_frame_idx = 0
        self.pending_box = None
        if not preserve_tracks:
            self.tracks = {}
            self.next_track_id = 1
            self.selected_track_id = None
            self.dirty = False
        self.timeline.setRange(0, max(0, self.total_frames - 1))
        self._show_frame(0)
        self.status.setText(f"Loaded {Path(path).name} — {self.width_px}×{self.height_px} at {self.fps:.2f} FPS")
        self._update_controls()
        self._schedule_autosave()
        return True

    def _show_frame(self, index):
        if not self.cap or self.total_frames <= 0:
            return
        index = max(0, min(int(index), self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.cap.read()
        if not ok: return
        self.current_frame_idx = index
        self.current_frame = frame
        self.pending_box = None
        self.canvas.pending_box = None
        self.timeline.blockSignals(True)
        self.timeline.setValue(index)
        self.timeline.blockSignals(False)
        self.frame_label.setText(f"Frame {index + 1} / {self.total_frames}")
        self.time_label.setText(f"{self._time(index)} / {self._time(max(0, self.total_frames - 1))}")
        self._refresh_canvas()
        self._refresh_track_info()
        self._schedule_autosave()

    def show_frame(self, index):
        self._show_frame(index)

    def _time(self, frame):
        seconds = frame / max(self.fps, 1.0)
        return f"{int(seconds // 60):02d}:{seconds % 60:06.3f}"

    def step_frames(self, amount):
        self.stop_playback()
        self.show_frame(self.current_frame_idx + amount)

    def goto_last(self):
        self.stop_playback()
        self.show_frame(self.total_frames - 1)

    def toggle_play(self):
        if not self.cap: return
        if self.playing: self.stop_playback()
        else:
            self.playing = True
            self.play_button.setText("Pause")
            self.timer.start(max(1, round(1000 / self.fps)))

    def stop_playback(self):
        self.playing = False
        self.timer.stop()
        self.play_button.setText("Play")
        self._write_autosave()

    def _play_next(self):
        if not self.playing: return
        if self.current_frame_idx + 1 >= self.total_frames:
            self.stop_playback()
            return
        self._show_frame(self.current_frame_idx + 1)

    def _class_changed(self, index):
        self.current_class = index if index >= 0 else -1
        self.remove_class_button.setEnabled(index >= 0)
        self._schedule_autosave()

    def _refresh_class_combo(self, select=-1):
        self.class_combo.blockSignals(True)
        self.class_combo.clear()
        self.class_combo.addItems(self.classes)
        self.class_combo.setCurrentIndex(select if 0 <= select < len(self.classes) else -1)
        self.class_combo.blockSignals(False)
        self.current_class = self.class_combo.currentIndex()
        self.remove_class_button.setEnabled(self.current_class >= 0)

    def add_class(self):
        name, accepted = QInputDialog.getText(self, "Add class", "Class name:")
        if not accepted:
            return
        name = "_".join(name.strip().lower().split())
        if not name:
            QMessageBox.warning(self, "Add class", "Enter a class name.")
            return
        if name in self.classes:
            QMessageBox.warning(self, "Add class", "That class already exists.")
            return
        self.classes.append(name)
        self._refresh_class_combo(len(self.classes) - 1)
        self.dirty = True
        self._schedule_autosave()
        self.status.setText(f"Added class: {name}")

    def remove_class(self):
        index = self.class_combo.currentIndex()
        if index < 0:
            return
        if any(track["class_id"] == index for track in self.tracks.values()):
            QMessageBox.warning(self, "Class in use", "Change or remove all tracks using this class before removing it.")
            return
        self.classes.pop(index)
        for track in self.tracks.values():
            if track["class_id"] > index:
                track["class_id"] -= 1
                track["class_name"] = self.classes[track["class_id"]]
        self._refresh_class_combo(min(index, len(self.classes) - 1))
        self.dirty = True
        self._schedule_autosave()
        self._refresh_tracks()
        self._refresh_canvas()

    def _box_drawn(self, box):
        self.pending_box = box
        self._refresh_canvas()
        self.status.setText("Box drawn. Create a new track, update the selected track, or add a keyframe.")

    def _refresh_canvas(self):
        if self.current_frame is not None:
            self.canvas.set_view(self.current_frame, boxes_for_frame(self.tracks, self.current_frame_idx),
                                 self.selected_track_id, self.pending_box)

    def _new_track(self, box):
        if self.current_class < 0 or self.current_class >= len(self.classes):
            QMessageBox.warning(self, "Add a class first", "Add and select a class before creating tracks.")
            return
        track_id = self.next_track_id
        self.next_track_id += 1
        self.tracks[track_id] = {"id": track_id, "class_id": self.current_class,
            "class_name": self.classes[self.current_class], "start": self.current_frame_idx,
            "end": self.total_frames - 1, "keyframes": {str(self.current_frame_idx): dict(box)}}
        self.selected_track_id = track_id
        self.pending_box = None
        self.dirty = True
        self._refresh_tracks()
        self._refresh_canvas()
        self.status.setText(f"Created track {track_id} for {self.classes[self.current_class]}.")
        self._schedule_autosave()

    def create_track(self):
        if self.pending_box is None:
            QMessageBox.warning(self, "Draw a box", "Draw a bounding box on the current frame first.")
            return
        self._new_track(self.pending_box)

    def update_track(self):
        track = self.tracks.get(self.selected_track_id)
        if not track or self.pending_box is None:
            QMessageBox.warning(self, "Update track", "Select a track and draw its new box first.")
            return
        frame = self.current_frame_idx
        track["start"] = min(track["start"], frame)
        track["end"] = max(track["end"], frame)
        track["keyframes"][str(frame)] = dict(self.pending_box)
        self.pending_box = None
        self.dirty = True
        self._refresh_tracks()
        self._refresh_canvas()
        self.status.setText(f"Updated track {track['id']} at frame {frame + 1}.")
        self._schedule_autosave()

    def add_keyframe(self):
        track = self.tracks.get(self.selected_track_id)
        if not track or self.pending_box is None:
            QMessageBox.warning(self, "Add keyframe", "Select a track and draw the object's new box first.")
            return
        if self.current_frame_idx < track["start"]:
            QMessageBox.warning(self, "Add keyframe", "The current frame is before this track's start.")
            return
        track["end"] = max(track["end"], self.current_frame_idx)
        track["keyframes"][str(self.current_frame_idx)] = dict(self.pending_box)
        if 0 <= self.current_class < len(self.classes):
            track["class_id"] = self.current_class
            track["class_name"] = self.classes[self.current_class]
        self.pending_box = None
        self.dirty = True
        self._refresh_tracks()
        self._refresh_canvas()
        self.status.setText(f"Added a keyframe to track {track['id']}.")
        self._schedule_autosave()

    def set_track_start(self):
        self._set_track_bound("start")

    def set_track_end(self):
        self._set_track_bound("end")

    def _set_track_bound(self, bound):
        track = self.tracks.get(self.selected_track_id)
        if not track:
            QMessageBox.warning(self, "Track range", "Select a track first.")
            return
        frame = self.current_frame_idx
        new_start = frame if bound == "start" else track["start"]
        new_end = frame if bound == "end" else track["end"]
        if new_start > new_end:
            QMessageBox.warning(self, "Track range", "Track start must be before or equal to its end.")
            return
        track["start"], track["end"] = new_start, new_end
        track["keyframes"] = {key: box for key, box in track["keyframes"].items()
                               if new_start <= int(key) <= new_end}
        if not track["keyframes"]:
            QMessageBox.warning(self, "Track range", "No keyframes remain inside that range; the track was removed.")
            self.tracks.pop(track["id"], None)
            self.selected_track_id = None
        self.dirty = True
        self._refresh_tracks()
        self._refresh_canvas()
        self._schedule_autosave()

    def change_track_class(self):
        track = self.tracks.get(self.selected_track_id)
        if not track:
            QMessageBox.warning(self, "Track class", "Select a track first.")
            return
        if self.current_class < 0 or self.current_class >= len(self.classes):
            QMessageBox.warning(self, "Add a class first", "Add and select a class before changing a track class.")
            return
        track["class_id"] = self.current_class
        track["class_name"] = self.classes[self.current_class]
        self.dirty = True
        self._refresh_tracks()
        self._refresh_canvas()
        self._schedule_autosave()

    def delete_track(self):
        if self.selected_track_id not in self.tracks: return
        answer = QMessageBox.question(self, "Delete track", f"Delete track ID {self.selected_track_id}?")
        if answer != QMessageBox.Yes: return
        del self.tracks[self.selected_track_id]
        self.selected_track_id = None
        self.dirty = True
        self._refresh_tracks()
        self._refresh_canvas()
        self._schedule_autosave()

    def _track_selected(self, row):
        track_id = self.track_list.item(row).data(Qt.UserRole) if row >= 0 and self.track_list.item(row) else None
        if track_id in self.tracks:
            self.selected_track_id = track_id
            self.current_class = self.tracks[track_id]["class_id"]
            self.class_combo.setCurrentIndex(self.current_class)
            self._refresh_track_info()
            self._refresh_canvas()
            self._schedule_autosave()

    def _refresh_tracks(self):
        self.track_list.blockSignals(True)
        self.track_list.clear()
        for track_id in sorted(self.tracks):
            track = self.tracks[track_id]
            text = f"ID {track_id}: {track['class_name']} | {self._time(track['start'])} → {self._time(track['end'])}"
            from PySide6.QtWidgets import QListWidgetItem
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, track_id)
            self.track_list.addItem(item)
            if track_id == self.selected_track_id:
                self.track_list.setCurrentItem(item)
        self.track_list.blockSignals(False)
        self._refresh_track_info()

    def _refresh_track_info(self):
        track = self.tracks.get(self.selected_track_id)
        if not track:
            self.track_info.setText("No track selected.")
            return
        duration = max(0, track["end"] - track["start"]) / max(self.fps, 1)
        self.track_info.setText(f"Track ID: {track['id']}\nClass: {track['class_name']}\n"
            f"Start: {self._time(track['start'])}\nEnd: {self._time(track['end'])}\n"
            f"Duration: {duration:.2f} sec\nKeyframes: {len(track['keyframes'])}")

    def _update_controls(self):
        available = self.cap is not None and self.total_frames > 0
        self.play_button.setEnabled(available)
        self.timeline.setEnabled(available)

    def _confirm_discard(self):
        if not self.video_path:
            return True
        if not self.dirty: return True
        choice = QMessageBox.question(self, "Unsaved annotations", "Save annotation project before continuing?",
                                      QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if choice == QMessageBox.Cancel: return False
        if choice == QMessageBox.Save: return self.save_project()
        return True

    def _schedule_autosave(self):
        if hasattr(self, "autosave_timer"):
            self.autosave_timer.start(700)

    def _write_autosave(self):
        try:
            save_state("video_annotator.json", {
                "version": 1,
                "video_path": self.video_path,
                "frame": self.current_frame_idx,
                "fps": self.fps,
                "classes": self.classes,
                "tracks": self.tracks,
                "next_track_id": self.next_track_id,
                "selected_track_id": self.selected_track_id,
                "current_class": self.current_class,
            })
        except OSError as exc:
            self.status.setText(f"Could not autosave annotator session: {exc}")

    def _restore_autosave(self):
        state = load_state("video_annotator.json")
        path = state.get("video_path")
        classes = state.get("classes") or []
        self.classes = [str(name) for name in classes]
        self._refresh_class_combo(int(state.get("current_class", -1)))
        if not path:
            return
        if not Path(path).is_file():
            self.status.setText("Previous annotation session found, but its video is unavailable. Use Load Project or Open Video.")
            return
        try:
            tracks = {int(key): value for key, value in state.get("tracks", {}).items()}
            for track in tracks.values():
                track["keyframes"] = {str(key): value for key, value in track.get("keyframes", {}).items()}
                class_id = int(track.get("class_id", -1))
                if not 0 <= class_id < len(classes):
                    raise ValueError("Saved track refers to a class that is missing.")
                track["class_id"] = class_id
                track["class_name"] = classes[class_id]
            if not self._open_video_path(path, preserve_tracks=True):
                return
            self.classes = [str(name) for name in classes]
            self._refresh_class_combo(int(state.get("current_class", -1)))
            self.tracks = tracks
            self.next_track_id = max(int(state.get("next_track_id", 1)), max(tracks, default=0) + 1)
            self.selected_track_id = state.get("selected_track_id")
            if self.selected_track_id is not None:
                self.selected_track_id = int(self.selected_track_id)
            self.dirty = bool(tracks or classes)
            self._refresh_tracks()
            self._show_frame(int(state.get("frame", 0)))
            self.status.setText(f"Resumed annotation session: {Path(path).name}")
        except Exception as exc:
            self.status.setText(f"Could not restore annotation session: {exc}")

    def save_project(self):
        if not self.video_path:
            QMessageBox.warning(self, "No video", "Open a video before saving an annotation project.")
            return False
        path, _ = QFileDialog.getSaveFileName(self, "Save annotation project", str(Path(self.video_path).with_suffix(".annot.json")), "Annotation project (*.json)")
        if not path: return False
        payload = {"format": "sawing_safety_video_annotator", "version": 1, "video_path": self.video_path,
                   "fps": self.fps, "total_frames": self.total_frames, "width": self.width_px,
                   "height": self.height_px, "classes": self.classes, "tracks": self.tracks,
                   "next_track_id": self.next_track_id, "frame": self.current_frame_idx,
                   "current_class": self.current_class}
        try:
            Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            QMessageBox.critical(self, "Save project failed", str(exc))
            return False
        self.dirty = False
        self._write_autosave()
        self.status.setText(f"Project saved: {path}")
        return True

    def load_project(self):
        if not self._confirm_discard(): return
        path, _ = QFileDialog.getOpenFileName(self, "Load annotation project", "", "Annotation project (*.json)")
        if not path: return
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            video_path = payload["video_path"]
            if not Path(video_path).is_file():
                video_path, _ = QFileDialog.getOpenFileName(self, "Locate source video", "", "Video files (*.mp4 *.avi *.mov *.mkv *.m4v)")
                if not video_path: return
            tracks = {int(key): track for key, track in payload.get("tracks", {}).items()}
            classes = list(payload.get("classes") or [])
            if not classes and tracks:
                # Read older projects created with the earlier built-in four-class annotator.
                classes = list(LEGACY_CLASSES)
            for track in tracks.values():
                track["keyframes"] = {str(k): v for k, v in track.get("keyframes", {}).items()}
                if not track["keyframes"]: raise ValueError(f"Track {track.get('id')} has no keyframes.")
                track["class_id"] = int(track["class_id"])
                if not 0 <= track["class_id"] < len(classes): raise ValueError("Project contains an unknown class index.")
                track["class_name"] = classes[track["class_id"]]
            if not self._open_video_path(video_path, preserve_tracks=True): return
            self.classes = classes
            self._refresh_class_combo(int(payload.get("current_class", -1)))
            self.tracks = tracks
            self.next_track_id = max(int(payload.get("next_track_id", 1)), max(tracks, default=0) + 1)
            self.selected_track_id = None
            self.dirty = False
            self._refresh_tracks()
            self._refresh_canvas()
            self._show_frame(int(payload.get("frame", 0)))
            self.status.setText(f"Loaded annotation project: {Path(path).name}")
            self._write_autosave()
        except Exception as exc:
            QMessageBox.critical(self, "Load project failed", f"{type(exc).__name__}: {exc}")

    def export_video(self):
        self._start_export("video")

    def export_yolo(self):
        self._start_export("yolo")

    def _start_export(self, action):
        if not self.video_path:
            QMessageBox.warning(self, "No video", "Open a video first.")
            return
        if action == "yolo" and not self.classes:
            QMessageBox.warning(self, "Add a class first", "Add at least one class before exporting a YOLO dataset.")
            return
        if action == "video":
            path, _ = QFileDialog.getSaveFileName(self, "Export annotated video", str(Path(self.video_path).with_name(Path(self.video_path).stem + "_annotated.mp4")), "MP4 video (*.mp4)")
        else:
            path = QFileDialog.getExistingDirectory(self, "Select YOLO dataset export folder")
        if not path: return
        self.progress.setValue(0)
        self.progress.show()
        self.export_worker = ExportWorker(action, self.video_path, path, self.tracks, self.classes, self.fps, self.width_px, self.height_px)
        self.export_worker.progress.connect(self._export_progress)
        self.export_worker.completed.connect(self._export_done)
        self.export_worker.failed.connect(self._export_failed)
        self.export_worker.start()

    def _export_progress(self, percent, message):
        self.progress.setValue(percent)
        self.status.setText(message)

    def _export_done(self, path):
        self.progress.hide()
        self.status.setText(f"Export complete: {path}")
        QMessageBox.information(self, "Export complete", f"Saved to:\n{path}")

    def _export_failed(self, error):
        self.progress.hide()
        self.status.setText("Export failed")
        QMessageBox.critical(self, "Export failed", error)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.toggle_play()
        elif event.key() == Qt.Key_Left:
            self.step_frames(-1)
        elif event.key() == Qt.Key_Right:
            self.step_frames(1)
        elif event.key() == Qt.Key_Delete:
            self.delete_track()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self._write_autosave()
        self.stop_playback()
        if self.export_worker and self.export_worker.isRunning():
            self.export_worker.wait()
        if self.cap:
            self.cap.release()
        event.accept()
