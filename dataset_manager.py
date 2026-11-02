from __future__ import annotations

import random
import shutil
import hashlib
import zipfile
import json
import os
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from config import APP_ROOT, APP_STATE_DIR, IMAGE_EXTENSIONS, normalize_class_name
from annotation_formats import (
    _coco_image, _coco_index, _labelme_json_for_image, class_names_for,
    clear_annotation_cache, detect_annotation_format, read_boxes, write_boxes,
)

@dataclass
class DatasetInfo:
    root: Path
    kind: str  # "raw" or "yolo"
    images_root: Path
    labels_root: Optional[Path]
    class_file: Path
    data_yaml: Optional[Path]
    split_dirs: Dict[str, Tuple[Path, Optional[Path]]]
    annotation_format: str
    annotation_source: Optional[Path]

class DatasetError(Exception):
    pass

class DatasetManager:
    def __init__(self):
        self.info: Optional[DatasetInfo] = None
        self.classes: List[str] = []

    @staticmethod
    def _resolve(path: Path) -> Path:
        return Path(path).expanduser().resolve()

    @staticmethod
    def _inside(path: Path, parent: Path) -> bool:
        try:
            path.relative_to(parent)
            return True
        except ValueError:
            return False

    @staticmethod
    def extract_cvat_archive(archive: Path) -> Path:
        """Safely extract a CVAT ZIP into the app cache and return its dataset root."""
        archive = Path(archive).expanduser().resolve()
        if not archive.is_file() or archive.suffix.lower() != ".zip":
            raise DatasetError("Choose a CVAT .zip archive.")
        digest = hashlib.sha256(f"{archive}|{archive.stat().st_size}|{archive.stat().st_mtime_ns}".encode()).hexdigest()[:20]
        cache = APP_STATE_DIR / "cvat_imports" / digest
        if not cache.is_dir():
            cache.mkdir(parents=True, exist_ok=False)
            try:
                with zipfile.ZipFile(archive) as zipped:
                    base = cache.resolve()
                    for item in zipped.infolist():
                        destination = (cache / item.filename).resolve()
                        if not DatasetManager._inside(destination, base):
                            raise DatasetError("The ZIP contains a path that escapes its extraction folder.")
                        if item.is_dir():
                            destination.mkdir(parents=True, exist_ok=True)
                        else:
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            with zipped.open(item) as source, destination.open("wb") as target:
                                shutil.copyfileobj(source, target)
            except Exception:
                shutil.rmtree(cache, ignore_errors=True)
                raise
        candidates = [xml.parent for xml in cache.rglob("annotations.xml")
                      if (xml.parent / "images").is_dir()]
        if candidates:
            return candidates[0]
        raise DatasetError("No CVAT export was found in the ZIP. Expected annotations.xml and an images/ folder.")

    def _find_images_root(self, root: Path):
        images = root / "images"
        if images.is_dir():
            return images
        if (root / "JPEGImages").is_dir():
            return root / "JPEGImages"
        # A dataset can also be selected directly at images/.
        if root.name.lower() == "images" and root.is_dir():
            return root
        if root.is_dir() and any(p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS for p in root.iterdir()):
            return root
        return None

    def detect(self, root: Path) -> DatasetInfo:
        root = self._resolve(root)
        if not root.is_dir():
            raise DatasetError(f"Dataset folder does not exist:\n{root}")

        images_root = self._find_images_root(root)
        data_yaml = root / "data.yaml"
        labels_root = root / "labels"
        annotations_root = root / "annotations"

        if images_root is None:
            raise DatasetError(
                "No images folder was found.\n\n"
                "Expected either:\n"
                "  dataset/images/\n"
                "or select the images folder itself."
            )

        is_yolo = (
            data_yaml.is_file()
            or (images_root / "train").is_dir()
            or (images_root / "val").is_dir()
            or (images_root / "test").is_dir()
        )

        if labels_root.is_dir():
            label_root = labels_root
        elif (root / "Annotations").is_dir():
            label_root = root / "Annotations"
        elif annotations_root.is_dir() and not any(annotations_root.glob("*.json")):
            label_root = annotations_root
        else:
            label_root = None

        annotation_format, annotation_source = detect_annotation_format(root, images_root, label_root)

        if is_yolo:
            splits = {}
            for split in ("train", "val", "test"):
                img_dir = images_root / split
                if img_dir.is_dir():
                    lbl_dir = None
                    if label_root:
                        candidate = label_root / split
                        if candidate.is_dir():
                            lbl_dir = candidate
                    splits[split] = (img_dir, lbl_dir)
            if not splits:
                raise DatasetError(
                    "This looks like a YOLO dataset, but no train/val/test image "
                    "folders were found under images/."
                )
            kind = "yolo"
        else:
            splits = {"all": (images_root, label_root)}
            kind = "raw"

        return DatasetInfo(
            root=root,
            kind=kind,
            images_root=images_root,
            labels_root=label_root,
            class_file=root / "classes.txt",
            data_yaml=data_yaml if data_yaml.is_file() else None,
            split_dirs=splits,
            annotation_format=annotation_format,
            annotation_source=annotation_source,
        )

    def open(self, root: Path):
        clear_annotation_cache()
        info = self.detect(root)
        self.info = info
        self.classes = self._load_or_request_classes()
        return info

    def _load_classes_file(self) -> List[str]:
        if not self.info or not self.info.class_file.is_file():
            return []
        try:
            return [
                normalize_class_name(x)
                for x in self.info.class_file.read_text(encoding="utf-8").splitlines()
                if x.strip()
            ]
        except OSError:
            return []

    def _load_yaml_names(self) -> List[str]:
        if not self.info or not self.info.data_yaml:
            return []
        try:
            data = yaml.safe_load(self.info.data_yaml.read_text(encoding="utf-8")) or {}
            names = data.get("names", [])
            if isinstance(names, dict):
                pairs = sorted(((int(k), v) for k, v in names.items()), key=lambda x: x[0])
                return [normalize_class_name(str(v)) for _, v in pairs]
            if isinstance(names, list):
                return [normalize_class_name(str(v)) for v in names]
        except Exception:
            return []
        return []

    def _label_files(self) -> List[Path]:
        if not self.info:
            return []
        files = []
        for _, (_, label_dir) in self.info.split_dirs.items():
            if label_dir and label_dir.is_dir():
                files.extend(label_dir.rglob("*.txt"))
        return sorted(set(files))

    def detect_class_ids(self) -> List[int]:
        if self.info and self.info.annotation_format not in ("yolo", "none"):
            return list(range(len(class_names_for(self.info.annotation_format, self.info.annotation_source))))
        ids = set()
        for label in self._label_files():
            try:
                for line in label.read_text(encoding="utf-8").splitlines():
                    p = line.split()
                    if len(p) >= 1:
                        ids.add(int(p[0]))
            except (OSError, ValueError):
                continue
        return sorted(i for i in ids if i >= 0)

    def _write_classes(self, classes: List[str]):
        if not self.info:
            raise DatasetError("No dataset is open.")
        self.info.class_file.parent.mkdir(parents=True, exist_ok=True)
        self.info.class_file.write_text(
            "\n".join(classes) + ("\n" if classes else ""),
            encoding="utf-8",
        )

    def set_classes(self, classes: List[str]):
        cleaned = [normalize_class_name(x) for x in classes if normalize_class_name(x)]
        if not cleaned:
            raise DatasetError("At least one class is required.")
        if len(set(cleaned)) != len(cleaned):
            raise DatasetError("Class names must be unique.")
        if self.info and cleaned != self.classes:
            touched = [self.info.class_file]
            if self.info.data_yaml:
                touched.append(self.info.data_yaml)
            self._recovery_snapshot(touched)
        self.classes = cleaned
        self._write_classes(cleaned)
        if self.info and self.info.data_yaml:
            data = yaml.safe_load(self.info.data_yaml.read_text(encoding="utf-8")) or {}
            data["names"] = {i: name for i, name in enumerate(cleaned)}
            data["nc"] = len(cleaned)
            self.info.data_yaml.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")

    def rename_class(self, class_id: int, name: str):
        if not self.info or not 0 <= class_id < len(self.classes):
            raise DatasetError("Choose a valid class.")
        cleaned = normalize_class_name(name)
        if not cleaned:
            raise DatasetError("Class name cannot be empty.")
        if cleaned in self.classes and self.classes[class_id] != cleaned:
            raise DatasetError("Class names must be unique.")
        old = self.classes[class_id]
        if old == cleaned:
            return
        fmt = self.info.annotation_format
        files = []
        if fmt == "coco" and self.info.annotation_source:
            files = [self.info.annotation_source]
        elif fmt == "cvat" and self.info.annotation_source:
            files = [self.info.annotation_source]
        elif fmt in ("voc", "labelme"):
            files = list(dict.fromkeys(p for image, _, _ in self.image_records()
                                       if (p := self.annotation_storage_for(image)) and p.is_file()))
        touched = [self.info.class_file, *files]
        if self.info.data_yaml:
            touched.append(self.info.data_yaml)
        self._recovery_snapshot(touched)
        try:
            for path in files:
                if fmt == "coco":
                    data = json.loads(path.read_text(encoding="utf-8"))
                    for category in data.get("categories", []):
                        if normalize_class_name(str(category.get("name", ""))) == old:
                            category["name"] = cleaned
                    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
                elif fmt == "labelme":
                    data = json.loads(path.read_text(encoding="utf-8"))
                    for shape in data.get("shapes", data.get("objects", [])):
                        key = "label" if "label" in shape else "class"
                        if normalize_class_name(str(shape.get(key, ""))) == old:
                            shape[key] = cleaned
                    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
                else:
                    tree = ET.parse(path)
                    root = tree.getroot()
                    if fmt == "voc":
                        for node in root.findall("object/name"):
                            if normalize_class_name(node.text or "") == old:
                                node.text = cleaned
                    else:
                        for node in root.findall(".//label/name"):
                            if normalize_class_name(node.text or "") == old:
                                node.text = cleaned
                        for node in list(root.findall("track")) + list(root.findall(".//image/box")):
                            if normalize_class_name(node.get("label", "")) == old:
                                node.set("label", cleaned)
                    tree.write(path, encoding="utf-8", xml_declaration=True)
            self.classes[class_id] = cleaned
            self._write_classes(self.classes)
            if self.info.data_yaml:
                data = yaml.safe_load(self.info.data_yaml.read_text(encoding="utf-8")) or {}
                data["names"] = {i: value for i, value in enumerate(self.classes)}
                data["nc"] = len(self.classes)
                self.info.data_yaml.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
            clear_annotation_cache()
        except Exception:
            self.classes[class_id] = old
            raise

    def reorder_classes(self, order: List[int]):
        if not self.info or sorted(order) != list(range(len(self.classes))):
            raise DatasetError("Class order must contain every current class ID exactly once.")
        if order == list(range(len(self.classes))):
            return
        if self.info.annotation_format == "yolo":
            old_classes = list(self.classes)
            remap = {old_id: new_id for new_id, old_id in enumerate(order)}
            updates, _removed, _remapped = self._stage_yolo_class_edit(lambda cid: remap.get(cid, cid))
            touched = [self.info.class_file, *[label for _image, label, _content in updates]]
            if self.info.data_yaml:
                touched.append(self.info.data_yaml)
            self._recovery_snapshot(touched)
            self.classes = [old_classes[old_id] for old_id in order]
            try:
                self._write_classes(self.classes)
                if self.info.data_yaml:
                    data = yaml.safe_load(self.info.data_yaml.read_text(encoding="utf-8")) or {}
                    data["names"] = {i: name for i, name in enumerate(self.classes)}
                    data["nc"] = len(self.classes)
                    self.info.data_yaml.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
                self._write_staged_yolo(updates)
            except Exception:
                self.classes = old_classes
                raise
            return
        import cv2
        old_classes = list(self.classes)
        remap = {old_id: new_id for new_id, old_id in enumerate(order)}
        changed, touched = [], [self.info.class_file]
        if self.info.data_yaml:
            touched.append(self.info.data_yaml)
        for image, _, _ in self.image_records():
            frame = cv2.imread(str(image))
            if frame is None:
                if self.has_annotations(image):
                    raise DatasetError(f"Cannot safely remap classes because this annotated image is unreadable: {image}")
                continue
            h, w = frame.shape[:2]
            boxes = self.boxes_for_image(image, w, h)
            if any(int(b.get("class_id", -1)) in remap and remap[int(b["class_id"])] != int(b["class_id"]) for b in boxes):
                updated = [dict(b, class_id=remap[int(b["class_id"])]) if int(b.get("class_id", -1)) in remap else b for b in boxes]
                storage = self.annotation_storage_for(image)
                if storage:
                    touched.append(storage)
                changed.append((image, w, h, updated))
        self._recovery_snapshot(touched)
        self.classes = [old_classes[old_id] for old_id in order]
        try:
            self._write_classes(self.classes)
            if self.info.data_yaml:
                data = yaml.safe_load(self.info.data_yaml.read_text(encoding="utf-8")) or {}
                data["names"] = {i: name for i, name in enumerate(self.classes)}
                data["nc"] = len(self.classes)
                self.info.data_yaml.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
            for image, w, h, boxes in changed:
                self.save_boxes(image, w, h, boxes)
        except Exception:
            self.classes = old_classes
            raise

    def class_statistics(self) -> dict:
        """Summarize the open dataset without retaining image contents in memory."""
        if not self.info:
            raise DatasetError("No dataset is open.")
        counts = [0] * len(self.classes)
        image_counts = [0] * len(self.classes)
        images = annotations = labeled = empty = missing = 0
        formats: Dict[str, int] = {}
        dimensions: Dict[str, int] = {}
        problems = []
        try:
            import cv2
        except ImportError:
            cv2 = None
        for image, _label, split in self.image_records():
            images += 1
            formats[image.suffix.lower()] = formats.get(image.suffix.lower(), 0) + 1
            if not self.has_annotations(image):
                missing += 1
            try:
                frame = cv2.imread(str(image)) if cv2 else None
                if frame is None:
                    problems.append(f"Unreadable image: {image}")
                    continue
                h, w = frame.shape[:2]
                dim = f"{w}×{h}"
                dimensions[dim] = dimensions.get(dim, 0) + 1
                boxes = self.boxes_for_image(image, w, h)
                if boxes:
                    labeled += 1
                else:
                    empty += 1
                seen = set()
                for box in boxes:
                    cid = int(box.get("class_id", -1))
                    if cid < 0 or cid >= len(counts):
                        problems.append(f"Invalid class ID {cid}: {image}")
                        continue
                    counts[cid] += 1
                    annotations += 1
                    seen.add(cid)
                for cid in seen:
                    image_counts[cid] += 1
            except (OSError, ValueError, TypeError, KeyError) as exc:
                problems.append(f"Invalid annotation for {image}: {exc}")
        return {"images": images, "annotations": annotations, "labeled": labeled,
                "empty": empty, "missing": missing, "classes": [
                    {"id": i, "name": name, "images": image_counts[i], "annotations": counts[i]}
                    for i, name in enumerate(self.classes)],
                "formats": formats, "dimensions": dimensions, "problems": problems}

    def _recovery_snapshot(self, paths: List[Path]) -> Path:
        """Back up touched files before mutation; retain snapshots for Undo."""
        if not self.info:
            raise DatasetError("No dataset is open.")
        root = self.info.root
        recovery = root / ".dataset_studio_recovery"
        recovery.mkdir(parents=True, exist_ok=True)
        stamp = f"{time.time_ns()}"
        snapshot = recovery / stamp
        snapshot.mkdir()
        manifest = []
        try:
            for index, path in enumerate(dict.fromkeys(Path(p).resolve() for p in paths)):
                try:
                    rel = path.relative_to(root)
                except ValueError:
                    continue
                exists = path.is_file()
                entry = {"path": str(rel), "exists": exists, "backup": str(index)}
                manifest.append(entry)
                if exists:
                    backup = snapshot / str(index)
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, backup)
            (snapshot / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            history_path = recovery / "history.log"
            if not history_path.exists():
                migrated = False
                try:
                    legacy = json.loads((recovery / "history.json").read_text(encoding="utf-8"))
                    if isinstance(legacy, list):
                        with history_path.open("a", encoding="utf-8") as history_file:
                            for old_stamp in legacy:
                                history_file.write(f"ADD {old_stamp}\n")
                        migrated = bool(legacy)
                except (OSError, ValueError):
                    pass
                if not migrated and (recovery / "latest.txt").is_file():
                    try:
                        previous = (recovery / "latest.txt").read_text(encoding="utf-8").strip()
                        if previous:
                            with history_path.open("a", encoding="utf-8") as history_file:
                                history_file.write(f"ADD {previous}\n")
                    except OSError:
                        pass
            with history_path.open("a", encoding="utf-8") as history_file:
                history_file.write(f"ADD {stamp}\n")
                history_file.flush()
                os.fsync(history_file.fileno())
        except Exception:
            shutil.rmtree(snapshot, ignore_errors=True)
            raise
        return snapshot

    def undo_last_operation(self) -> str:
        if not self.info:
            raise DatasetError("No dataset is open.")
        recovery = self.info.root / ".dataset_studio_recovery"
        history_path = recovery / "history.log"
        latest = recovery / "latest.txt"
        history = []
        if history_path.is_file():
            try:
                for line in history_path.read_text(encoding="utf-8").splitlines():
                    action, _, stamp = line.partition(" ")
                    if action == "ADD" and stamp:
                        history.append(stamp)
                    elif action == "UNDO" and stamp and history and history[-1] == stamp:
                        history.pop()
            except OSError:
                history = []
        # Read the JSON stack written by the earlier Phase 1 implementation.
        if not history and not history_path.is_file():
            try:
                history = json.loads((recovery / "history.json").read_text(encoding="utf-8"))
                if not isinstance(history, list):
                    history = []
            except (OSError, ValueError):
                history = []
        # Read the original single-snapshot layout for backward compatibility.
        if not history and not history_path.is_file() and latest.is_file():
            history = [latest.read_text(encoding="utf-8").strip()]
        if not history:
            raise DatasetError("There is no saved dataset operation to undo.")
        snapshot = recovery / str(history[-1])
        manifest_path = snapshot / "manifest.json"
        if not manifest_path.is_file():
            raise DatasetError("The recovery snapshot is incomplete.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for index, item in reversed(list(enumerate(manifest))):
            target = self.info.root / item["path"]
            if item["exists"]:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(snapshot / item.get("backup", str(index)), target)
            else:
                target.unlink(missing_ok=True)
        with history_path.open("a", encoding="utf-8") as history_file:
            history_file.write(f"UNDO {history[-1]}\n")
            history_file.flush()
            os.fsync(history_file.fileno())
        history.pop()
        clear_annotation_cache()
        self.open(self.info.root)
        return f"Restored {len(manifest)} file(s)."

    def images_that_become_unannotated(self, class_id: int, images: Optional[List[Path]] = None) -> List[Path]:
        if not self.info or not 0 <= class_id < len(self.classes):
            raise DatasetError("Choose a valid class.")
        allowed = {self._resolve(p) for p in images} if images is not None else None
        import cv2
        result = []
        for image, _, _ in self.image_records():
            if allowed is not None and self._resolve(image) not in allowed:
                continue
            frame = cv2.imread(str(image))
            if frame is None:
                if self.has_annotations(image):
                    raise DatasetError(f"Cannot safely remove annotations because this annotated image is unreadable: {image}")
                continue
            h, w = frame.shape[:2]
            boxes = self.boxes_for_image(image, w, h)
            if boxes and all(int(box.get("class_id", -1)) == class_id for box in boxes):
                result.append(image)
        return result

    def _stage_yolo_class_edit(self, transform, images=None):
        """Prepare lossless YOLO class-ID edits without decoding images or boxes."""
        allowed = {self._resolve(p) for p in images} if images is not None else None
        updates, removed, remapped = [], 0, 0
        for image, _, _ in self.image_records():
            if allowed is not None and self._resolve(image) not in allowed:
                continue
            label = self.label_path_for(image)
            if not label or not label.is_file():
                continue
            raw = label.read_text(encoding="utf-8")
            output = []
            changed = False
            for line in raw.splitlines():
                parts = line.split()
                if not parts:
                    output.append(line)
                    continue
                try:
                    cid = int(parts[0])
                except ValueError as exc:
                    raise DatasetError(f"Cannot safely remap class IDs in {label}: invalid class ID {parts[0]!r}") from exc
                if cid < 0 or cid >= len(self.classes):
                    raise DatasetError(f"Cannot safely remap class ID {cid} in {label}; fix the invalid ID first.")
                mapped = transform(cid)
                if mapped is None:
                    removed += 1
                    changed = True
                    continue
                if mapped != cid:
                    prefix = line[:len(line) - len(line.lstrip())]
                    token_end = len(prefix) + len(parts[0])
                    line = prefix + str(mapped) + line[token_end:]
                    changed = True
                    remapped += 1
                output.append(line)
            if changed:
                newline = "\r\n" if "\r\n" in raw else "\n"
                suffix = newline if raw.endswith(("\n", "\r")) and output else ""
                updates.append((image, label, newline.join(output) + suffix))
        return updates, removed, remapped

    @staticmethod
    def _write_staged_yolo(updates):
        for _image, label, content in updates:
            label.write_text(content, encoding="utf-8")

    def remove_class_annotations(self, class_id: int, images: Optional[List[Path]] = None,
                                 delete_empty_images: Optional[List[Path]] = None) -> dict:
        """Remove one class's boxes while preserving images and all other boxes."""
        if not self.info or not 0 <= class_id < len(self.classes):
            raise DatasetError("Choose a valid class.")
        records = self.image_records()
        allowed = {self._resolve(p) for p in images} if images is not None else None
        changed, removed, touched = [], 0, []
        delete_set = {self._resolve(p) for p in (delete_empty_images or [])}
        if self.info.annotation_format == "yolo":
            updates, removed, _remapped = self._stage_yolo_class_edit(lambda cid: None if cid == class_id else cid, images)
            empty_paths = {self._resolve(image) for image, _label, content in updates if not content.strip()}
            if delete_set.difference(empty_paths):
                raise DatasetError("Only images that become unannotated in this operation can be selected for deletion.")
            touched = [label for _image, label, _content in updates]
            touched.extend(image for image, _label, _content in updates if self._resolve(image) in delete_set)
            if not updates:
                if delete_set:
                    raise DatasetError("No annotations of this class were found in the selected images.")
                return {"images": 0, "annotations": 0, "deleted_images": 0}
            self._recovery_snapshot(touched)
            kept_updates = []
            for image, label, content in updates:
                if self._resolve(image) in delete_set:
                    label.unlink(missing_ok=True)
                    image.unlink()
                else:
                    kept_updates.append((image, label, content))
            self._write_staged_yolo(kept_updates)
            return {"images": len(updates), "annotations": removed, "deleted_images": len(delete_set)}
        import cv2
        for image, _label, _split in records:
            if allowed is not None and self._resolve(image) not in allowed:
                continue
            frame = cv2.imread(str(image))
            if frame is None:
                if self.has_annotations(image):
                    raise DatasetError(f"Cannot safely remove annotations because this annotated image is unreadable: {image}")
                continue
            h, w = frame.shape[:2]
            boxes = self.boxes_for_image(image, w, h)
            keep = [box for box in boxes if int(box.get("class_id", -1)) != class_id]
            n = len(boxes) - len(keep)
            if not n:
                continue
            target = self.annotation_storage_for(image)
            if target is None:
                raise DatasetError("This annotation format has no writable source.")
            touched.append(target)
            if self._resolve(image) in delete_set:
                touched.append(image)
            changed.append((image, w, h, keep))
            removed += n
        changed_paths = {self._resolve(image) for image, _, _, boxes in changed if not boxes}
        if delete_set.difference(changed_paths):
            raise DatasetError("Only images that become unannotated in this operation can be selected for deletion.")
        if not changed:
            return {"images": 0, "annotations": 0, "deleted_images": 0}
        self._recovery_snapshot(touched)
        for image, w, h, boxes in changed:
            if self._resolve(image) in delete_set:
                self.remove_image_annotations([image])
                image.unlink()
            else:
                self.save_boxes(image, w, h, boxes)
        return {"images": len(changed), "annotations": removed, "deleted_images": len(delete_set)}

    def edit_annotations(self, image_path: Path, indexes: List[int], class_id: Optional[int] = None,
                         delete: bool = False) -> dict:
        if not self.info:
            raise DatasetError("No dataset is open.")
        image_path = self._resolve(image_path)
        known = {self._resolve(image) for image, _, _ in self.image_records()}
        if image_path not in known:
            raise DatasetError("Choose an image from the open dataset.")
        if not indexes or len(set(indexes)) != len(indexes):
            raise DatasetError("Select one or more unique annotations.")
        if class_id is not None and not 0 <= class_id < len(self.classes):
            raise DatasetError("Choose a valid target class.")
        import cv2
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise DatasetError(f"Image could not be read: {image_path}")
        h, w = frame.shape[:2]
        boxes = self.boxes_for_image(image_path, w, h)
        if any(i < 0 or i >= len(boxes) for i in indexes):
            raise DatasetError("The annotation selection is out of date. Refresh and try again.")
        updated = list(boxes)
        if delete:
            for i in sorted(indexes, reverse=True):
                updated.pop(i)
        elif class_id is not None:
            for i in indexes:
                updated[i] = dict(updated[i], class_id=class_id)
        target = self.annotation_storage_for(image_path)
        if target is None:
            raise DatasetError("This annotation format has no writable source.")
        self._recovery_snapshot([target])
        self.save_boxes(image_path, w, h, updated)
        return {"images": 1, "annotations": len(indexes), "remaining": len(updated)}

    def replace_class(self, source_id: int, target_id: int) -> dict:
        if not self.info or not (0 <= source_id < len(self.classes) and 0 <= target_id < len(self.classes)):
            raise DatasetError("Choose valid classes.")
        if source_id == target_id:
            raise DatasetError("Choose two different classes.")
        if self.info.annotation_format == "yolo":
            updates, _removed, remapped = self._stage_yolo_class_edit(
                lambda cid: target_id if cid == source_id else cid)
            if updates:
                self._recovery_snapshot([label for _image, label, _content in updates])
                self._write_staged_yolo(updates)
            return {"images": len(updates), "annotations": remapped}
        import cv2
        records = self.image_records()
        changed, touched, count = [], [], 0
        for image, _label, _split in records:
            frame = cv2.imread(str(image))
            if frame is None:
                if self.has_annotations(image):
                    raise DatasetError(f"Cannot safely replace classes because this annotated image is unreadable: {image}")
                continue
            h, w = frame.shape[:2]
            boxes = self.boxes_for_image(image, w, h)
            n = sum(int(b.get("class_id", -1)) == source_id for b in boxes)
            if n:
                for box in boxes:
                    if int(box.get("class_id", -1)) == source_id:
                        box["class_id"] = target_id
                target = self.annotation_storage_for(image)
                if target is None:
                    raise DatasetError("This annotation format has no writable source.")
                touched.append(target); changed.append((image, w, h, boxes)); count += n
        if changed:
            self._recovery_snapshot(touched)
            for image, w, h, boxes in changed:
                self.save_boxes(image, w, h, boxes)
        return {"images": len(changed), "annotations": count}

    def delete_images(self, images: List[Path]) -> dict:
        """Delete explicitly selected dataset images after snapshotting their data."""
        if not self.info:
            raise DatasetError("No dataset is open.")
        known = {self._resolve(image): image for image, _, _ in self.image_records()}
        selected = list(dict.fromkeys(self._resolve(path) for path in images))
        if not selected or any(path not in known for path in selected):
            raise DatasetError("Select one or more images from the open dataset.")
        targets = []
        annotation_counts = {}
        import cv2
        for image in selected:
            targets.append(image)
            storage = self.annotation_storage_for(image)
            if storage and storage.is_file():
                targets.append(storage)
            frame = cv2.imread(str(image))
            if frame is None:
                annotation_counts[image] = 0
            else:
                h, w = frame.shape[:2]
                annotation_counts[image] = len(self.boxes_for_image(image, w, h))
        self._recovery_snapshot(targets)
        try:
            self.remove_image_annotations(selected)
            for image in selected:
                image.unlink()
        except Exception:
            raise DatasetError("Deletion stopped after an error. Use Undo last operation to restore the saved snapshot.")
        return {"images": len(selected), "annotations": sum(annotation_counts.values())}

    def remove_image_annotations(self, images: List[Path]):
        """Remove annotation records for images being deleted, preserving other records."""
        if not self.info or self.info.annotation_format == "none":
            return
        fmt, source = self.info.annotation_format, self.info.annotation_source
        if fmt == "yolo":
            for image in images:
                label = self.label_path_for(image)
                if label:
                    label.unlink(missing_ok=True)
            return
        if fmt in ("voc", "labelme"):
            for image in images:
                storage = self.annotation_storage_for(image)
                if storage:
                    storage.unlink(missing_ok=True)
            return
        if not source or not source.is_file():
            return
        selected = {self._resolve(image) for image in images}
        if fmt == "coco":
            data = json.loads(source.read_text(encoding="utf-8"))
            remove_ids = set()
            for image in selected:
                item = _coco_image(image, data, source)
                if item is not None:
                    remove_ids.add(int(item["id"]))
            data["images"] = [item for item in data.get("images", []) if int(item.get("id", -1)) not in remove_ids]
            data["annotations"] = [a for a in data.get("annotations", []) if int(a.get("image_id", -1)) not in remove_ids]
            source.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            clear_annotation_cache()
            return
        if fmt == "cvat":
            tree = ET.parse(source); root = tree.getroot()
            nodes = root.findall("image")
            selected_names = {p.name.casefold() for p in selected}
            removed_nodes = [node for node in nodes if Path(node.get("name", "")).name.casefold() in selected_names]
            image_root = source.parent / "images"
            frame_paths = sorted((p for p in image_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
                                 key=lambda p: str(p).casefold()) if image_root.is_dir() else []
            removed_frames = {i for i, path in enumerate(frame_paths) if self._resolve(path) in selected}
            for node in removed_nodes:
                root.remove(node)
            if root.findall("track") and removed_frames:
                for track in root.findall("track"):
                    for box in list(track.findall("box")):
                        frame_no = int(box.get("frame", -1))
                        if frame_no in removed_frames:
                            track.remove(box)
                        elif frame_no >= 0:
                            box.set("frame", str(frame_no - sum(i < frame_no for i in removed_frames)))
            size = root.find(".//task/size")
            if size is not None and size.text and removed_frames:
                try: size.text = str(max(0, int(size.text) - len(removed_frames)))
                except ValueError: pass
            tree.write(source, encoding="utf-8", xml_declaration=True)
            clear_annotation_cache()

    def delete_class(self, class_id: int) -> dict:
        """Delete one class, remap later IDs, and snapshot every edited file."""
        if not self.info or not 0 <= class_id < len(self.classes):
            raise DatasetError("Choose a valid class.")
        if len(self.classes) == 1:
            raise DatasetError("A dataset must keep at least one class.")
        if self.info.annotation_format == "yolo":
            updates, removed, _remapped = self._stage_yolo_class_edit(
                lambda cid: None if cid == class_id else (cid - 1 if cid > class_id else cid))
            old_classes = list(self.classes)
            touched = [self.info.class_file, *[label for _image, label, _content in updates]]
            if self.info.data_yaml:
                touched.append(self.info.data_yaml)
            self._recovery_snapshot(touched)
            self.classes = old_classes[:class_id] + old_classes[class_id + 1:]
            try:
                self._write_classes(self.classes)
                if self.info.data_yaml:
                    data = yaml.safe_load(self.info.data_yaml.read_text(encoding="utf-8")) or {}
                    data["names"] = {i: name for i, name in enumerate(self.classes)}
                    data["nc"] = len(self.classes)
                    self.info.data_yaml.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
                self._write_staged_yolo(updates)
            except Exception:
                self.classes = old_classes
                raise
            return {"images": len(updates), "annotations": removed}
        import cv2
        records = self.image_records()
        changed, touched, removed = [], [self.info.class_file], 0
        if self.info.data_yaml:
            touched.append(self.info.data_yaml)
        for image, _label, _split in records:
            frame = cv2.imread(str(image))
            if frame is None:
                if self.has_annotations(image):
                    raise DatasetError(f"Cannot safely delete a class because this annotated image is unreadable: {image}")
                continue
            h, w = frame.shape[:2]
            boxes = self.boxes_for_image(image, w, h)
            if not boxes:
                continue
            mapped = []
            for box in boxes:
                cid = int(box.get("class_id", -1))
                if cid == class_id:
                    removed += 1
                    continue
                if cid > class_id:
                    box = dict(box, class_id=cid - 1)
                mapped.append(box)
            if len(mapped) != len(boxes) or any(int(a.get("class_id", -1)) != int(b.get("class_id", -1))
                                                for a, b in zip(mapped, boxes)):
                target = self.annotation_storage_for(image)
                if target is None:
                    raise DatasetError("This annotation format has no writable source.")
                touched.append(target)
                changed.append((image, w, h, mapped))
        # Fail closed when labels cannot be parsed; never silently discard them.
        self._recovery_snapshot(touched)
        old_classes = list(self.classes)
        self.classes = old_classes[:class_id] + old_classes[class_id + 1:]
        try:
            self._write_classes(self.classes)
            if self.info.data_yaml:
                data = yaml.safe_load(self.info.data_yaml.read_text(encoding="utf-8")) or {}
                data["names"] = {i: name for i, name in enumerate(self.classes)}
                data["nc"] = len(self.classes)
                self.info.data_yaml.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
            for image, w, h, boxes in changed:
                self.save_boxes(image, w, h, boxes)
        except Exception:
            self.classes = old_classes
            raise
        return {"images": len(changed), "annotations": removed}

    def _load_or_request_classes(self) -> List[str]:
        classes = self._load_classes_file()
        if classes:
            return classes

        classes = self._load_yaml_names()
        if classes:
            self._write_classes(classes)
            return classes

        if self.info and self.info.annotation_format not in ("yolo", "none"):
            classes = [normalize_class_name(name) for name in class_names_for(
                self.info.annotation_format, self.info.annotation_source)]
            classes = [name for name in classes if name]
            if classes:
                self._write_classes(classes)
                return classes

        # The UI can call prepare_missing_classes() to ask the user.
        return []

    def prepare_missing_classes(self, class_names: List[str]):
        ids = self.detect_class_ids()
        if not ids:
            raise DatasetError("No class IDs were found in the available label files.")
        if len(class_names) <= max(ids):
            raise DatasetError(
                f"Need names for class IDs 0 through {max(ids)}."
            )
        self.set_classes(class_names[: max(ids) + 1])

    def image_records(self) -> List[Tuple[Path, Optional[Path], str]]:
        if not self.info:
            return []

        records = []
        for split, (img_root, label_root) in self.info.split_dirs.items():
            for image in sorted(
                [p for p in img_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
                key=lambda p: str(p).lower(),
            ):
                label = None
                if label_root:
                    try:
                        rel = image.relative_to(img_root)
                        candidate = label_root / rel.with_suffix(".txt")
                        if candidate.is_file():
                            label = candidate
                    except ValueError:
                        pass
                records.append((image, label, split))
        return records

    def label_path_for(self, image_path: Path) -> Optional[Path]:
        if not self.info:
            return None
        image_path = self._resolve(image_path)
        for _, (img_root, label_root) in self.info.split_dirs.items():
            if label_root:
                try:
                    rel = image_path.relative_to(img_root)
                    return label_root / rel.with_suffix(".txt")
                except ValueError:
                    continue
        if self.info.annotation_format == "yolo" and self.info.labels_root:
            for _, (img_root, _label_root) in self.info.split_dirs.items():
                try:
                    return self.info.labels_root / image_path.relative_to(img_root).with_suffix(".txt")
                except ValueError:
                    continue
        return None

    def has_annotations(self, image_path: Path) -> bool:
        if not self.info or self.info.annotation_format == "none":
            return False
        fmt, source = self.info.annotation_format, self.info.annotation_source
        if fmt == "yolo":
            label = self.label_path_for(image_path)
            return bool(label and label.is_file())
        if fmt == "coco" and source:
            try:
                stat = source.stat()
                index = _coco_index(str(source.resolve()), stat.st_mtime_ns, stat.st_size)
                return _coco_image(image_path, index["data"], source) is not None
            except (OSError, ValueError, TypeError):
                return False
        if fmt == "voc" and source:
            return (source / image_path.with_suffix(".xml").name).is_file()
        if fmt == "labelme" and source:
            return _labelme_json_for_image(source, image_path) is not None
        if fmt == "cvat" and source:
            try:
                xml = ET.parse(source).getroot()
                if any(Path(node.get("name", "")).name.casefold() == image_path.name.casefold()
                       for node in xml.findall("image")):
                    return True
                image_root = source.parent / "images"
                frame_paths = sorted((p for p in image_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
                                     key=lambda p: str(p).casefold())
                frame_index = next((i for i, path in enumerate(frame_paths) if path.resolve() == image_path.resolve()), -1)
                return frame_index >= 0 and any(
                    box.get("frame") == str(frame_index) and box.get("outside", "0") != "1"
                    for track in xml.findall("track") for box in track.findall("box"))
            except (OSError, ET.ParseError):
                return False
        return False

    def boxes_for_image(self, image_path: Path, width: int, height: int) -> List[dict]:
        if not self.info:
            return []
        source = self.info.annotation_source
        if self.info.annotation_format == "yolo":
            source = self.label_path_for(image_path)
        return read_boxes(self.info.annotation_format, source, image_path, width, height, self.classes)

    def annotation_storage_for(self, image_path: Path) -> Optional[Path]:
        if not self.info:
            return None
        fmt = self.info.annotation_format
        if fmt == "yolo":
            return self.label_path_for(image_path)
        if fmt == "voc" and self.info.annotation_source:
            return self.info.annotation_source / image_path.with_suffix(".xml").name
        if fmt == "labelme" and self.info.annotation_source:
            return _labelme_json_for_image(self.info.annotation_source, image_path) or image_path.with_suffix(".json")
        return self.info.annotation_source

    def select_annotation_format(self, fmt: str):
        if not self.info:
            raise DatasetError("No dataset is open.")
        key = {"YOLO": "yolo", "COCO JSON": "coco", "Pascal VOC XML": "voc",
               "LabelMe JSON": "labelme", "CVAT XML": "cvat"}.get(fmt, fmt.lower())
        root = self.info.root
        source = None
        if key == "yolo":
            self.info.labels_root = root / "labels"
            self.info.labels_root.mkdir(parents=True, exist_ok=True)
            source = self.info.labels_root
        elif key == "coco":
            source = root / "annotations" / "instances_default.json"
            source.parent.mkdir(parents=True, exist_ok=True)
        elif key == "voc":
            source = root / "Annotations"
            source.mkdir(parents=True, exist_ok=True)
        elif key == "labelme":
            source = root
        elif key == "cvat":
            source = root / "annotations.xml"
        else:
            raise DatasetError(f"Unsupported annotation format: {fmt}")
        self.info.annotation_format = key
        self.info.annotation_source = source

    def save_boxes(self, image_path: Path, width: int, height: int, boxes: List[dict]) -> Path:
        if not self.info or self.info.annotation_format == "none":
            raise DatasetError("Choose an annotation format before saving.")
        target = write_boxes(self.info.annotation_format, self.info.annotation_source, self.info.root,
                             image_path, width, height, boxes, self.classes)
        clear_annotation_cache()
        if self.info.annotation_format in ("coco", "cvat"):
            self.info.annotation_source = target
        elif self.info.annotation_format == "voc":
            self.info.annotation_source = target.parent
        elif self.info.annotation_format == "yolo":
            self.info.labels_root = self.info.root / "labels"
            self.info.annotation_source = self.info.labels_root
        return target

    def validate_labels(self):
        if not self.info:
            raise DatasetError("No dataset is open.")
        if not self.classes:
            raise DatasetError("Class names are missing. Define them before preparation.")

        problems = []
        max_class = len(self.classes) - 1
        for image, _label, split in self.image_records():
            if not self.has_annotations(image):
                problems.append(f"Missing label: {image}")
                continue
            if self.info.annotation_format == "yolo":
                label = self.label_path_for(image)
                try:
                    lines = label.read_text(encoding="utf-8").splitlines() if label else []
                except OSError as exc:
                    problems.append(f"Cannot read label {label}: {exc}")
                    continue
                for line_no, line in enumerate(lines, 1):
                    parts = line.split()
                    if len(parts) != 5:
                        problems.append(f"{label}:{line_no} must have 5 fields")
                        continue
                    try:
                        cid = int(parts[0]); values = [float(value) for value in parts[1:]]
                    except ValueError:
                        problems.append(f"{label}:{line_no} contains non-numeric values")
                        continue
                    if cid < 0 or cid > max_class:
                        problems.append(f"{label}:{line_no} class {cid} is outside 0..{max_class}")
                    if any(value < 0 or value > 1 for value in values):
                        problems.append(f"{label}:{line_no} coordinates must be between 0 and 1")
                continue
            try:
                import cv2
                frame = cv2.imread(str(image))
                if frame is None:
                    problems.append(f"Cannot read image: {image}")
                    continue
                height, width = frame.shape[:2]
                boxes = self.boxes_for_image(image, width, height)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                problems.append(f"Cannot read annotation for {image}: {exc}")
                continue
            for box in boxes:
                cid = int(box["class_id"])
                x1, y1, x2, y2 = box["bbox"]
                if cid < 0 or cid > max_class:
                    problems.append(f"{image}: class {cid} is outside 0..{max_class}")
                if x2 <= x1 or y2 <= y1:
                    problems.append(f"{image}: bounding box has zero or negative area")

        return problems

    def validate_dataset(self) -> dict:
        """Return actionable dataset-wide errors and warnings without changing files."""
        if not self.info:
            raise DatasetError("No dataset is open.")
        import math
        import cv2
        issues, records = [], self.image_records()
        def report(severity, code, message, image=None, annotation=None, suggestion=""):
            issues.append({"severity": severity, "code": code, "message": message,
                           "image": str(image) if image else "",
                           "annotation": str(annotation) if annotation else "",
                           "suggestion": suggestion})
        seen_hashes = {}
        observed_ids = set()
        known = {self._resolve(image) for image, _, _ in records}
        for image, label, _split in records:
            storage = self.annotation_storage_for(image)
            if not image.is_file():
                report("error", "missing_image", "Image file is missing", image, storage,
                       "Restore the image from backup or remove its dataset reference.")
                continue
            try:
                frame = cv2.imread(str(image))
            except Exception:
                frame = None
            if frame is None:
                report("error", "corrupt_image", "Image cannot be decoded", image, storage,
                       "Replace the image with a readable copy.")
                continue
            h, w = frame.shape[:2]
            if self.info.annotation_format != "none" and not self.has_annotations(image):
                report("warning", "missing_annotation", "Image has no annotation record", image, storage,
                       "Add annotations or create an explicit empty label for a background image.")
            if self.info.annotation_format == "yolo":
                label = self.label_path_for(image)
                if label and label.is_file():
                    try:
                        lines = label.read_text(encoding="utf-8").splitlines()
                    except OSError as exc:
                        report("error", "unreadable_annotation", str(exc), image, label,
                               "Check file permissions or restore the label from backup.")
                        continue
                    if not lines:
                        report("warning", "empty_annotation", "Annotation file is empty", image, label,
                               "Keep it for a valid background image, or add missing object boxes.")
                    for line_no, line in enumerate(lines, 1):
                        parts = line.split()
                        if len(parts) != 5:
                            report("error", "invalid_row", f"Line {line_no} must contain class and four box values", image, label,
                                   "Correct the row to: class_id center_x center_y width height.")
                            continue
                        try:
                            cid = int(parts[0]); values = [float(v) for v in parts[1:]]
                        except ValueError:
                            report("error", "invalid_number", f"Line {line_no} contains a non-numeric value", image, label,
                                   "Use an integer class ID and finite normalized coordinates.")
                            continue
                        observed_ids.add(cid)
                        if cid < 0 or cid >= len(self.classes):
                            report("error", "invalid_class_id", f"Line {line_no} uses class ID {cid}", image, label,
                                   "Map the annotation to an existing class or add the intended class.")
                        cx, cy, bw, bh = values
                        if not all(math.isfinite(v) for v in values) or bw <= 0 or bh <= 0 or cx < 0 or cy < 0 or cx > 1 or cy > 1 or bw > 1 or bh > 1:
                            report("error", "invalid_coordinates", f"Line {line_no} has invalid normalized coordinates", image, label,
                                   "Use finite center/size values in (0, 1] and keep the center within the image.")
                        elif cx - bw / 2 < 0 or cy - bh / 2 < 0 or cx + bw / 2 > 1 or cy + bh / 2 > 1:
                            report("error", "box_outside_image", f"Line {line_no} extends outside the image", image, label,
                                   "Adjust the box so all four edges fit inside the image.")
            try:
                boxes = self.boxes_for_image(image, w, h)
                if not boxes and self.info.annotation_format != "yolo" and self.has_annotations(image):
                    report("warning", "empty_annotation", "Image has an annotation record but no boxes", image, storage,
                           "Keep it as a background image or review whether its annotations were lost.")
                if self.info.annotation_format != "yolo":
                    for box in boxes:
                        cid = int(box.get("class_id", -1)); observed_ids.add(cid)
                        if cid < 0 or cid >= len(self.classes):
                            report("error", "invalid_class_id", f"Invalid class ID {cid}", image, storage,
                                   "Map the annotation to an existing class.")
                        x1, y1, x2, y2 = map(float, box["bbox"])
                        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
                            report("error", "invalid_box", "Bounding box has invalid coordinates or area", image, storage,
                                   "Edit the box to have positive width and height.")
                        elif x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                            report("error", "box_outside_image", "Bounding box extends outside image boundaries", image, storage,
                                   "Move or resize the box within the image.")
            except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
                report("error", "invalid_annotation", str(exc), image, storage,
                       "Repair or restore the annotation file.")
            try:
                digest = hashlib.sha256()
                with image.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                key = digest.hexdigest()
                if key in seen_hashes:
                    report("warning", "duplicate_image", f"Duplicate image; same content as {seen_hashes[key]}", image, storage,
                           "Review the duplicate and remove it only if it is unintended.")
                else:
                    seen_hashes[key] = image
            except OSError as exc:
                report("error", "image_read_error", str(exc), image, storage,
                       "Check file permissions or restore the image.")
        for cid, name in enumerate(self.classes):
            if cid not in observed_ids:
                report("warning", "unused_class", f"Class {cid} ({name}) has no annotations", None, self.info.class_file,
                       "Remove the class only if it is not planned for future annotations.")
        normalized = [normalize_class_name(name) for name in self.classes]
        if len(set(normalized)) != len(normalized):
            report("error", "duplicate_class_name", "Class definitions contain duplicate normalized names", None,
                   self.info.class_file, "Rename duplicate classes so every class name is unique.")
        if self.info.data_yaml and self.info.data_yaml.is_file():
            try:
                payload = yaml.safe_load(self.info.data_yaml.read_text(encoding="utf-8")) or {}
                raw_names = payload.get("names", [])
                if isinstance(raw_names, dict):
                    yaml_names = [normalize_class_name(str(v)) for _, v in sorted(raw_names.items(), key=lambda pair: int(pair[0]))]
                elif isinstance(raw_names, list):
                    yaml_names = [normalize_class_name(str(v)) for v in raw_names]
                else:
                    yaml_names = []
                if yaml_names != self.classes:
                    report("error", "class_definition_mismatch", "data.yaml names differ from classes.txt", None,
                           self.info.data_yaml, "Review the class mapping, then synchronize the two definitions.")
            except Exception:
                report("error", "invalid_class_definition", "data.yaml class definitions cannot be read", None,
                       self.info.data_yaml, "Repair the YAML names mapping.")
        if self.info.annotation_format == "yolo" and self.info.labels_root and self.info.labels_root.is_dir():
            linked = {self._resolve(self.label_path_for(image)) for image, _, _ in records if self.label_path_for(image)}
            for label_path in self._label_files():
                if self._resolve(label_path) not in linked:
                    report("error", "missing_image", "Label file has no matching image", None, label_path,
                           "Restore the matching image or move this label out of the dataset.")
        elif self.info.annotation_format == "coco" and self.info.annotation_source and self.info.annotation_source.is_file():
            try:
                data = json.loads(self.info.annotation_source.read_text(encoding="utf-8"))
                root = self.info.root
                for entry in data.get("images", []):
                    ref = Path(str(entry.get("file_name", "")))
                    candidates = [root / ref, root / "images" / ref, self.info.images_root / ref]
                    if not any(path.is_file() for path in candidates):
                        report("error", "missing_image", f"COCO references missing image {ref}", ref,
                               self.info.annotation_source, "Restore the referenced image or remove its COCO image and annotation records.")
                categories = {int(item.get("id", -1)): normalize_class_name(str(item.get("name", "")))
                              for item in data.get("categories", [])}
                for name in categories.values():
                    if name and name not in self.classes:
                        report("error", "class_definition_mismatch", f"COCO category {name} is missing from the class list",
                               None, self.info.annotation_source, "Map the category to a dataset class or update the class definition.")
                for annotation in data.get("annotations", []):
                    if int(annotation.get("category_id", -1)) not in categories:
                        report("error", "invalid_class_id", "COCO annotation references an unknown category ID", None,
                               self.info.annotation_source, "Restore or map the missing category definition.")
            except (OSError, ValueError, TypeError, KeyError) as exc:
                report("error", "invalid_annotation_document", str(exc), None, self.info.annotation_source,
                       "Repair or restore the COCO JSON document.")
        elif self.info.annotation_format == "voc" and self.info.annotation_source and self.info.annotation_source.is_dir():
            image_names = {image.stem.casefold() for image, _, _ in records}
            for xml_path in self.info.annotation_source.rglob("*.xml"):
                if xml_path.stem.casefold() not in image_names:
                    report("error", "missing_image", "Pascal VOC annotation has no matching image", None, xml_path,
                           "Restore the matching image or remove the orphan annotation file.")
                try:
                    xml_root = ET.parse(xml_path).getroot()
                    for node in xml_root.findall("object/name"):
                        name = normalize_class_name(node.text or "")
                        if name and name not in self.classes:
                            report("error", "class_definition_mismatch", f"VOC object class {name} is missing from the class list",
                                   None, xml_path, "Map the object name to a class or update the class definition.")
                except (OSError, ET.ParseError) as exc:
                    report("error", "invalid_annotation_document", str(exc), None, xml_path,
                           "Repair or restore the Pascal VOC XML file.")
        elif self.info.annotation_format == "labelme" and self.info.annotation_source and self.info.annotation_source.is_dir():
            expected = {p.resolve() for image, _, _ in records
                        if (p := _labelme_json_for_image(self.info.annotation_source, image)) is not None}
            for json_path in self.info.annotation_source.rglob("*.json"):
                if json_path.resolve() not in expected:
                    try:
                        payload = json.loads(json_path.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        continue
                    if any(key in payload for key in ("shapes", "objects", "annotations", "boxes", "labels")):
                        report("error", "missing_image", "LabelMe annotation has no matching image", None, json_path,
                               "Restore the referenced image or remove the orphan annotation file.")
            for json_path in expected:
                try:
                    payload = json.loads(json_path.read_text(encoding="utf-8"))
                    entries = payload.get("shapes", payload.get("objects", payload.get("annotations", payload.get("boxes", payload.get("labels", [])))))
                    for shape in entries if isinstance(entries, list) else []:
                        name = normalize_class_name(str(shape.get("label", shape.get("class", shape.get("class_name", shape.get("name", ""))))))
                        if name and name not in self.classes:
                            report("error", "class_definition_mismatch", f"LabelMe class {name} is missing from the class list",
                                   None, json_path, "Map the shape label to a class or update the class definition.")
                except (OSError, ValueError, TypeError) as exc:
                    report("error", "invalid_annotation_document", str(exc), None, json_path,
                           "Repair or restore the LabelMe JSON file.")
        elif self.info.annotation_format == "cvat" and self.info.annotation_source and self.info.annotation_source.is_file():
            try:
                root = ET.parse(self.info.annotation_source).getroot()
                image_names = {image.name.casefold() for image, _, _ in records}
                for node in root.findall("image"):
                    if Path(node.get("name", "")).name.casefold() not in image_names:
                        report("error", "missing_image", "CVAT image annotation has no matching image", None,
                               self.info.annotation_source, "Restore the image or remove its CVAT image record.")
                names = {normalize_class_name(node.text or "") for node in root.findall(".//label/name")}
                names.update(normalize_class_name(node.get("label", "")) for node in root.findall("track"))
                names.update(normalize_class_name(node.get("label", "")) for node in root.findall(".//image/box"))
                for name in names - {""} - set(self.classes):
                    report("error", "class_definition_mismatch", f"CVAT class {name} is missing from the class list",
                           None, self.info.annotation_source, "Map the CVAT label to a class or update the class definition.")
            except (OSError, ET.ParseError) as exc:
                report("error", "invalid_annotation_document", str(exc), None, self.info.annotation_source,
                       "Repair or restore the CVAT XML document.")
        errors = sum(issue["severity"] == "error" for issue in issues)
        warnings = len(issues) - errors
        return {"errors": errors, "warnings": warnings, "issues": issues,
                "affected_images": sorted({i["image"] for i in issues if i["image"]}),
                "affected_annotations": sorted({i["annotation"] for i in issues if i["annotation"]})}

    def create_empty_yolo_labels(self, images: List[Path]) -> int:
        if not self.info or self.info.annotation_format != "yolo":
            raise DatasetError("Automatic label creation is available for YOLO datasets only.")
        known = {self._resolve(image) for image, _, _ in self.image_records()}
        targets = []
        for image in images:
            resolved = self._resolve(image)
            if resolved not in known:
                raise DatasetError(f"Image is not in this dataset: {image}")
            label = self.label_path_for(resolved)
            if label and not label.exists():
                targets.append(label)
        if not targets:
            return 0
        self._recovery_snapshot(targets)
        for path in targets:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")
        return len(targets)

    def _safe_output(self, output: Path):
        if not self.info:
            raise DatasetError("No dataset is open.")

        output = self._resolve(output)
        source = self.info.root
        app = APP_ROOT
        venv = app / "venv"

        if output == source or self._inside(output, source) or self._inside(source, output):
            raise DatasetError(
                "Unsafe output location. The YOLO output cannot be the source "
                "dataset or inside/outside it in an overlapping way."
            )
        if output == app:
            raise DatasetError("Do not use the application folder as the output.")
        if self._inside(output, venv):
            raise DatasetError("Never use the Python venv as a dataset output.")
        if output == venv:
            raise DatasetError("Never use the Python venv as a dataset output.")

        # Refuse known project-critical directories.
        protected = {app / ".git", app / "__pycache__", venv}
        for p in protected:
            if output == p or self._inside(output, p):
                raise DatasetError(f"Protected location cannot be used:\n{output}")

        return output

    def _copy_pair(self, image: Path, img_dest: Path, lbl_dest: Path, source_img_root: Path):
        rel = image.relative_to(source_img_root)
        target_img = img_dest / rel
        target_lbl = lbl_dest / rel.with_suffix(".txt")
        target_img.parent.mkdir(parents=True, exist_ok=True)
        target_lbl.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, target_img)
        import cv2
        frame = cv2.imread(str(image))
        if frame is None:
            raise DatasetError(f"Could not read image: {image}")
        height, width = frame.shape[:2]
        boxes = self.boxes_for_image(image, width, height)
        # Write in YOLO format to the destination prepared dataset.
        output_root = img_dest.parent.parent
        write_boxes("yolo", None, output_root, target_img, width, height, boxes, self.classes)

    def prepare_yolo(self, output: Path, val_fraction=0.2, seed=42):
        if not self.info:
            raise DatasetError("No dataset is open.")
        output = self._safe_output(output)

        if output.exists() and any(output.iterdir()):
            raise DatasetError(
                f"Output folder is not empty:\n{output}\n\n"
                "Choose a new folder or an empty folder. Nothing was deleted."
            )

        problems = self.validate_labels()
        if problems:
            preview = "\n".join(problems[:30])
            more = "" if len(problems) <= 30 else f"\n...and {len(problems)-30} more"
            raise DatasetError(
                f"Dataset validation failed ({len(problems)} issue(s)):\n\n{preview}{more}"
            )

        records = self.image_records()
        if not records:
            raise DatasetError("No images found.")

        output.mkdir(parents=True, exist_ok=True)
        for folder in (
            output / "images" / "train",
            output / "images" / "val",
            output / "labels" / "train",
            output / "labels" / "val",
        ):
            folder.mkdir(parents=True, exist_ok=True)

        random.Random(seed).shuffle(records)

        # If opening an existing YOLO dataset, preserve train/val when possible.
        if self.info.kind == "yolo" and any(split == "train" for _, _, split in records):
            train_records = [r for r in records if r[2] == "train"]
            val_records = [r for r in records if r[2] == "val"]
            test_records = [r for r in records if r[2] == "test"]
            if test_records:
                train_records.extend(test_records)
        else:
            n_val = max(1, round(len(records) * val_fraction)) if len(records) > 1 else 0
            val_records = records[:n_val]
            train_records = records[n_val:]

        for image, _label, _ in train_records:
            if not self.has_annotations(image):
                raise DatasetError(f"Missing label for {image}")
            source_root = next(
                img_root for _, (img_root, _) in self.info.split_dirs.items()
                if self._inside(image, img_root)
            )
            self._copy_pair(
                image,
                output / "images" / "train",
                output / "labels" / "train",
                source_root,
            )

        for image, _label, _ in val_records:
            if not self.has_annotations(image):
                raise DatasetError(f"Missing label for {image}")
            source_root = next(
                img_root for _, (img_root, _) in self.info.split_dirs.items()
                if self._inside(image, img_root)
            )
            self._copy_pair(
                image,
                output / "images" / "val",
                output / "labels" / "val",
                source_root,
            )

        data = {
            "path": str(output),
            "train": "images/train",
            "val": "images/val",
            "names": {i: name for i, name in enumerate(self.classes)},
        }
        (output / "data.yaml").write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        return output
