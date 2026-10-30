from __future__ import annotations

import random
import shutil
import hashlib
import zipfile
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
        self.classes = cleaned
        self._write_classes(cleaned)

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
