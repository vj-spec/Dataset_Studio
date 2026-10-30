"""Read and write rectangular object annotations in common dataset formats."""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from functools import lru_cache

from config import IMAGE_EXTENSIONS, normalize_class_name


SUPPORTED_FORMATS = ("YOLO", "COCO JSON", "Pascal VOC XML", "LabelMe JSON", "CVAT XML")
FORMAT_KEYS = {
    "YOLO": "yolo",
    "COCO JSON": "coco",
    "Pascal VOC XML": "voc",
    "LabelMe JSON": "labelme",
    "CVAT XML": "cvat",
}


@lru_cache(maxsize=128)
def _cached_json(path_text: str, modified_ns: int, size: int):
    try:
        return json.loads(Path(path_text).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}


def _json(path):
    try:
        stat = path.stat()
    except OSError:
        return {}
    return _cached_json(str(path.resolve()), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=16)
def _coco_index(path_text: str, modified_ns: int, size: int):
    data = _json(Path(path_text))
    by_name = {}
    by_stem = {}
    by_path = {}
    for image in data.get("images", []) if isinstance(data, dict) else []:
        name = _normalized_path(str(image.get("file_name", image.get("filename", image.get("name", image.get("path", ""))))))
        if not name:
            continue
        by_name.setdefault(Path(name).name, []).append(image)
        by_stem.setdefault(Path(name).stem, []).append(image)
        by_path.setdefault(name, []).append(image)
    annotations = {}
    for annotation in data.get("annotations", []) if isinstance(data, dict) else []:
        try:
            annotations.setdefault(int(annotation["image_id"]), []).append(annotation)
        except (KeyError, TypeError, ValueError):
            continue
    return {"data": data, "by_name": by_name, "by_stem": by_stem, "by_path": by_path,
            "annotations": annotations}


@lru_cache(maxsize=32)
def _labelme_index(source_text: str, modified_ns: int):
    source = Path(source_text)
    dataset_root = source if (source / "images").is_dir() else next(
        (parent for parent in source.parents if (parent / "images").is_dir()), source.parent)
    by_path = {}
    by_name = {}
    by_stem = {}
    by_layout = {}
    for candidate in source.rglob("*.json"):
        data = _json(candidate)
        by_name.setdefault(candidate.name.casefold(), []).append(candidate)
        by_stem.setdefault(candidate.stem.casefold(), []).append(candidate)
        try:
            rel = candidate.relative_to(source)
            for extension in IMAGE_EXTENSIONS:
                by_layout.setdefault(rel.with_suffix(extension).as_posix().casefold(), []).append(candidate)
        except ValueError:
            pass
        if not isinstance(data, dict):
            continue
        for key in ("imagePath", "image", "image_file", "image_name", "file_name", "filename", "path"):
            reference = data.get(key)
            if not isinstance(reference, str) or not reference.strip():
                continue
            ref = reference.strip()
            by_path.setdefault(_normalized_path(ref), []).append(candidate)
            for base in (candidate.parent, source, source.parent, dataset_root):
                try:
                    resolved = _normalized_path(str((base / ref).resolve()))
                    by_path.setdefault(resolved, []).append(candidate)
                except OSError:
                    pass
            ref_name = Path(ref.replace("\\", "/")).name.casefold()
            by_name.setdefault(ref_name, []).append(candidate)
            by_stem.setdefault(Path(ref_name).stem.casefold(), []).append(candidate)
    return {"by_path": by_path, "by_name": by_name, "by_stem": by_stem, "by_layout": by_layout,
            "dataset_root": dataset_root}


def clear_annotation_cache():
    _cached_json.cache_clear()
    _coco_index.cache_clear()
    _labelme_index.cache_clear()


def detect_annotation_format(root: Path, images_root: Path, labels_root: Path | None) -> tuple[str, Path | None]:
    annotations = root / "annotations"
    coco_files = sorted(annotations.rglob("*.json")) if annotations.is_dir() else []
    for path in sorted(coco_files, key=lambda p: (not p.name.lower().startswith("instances"), p.name.lower())):
        data = _json(path)
        if isinstance(data, dict) and isinstance(data.get("images"), list) and isinstance(data.get("annotations"), list):
            return "coco", path

    cvat = root / "annotations.xml"
    if cvat.is_file():
        try:
            if ET.parse(cvat).getroot().tag == "annotations":
                return "cvat", cvat
        except (OSError, ET.ParseError):
            pass

    voc_root = root / "Annotations"
    if voc_root.is_dir() and any(voc_root.glob("*.xml")):
        return "voc", voc_root

    if images_root == root and any(root.glob("*.json")):
        if any(p.is_file() and p.suffix.lower() == ".json" for p in root.iterdir()):
            return "labelme", root

    # LabelMe projects commonly keep their per-image JSON sidecars in a
    # separate annotations/ folder instead of beside the source images.
    if annotations.is_dir():
        json_files = list(annotations.rglob("*.json"))
        image_stems = {path.stem.casefold() for path in images_root.rglob("*")
                       if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS}
        for path in json_files:
            data = _json(path)
            if isinstance(data, dict) and (
                "shapes" in data or
                any(key in data for key in ("imagePath", "image", "image_file", "image_name", "file_name", "filename")) or
                (path.stem.casefold() in image_stems and any(key in data for key in ("objects", "annotations", "boxes", "labels")))
            ):
                return "labelme", annotations

    if labels_root and labels_root.is_dir() and any(labels_root.rglob("*.txt")):
        return "yolo", labels_root
    if annotations.is_dir() and any(annotations.rglob("*.txt")):
        return "yolo", annotations
    return "none", None


def class_names_for(fmt: str, source: Path | None) -> list[str]:
    names: set[str] = set()
    if not source:
        return []
    try:
        if fmt == "coco":
            names = {str(item.get("name", "")).strip() for item in _json(source).get("categories", [])}
        elif fmt == "voc":
            for path in source.rglob("*.xml"):
                for node in ET.parse(path).getroot().findall("object/name"):
                    if node.text:
                        names.add(node.text.strip())
        elif fmt == "labelme":
            for path in source.rglob("*.json"):
                data = _json(path)
                entries = data.get("shapes", data.get("objects", data.get("annotations", data.get("boxes", data.get("labels", [])))))
                for shape in entries if isinstance(entries, list) else []:
                    name = shape.get("label", shape.get("class", shape.get("class_name", shape.get("name"))))
                    if name:
                        names.add(str(name).strip())
        elif fmt == "cvat":
            root = ET.parse(source).getroot()
            names = {node.text.strip() for node in root.findall("./meta/task/labels/label/name") if node.text}
            if not names:
                names = {node.get("label", "").strip() for node in root.findall(".//box") if node.get("label")}
            if not names:
                names = {node.get("label", "").strip() for node in root.findall("track") if node.get("label")}
    except (OSError, ET.ParseError, ValueError, TypeError):
        return []
    return sorted({normalize_class_name(name) for name in names if name})


def _normalized_path(value: str) -> str:
    return "/".join(part for part in value.replace("\\", "/").split("/") if part not in ("", ".")).casefold()


def _coco_image(path: Path, data: dict, source: Path | None = None):
    if source:
        try:
            stat = source.stat()
            index = _coco_index(str(source.resolve()), stat.st_mtime_ns, stat.st_size)
            exact = {_normalized_path(str(path))}
            dataset_root = source if (source / "images").is_dir() else next(
                (parent for parent in source.parents if (parent / "images").is_dir()), source.parent.parent)
            for base in (dataset_root, dataset_root / "images"):
                try:
                    exact.add(_normalized_path(str(path.relative_to(base))))
                except ValueError:
                    pass
            matches = {id(image): image for key in exact for image in index["by_path"].get(key, [])}
            if len(matches) == 1:
                return next(iter(matches.values()))
            matches = index["by_name"].get(path.name.casefold(), [])
            if len(matches) == 1:
                return matches[0]
            matches = index["by_stem"].get(path.stem.casefold(), [])
            return matches[0] if len(matches) == 1 else None
        except OSError:
            return None
    images = data.get("images", [])
    matches = [image for image in images if Path(str(image.get("file_name", "")).replace("\\", "/")).name.casefold() == path.name.casefold()]
    return matches[0] if len(matches) == 1 else None


def _labelme_json_for_image(source: Path, image_path: Path) -> Path | None:
    if not source.is_dir():
        return None
    try:
        stat = source.stat()
        index = _labelme_index(str(source.resolve()), stat.st_mtime_ns)
    except OSError:
        return None
    dataset_root = index["dataset_root"]
    exact = {_normalized_path(str(image_path))}
    try:
        image_rel = image_path.relative_to(dataset_root / "images")
    except ValueError:
        image_rel = Path(image_path.name)
    exact.add(_normalized_path(image_rel.as_posix()))
    matches = {candidate for key in exact for candidate in index["by_path"].get(key, [])}
    if len(matches) == 1:
        return next(iter(matches))
    if len(matches) > 1:
        return None
    matches = index["by_layout"].get(image_rel.as_posix().casefold(), [])
    if len(matches) == 1:
        return matches[0]
    # The common convention is image.ext + image.json. Basename matching is
    # safe when unique, and stem matching is only used if it is also unique.
    direct = source / image_path.with_suffix(".json").name
    if direct.is_file():
        return direct
    matches = set(index["by_name"].get(image_path.with_suffix(".json").name.casefold(), []))
    if len(matches) == 1:
        return next(iter(matches))
    matches = set(index["by_stem"].get(image_path.stem.casefold(), []))
    if len(matches) == 1:
        return next(iter(matches))
    fallback = source / f"{image_path.stem}.json"
    if fallback.is_file():
        return fallback
    return None


def read_boxes(fmt: str, source: Path | None, image_path: Path, width: int, height: int,
               class_names: list[str]) -> list[dict]:
    """Return boxes as pixel xyxy coordinates and zero-based class IDs."""
    if fmt == "none" or not source:
        return []
    boxes = []
    try:
        if fmt == "yolo":
            label = source if source.is_file() else source / image_path.with_suffix(".txt").name
            for line in label.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                cid = int(parts[0]); cx, cy, bw, bh = map(float, parts[1:])
                boxes.append({"bbox": (int((cx-bw/2)*width), int((cy-bh/2)*height),
                                         int((cx+bw/2)*width), int((cy+bh/2)*height)), "class_id": cid})
        elif fmt == "coco":
            stat = source.stat()
            index = _coco_index(str(source.resolve()), stat.st_mtime_ns, stat.st_size)
            image = _coco_image(image_path, index["data"], source)
            if not image:
                return []
            category_ids = {int(cat["id"]): class_names.index(normalize_class_name(str(cat["name"])))
                            for cat in index["data"].get("categories", [])
                            if normalize_class_name(str(cat.get("name", ""))) in class_names}
            for item in index["annotations"].get(int(image["id"]), []):
                cid = category_ids.get(int(item.get("category_id", -1)))
                bbox = item.get("bbox", [])
                if cid is not None and len(bbox) >= 4:
                    x, y, w, h = map(float, bbox[:4])
                    boxes.append({"bbox": (int(x), int(y), int(x+w), int(y+h)), "class_id": cid})
        elif fmt == "voc":
            xml_path = source / image_path.with_suffix(".xml").name
            if not xml_path.is_file():
                xml_path = source / f"{image_path.stem}.xml"
            root = ET.parse(xml_path).getroot()
            for obj in root.findall("object"):
                name = normalize_class_name(obj.findtext("name") or "")
                coords = obj.find("bndbox")
                if name in class_names and coords is not None:
                    x1, y1, x2, y2 = (float(coords.findtext(k, "0")) for k in ("xmin", "ymin", "xmax", "ymax"))
                    boxes.append({"bbox": (int(x1)-1, int(y1)-1, int(x2), int(y2)), "class_id": class_names.index(name)})
        elif fmt == "labelme":
            json_path = _labelme_json_for_image(source, image_path)
            if not json_path:
                return []
            data = _json(json_path)
            entries = data.get("shapes", data.get("objects", data.get("annotations", data.get("boxes", data.get("labels", [])))))
            for shape in entries if isinstance(entries, list) else []:
                name = normalize_class_name(str(shape.get("label", shape.get("class", shape.get("class_name", shape.get("name", ""))))))
                try:
                    cid = class_names.index(name) if name in class_names else int(shape.get("class_id", shape.get("category_id", -1)))
                except (TypeError, ValueError):
                    cid = -1
                if cid < 0 or cid >= len(class_names):
                    continue
                points = shape.get("points", [])
                if points:
                    xs, ys = zip(*[(float(p[0]), float(p[1])) for p in points])
                    bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
                else:
                    raw = shape.get("bbox", shape.get("box", shape.get("bounding_box")))
                    if isinstance(raw, dict):
                        x1 = raw.get("x1", raw.get("xmin", raw.get("left", raw.get("x", 0))))
                        y1 = raw.get("y1", raw.get("ymin", raw.get("top", raw.get("y", 0))))
                        x2 = raw.get("x2", raw.get("xmax", raw.get("right")))
                        y2 = raw.get("y2", raw.get("ymax", raw.get("bottom")))
                        if x2 is None and "width" in raw:
                            x2 = float(x1) + float(raw["width"])
                        if y2 is None and "height" in raw:
                            y2 = float(y1) + float(raw["height"])
                        bbox = (int(float(x1)), int(float(y1)), int(float(x2 or 0)), int(float(y2 or 0)))
                    elif isinstance(raw, (list, tuple)) and len(raw) >= 4:
                        x, y, third, fourth = map(float, raw[:4])
                        box_format = str(shape.get("bbox_format", "xywh")).lower()
                        bbox = (int(x), int(y), int(third), int(fourth)) if box_format in ("xyxy", "pascal_voc") else (int(x), int(y), int(x+third), int(y+fourth))
                    elif all(key in shape for key in ("x", "y", "width", "height")):
                        x, y = float(shape["x"]), float(shape["y"])
                        bbox = (int(x), int(y), int(x+float(shape["width"])), int(y+float(shape["height"])))
                    else:
                        continue
                boxes.append({"bbox": bbox, "class_id": cid})
        elif fmt == "cvat":
            root = ET.parse(source).getroot()
            image_node = next((node for node in root.findall("image")
                               if Path(node.get("name", "")).name.casefold() == image_path.name.casefold()), None)
            nodes = image_node.findall("box") if image_node is not None else []
            if not nodes:
                image_root = source.parent / "images"
                frame_paths = sorted((p for p in image_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
                                     key=lambda p: str(p).casefold())
                frame_index = next((i for i, path in enumerate(frame_paths) if path.resolve() == image_path.resolve()), -1)
                if frame_index >= 0:
                    for track in root.findall("track"):
                        for box in track.findall("box"):
                            if int(box.get("frame", -1)) == frame_index and box.get("outside", "0") != "1":
                                box.set("label", track.get("label", ""))
                                nodes.append(box)
            for node in nodes:
                name = normalize_class_name(node.get("label", ""))
                if name in class_names:
                    boxes.append({"bbox": tuple(int(float(node.get(k, "0"))) for k in ("xtl", "ytl", "xbr", "ybr")),
                                  "class_id": class_names.index(name)})
    except (OSError, ValueError, KeyError, TypeError, ET.ParseError):
        return []
    return boxes


def _normalized_rows(boxes, width, height):
    rows = []
    for box in boxes:
        cid = int(box["class_id"])
        x1, y1, x2, y2 = map(float, box["bbox"])
        x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
        y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        rows.append((cid, x1, y1, x2, y2))
    return rows


def write_boxes(fmt: str, source: Path | None, root: Path, image_path: Path, width: int, height: int,
                boxes: list[dict], class_names: list[str]) -> Path:
    """Persist the boxes for one image in the chosen dataset format."""
    rows = _normalized_rows(boxes, width, height)
    rel = image_path.relative_to(root / "images") if (root / "images").is_dir() and image_path.is_relative_to(root / "images") else Path(image_path.name)
    if fmt == "yolo":
        labels = root / "labels"
        target = labels / rel.with_suffix(".txt")
        target.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"{cid} {((x1+x2)/2)/width:.6f} {((y1+y2)/2)/height:.6f} {(x2-x1)/width:.6f} {(y2-y1)/height:.6f}"
                 for cid, x1, y1, x2, y2 in rows]
        target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    elif fmt == "coco":
        target = source if source and source.suffix.lower() == ".json" else root / "annotations" / "instances_default.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        data = _json(target) if target.is_file() else {"info": {"description": root.name}, "licenses": [], "images": [], "annotations": [], "categories": []}
        if not data:
            data = {"info": {"description": root.name}, "licenses": [], "images": [], "annotations": [], "categories": []}
        for index, name in enumerate(class_names, 1):
            if not any(cat.get("name") == name for cat in data.setdefault("categories", [])):
                data["categories"].append({"id": index, "name": name, "supercategory": ""})
        image = _coco_image(image_path, data)
        if image is None:
            image_id = max((int(item.get("id", 0)) for item in data.setdefault("images", [])), default=0) + 1
            image = {"id": image_id, "file_name": str(rel).replace("\\", "/"), "width": width, "height": height}
            data["images"].append(image)
        else:
            image_id = int(image["id"])
        data.setdefault("annotations", [])
        old_ids = [int(a["id"]) for a in data["annotations"] if int(a.get("image_id", -1)) == image_id and "id" in a]
        data["annotations"] = [a for a in data["annotations"] if int(a.get("image_id", -1)) != image_id]
        next_id = max((int(a.get("id", 0)) for a in data["annotations"]), default=0) + 1
        category_by_name = {cat["name"]: int(cat["id"]) for cat in data["categories"]}
        for cid, x1, y1, x2, y2 in rows:
            data["annotations"].append({"id": next_id, "image_id": image_id, "category_id": category_by_name[class_names[cid]],
                "bbox": [x1, y1, x2-x1, y2-y1], "area": (x2-x1)*(y2-y1), "iscrowd": 0, "segmentation": []})
            next_id += 1
        target.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    elif fmt == "voc":
        target = root / "Annotations" / image_path.with_suffix(".xml").name
        target.parent.mkdir(parents=True, exist_ok=True)
        annotation = ET.Element("annotation")
        ET.SubElement(annotation, "filename").text = image_path.name
        size = ET.SubElement(annotation, "size")
        ET.SubElement(size, "width").text = str(width); ET.SubElement(size, "height").text = str(height); ET.SubElement(size, "depth").text = "3"
        ET.SubElement(annotation, "segmented").text = "0"
        for cid, x1, y1, x2, y2 in rows:
            obj = ET.SubElement(annotation, "object")
            ET.SubElement(obj, "name").text = class_names[cid]
            ET.SubElement(obj, "pose").text = "Unspecified"; ET.SubElement(obj, "truncated").text = "0"; ET.SubElement(obj, "difficult").text = "0"
            box = ET.SubElement(obj, "bndbox")
            for key, value in zip(("xmin", "ymin", "xmax", "ymax"), (int(x1)+1, int(y1)+1, int(x2), int(y2))):
                ET.SubElement(box, key).text = str(value)
        ET.indent(annotation, space="  ")
        target.write_bytes(ET.tostring(annotation, encoding="utf-8", xml_declaration=True))
    elif fmt == "labelme":
        target = image_path.with_suffix(".json")
        payload = {"version": "5.0.0", "flags": {}, "shapes": [], "imagePath": image_path.name,
                   "imageData": None, "imageHeight": height, "imageWidth": width}
        for cid, x1, y1, x2, y2 in rows:
            payload["shapes"].append({"label": class_names[cid], "points": [[x1, y1], [x2, y2]],
                "group_id": None, "description": "", "shape_type": "rectangle", "flags": {}})
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    elif fmt == "cvat":
        target = source if source and source.suffix.lower() == ".xml" else root / "annotations.xml"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file():
            xml = ET.parse(target).getroot()
        else:
            xml = ET.Element("annotations")
            ET.SubElement(xml, "version").text = "1.1"
            meta = ET.SubElement(xml, "meta")
            task = ET.SubElement(meta, "task")
            labels = ET.SubElement(task, "labels")
            for name in class_names:
                label = ET.SubElement(labels, "label")
                ET.SubElement(label, "name").text = name
                ET.SubElement(label, "color").text = "#ffffff"
                ET.SubElement(label, "type").text = "rectangle"
                ET.SubElement(label, "attributes")
        for node in list(xml.findall("image")):
            if Path(node.get("name", "")).name.casefold() == image_path.name.casefold():
                xml.remove(node)
        image_node = ET.SubElement(xml, "image", {"id": "0", "name": image_path.name, "width": str(width), "height": str(height)})
        for cid, x1, y1, x2, y2 in rows:
            ET.SubElement(image_node, "box", {"label": class_names[cid], "occluded": "0", "source": "manual",
                "xtl": f"{x1:.2f}", "ytl": f"{y1:.2f}", "xbr": f"{x2:.2f}", "ybr": f"{y2:.2f}", "z_order": "0"})
        ET.indent(xml, space="  ")
        target.write_bytes(ET.tostring(xml, encoding="utf-8", xml_declaration=True))
    else:
        raise ValueError(f"Unsupported annotation format: {fmt}")
    return target
