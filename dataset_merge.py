"""YOLO dataset merge planning, reconciliation, validation, and execution."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import shutil
import threading
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Callable, Optional

import yaml

from config import normalize_class_name
from dataset_manager import DatasetError, DatasetManager


class MergeError(Exception):
    pass


class MergeCancelled(MergeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_merge_source(root: Path) -> dict:
    """Inspect one dataset through the existing DatasetManager API."""
    manager = DatasetManager()
    manager.open(root)
    info = manager.info
    if info is None:
        raise MergeError("Could not open the source dataset.")
    stats = manager.class_statistics()
    records = []
    for image, label, split in manager.image_records():
        try:
            import cv2
            frame = cv2.imread(str(image))
            if frame is None:
                width = height = 0
                boxes = []
            else:
                height, width = frame.shape[:2]
                boxes = manager.boxes_for_image(image, width, height)
        except Exception:
            width = height = 0
            boxes = []
        image_root = next((img_root for _name, (img_root, _label_root) in info.split_dirs.items()
                           if DatasetManager._inside(image.resolve(), img_root.resolve())), info.images_root)
        try:
            relative = image.resolve().relative_to(image_root.resolve()).as_posix()
        except ValueError:
            relative = image.name
        records.append({
            "path": str(image.resolve()), "label": str(label.resolve()) if label else "",
            "split": split, "relative": relative, "width": width, "height": height,
            "annotations": len(boxes),
            "class_ids": sorted({int(box.get("class_id", -1)) for box in boxes}),
        })
    class_counts = {entry["id"]: entry for entry in stats["classes"]}
    return {
        "root": str(info.root), "name": info.root.name, "kind": info.kind,
        "format": info.annotation_format, "mergeable": info.annotation_format == "yolo" and bool(manager.classes),
        "images": len(records), "annotations": stats["annotations"],
        "classes": list(manager.classes), "class_distribution": class_counts,
        "records": records,
    }


def _parse_yolo_label(path: Path, class_count: int) -> tuple[list[dict], str]:
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise MergeError(f"Could not read label file {path}: {exc}") from exc
    rows = []
    for line_no, line in enumerate(raw.splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise MergeError(f"{path}:{line_no} must contain five YOLO fields.")
        try:
            cid = int(parts[0]); coords = [float(value) for value in parts[1:]]
        except ValueError as exc:
            raise MergeError(f"{path}:{line_no} contains an invalid class ID or coordinate.") from exc
        if cid < 0 or cid >= class_count:
            raise MergeError(f"{path}:{line_no} uses class ID {cid}, outside the source class list.")
        if not all(math.isfinite(value) for value in coords):
            raise MergeError(f"{path}:{line_no} contains a non-finite coordinate.")
        cx, cy, width, height = coords
        if (width <= 0 or height <= 0 or not 0 <= cx <= 1 or not 0 <= cy <= 1
                or width > 1 or height > 1 or cx - width / 2 < 0 or cy - height / 2 < 0
                or cx + width / 2 > 1 or cy + height / 2 > 1):
            raise MergeError(f"{path}:{line_no} has a box outside the normalized image bounds.")
        rows.append({"class_id": cid, "coords": parts[1:]})
    return rows, hashlib.sha256(raw.encode("utf-8")).hexdigest()


def analyze_merge(sources: list[dict], target_classes: list[str], mappings: dict) -> dict:
    """Validate selected sources and map their image records without writing output."""
    roots = [str(Path(source["root"]).resolve()) for source in sources if source.get("enabled")]
    if not roots:
        raise MergeError("Select at least one source dataset.")
    if len(set(path.casefold() for path in roots)) != len(roots):
        raise MergeError("The same source dataset was added more than once.")
    classes = [normalize_class_name(name) for name in target_classes]
    if not classes or any(not name for name in classes) or len(set(classes)) != len(classes):
        raise MergeError("Unified classes must be non-empty and uniquely named.")
    records = []
    active_sources = []
    duplicate_groups = defaultdict(list)
    for source_index, source in enumerate(sources):
        if not source.get("enabled"):
            continue
        root = Path(source["root"]).resolve()
        manager = DatasetManager()
        try:
            manager.open(root)
        except (DatasetError, OSError) as exc:
            raise MergeError(f"Could not open {root}: {exc}") from exc
        if not manager.info or manager.info.annotation_format != "yolo":
            raise MergeError(f"{root} uses {manager.info.annotation_format if manager.info else 'unknown'} annotations. Only YOLO TXT datasets are mergeable in this phase.")
        if not manager.classes:
            raise MergeError(f"{root} has no class definitions.")
        source_mapping = mappings.get(str(root), {})
        normalized_mapping = {int(key): value for key, value in source_mapping.items()}
        missing = [cid for cid in range(len(manager.classes)) if cid not in normalized_mapping]
        if missing:
            raise MergeError(f"Map or explicitly exclude every class in {root}; missing IDs: {missing}.")
        for cid, target_id in normalized_mapping.items():
            if target_id is not None and (not isinstance(target_id, int) or target_id < 0 or target_id >= len(classes)):
                raise MergeError(f"Class mapping {cid} in {root} points to an invalid unified class ID.")
        selected_paths = {str(Path(p).resolve()) for p in source.get("selected_images", [])}
        all_records = manager.image_records()
        known_paths = {str(image.resolve()) for image, _, _ in all_records}
        if selected_paths.difference(known_paths):
            raise MergeError(f"The image selection for {root} contains a path that is no longer in the dataset.")
        if not selected_paths:
            raise MergeError(f"No images are selected from {root}.")
        validation = manager.validate_dataset()
        errors = [issue for issue in validation["issues"] if issue["severity"] == "error"]
        if errors:
            first = errors[0]
            raise MergeError(f"Source dataset {root} failed validation: {first['message']} ({first.get('annotation') or first.get('image')})")
        active_sources.append({
            "source_index": source_index, "root": str(root), "name": manager.info.root.name,
            "classes": list(manager.classes), "mapping": {str(cid): target for cid, target in normalized_mapping.items()},
            "validation": validation,
        })
        for image, label, split in all_records:
            image_path = str(image.resolve())
            if image_path not in selected_paths:
                continue
            label_exists = bool(label and label.is_file())
            if label_exists:
                rows, label_hash = _parse_yolo_label(label, len(manager.classes))
            else:
                # YOLO permits a missing label to represent a background image;
                # surface the warning in the preview and materialize an empty
                # label file in the result for a consistent image-label pair.
                rows, label_hash = [], hashlib.sha256(b"").hexdigest()
            mapped_rows = []
            for row in rows:
                target_id = normalized_mapping[row["class_id"]]
                if target_id is not None:
                    mapped_rows.append({"class_id": target_id, "coords": row["coords"]})
            image_hash = _sha256(image)
            source_root = next((img_root for _split, (img_root, _label_root) in manager.info.split_dirs.items()
                                if DatasetManager._inside(image.resolve(), img_root.resolve())), manager.info.images_root)
            relative = image.resolve().relative_to(source_root.resolve()).as_posix()
            key = f"{source_index}:{image_path}"
            record = {
                "key": key, "source_index": source_index, "source_root": str(root),
                "source_name": manager.info.root.name, "image": image_path,
                "label": str(label.resolve()) if label_exists else "", "image_hash": image_hash, "label_hash": label_hash,
                "relative": relative, "source_split": split, "rows": mapped_rows,
                "class_ids": sorted({row["class_id"] for row in mapped_rows}),
                "annotation_count": len(mapped_rows),
            }
            duplicate_groups[image_hash].append(len(records))
            records.append(record)
    if not records:
        raise MergeError("No selected images are available to merge.")
    duplicates = []
    for image_hash, indexes in duplicate_groups.items():
        if len(indexes) > 1:
            duplicates.append({"hash": image_hash, "records": indexes})
    return {"sources": active_sources, "target_classes": classes, "records": records,
            "duplicate_groups": duplicates}


def _select_duplicate_records(analysis: dict, policy: str, decisions: Optional[dict] = None) -> list[dict]:
    records = analysis["records"]
    groups = analysis["duplicate_groups"]
    by_index = {index for group in groups for index in group["records"]}
    keep_indexes = set(range(len(records))) - by_index
    decisions = decisions or {}
    for group in groups:
        indexes = group["records"]
        if policy == "keep_all":
            keep_indexes.update(indexes)
        elif policy == "keep_first":
            chosen = min(indexes, key=lambda i: (records[i]["source_index"], records[i]["image"].casefold()))
            keep_indexes.add(chosen)
        elif policy == "keep_latest":
            chosen = max(indexes, key=lambda i: (records[i]["source_index"], records[i]["image"].casefold()))
            keep_indexes.add(chosen)
        elif policy == "manual":
            decision = decisions.get(group["hash"])
            if decision == "all":
                keep_indexes.update(indexes)
            elif decision in {records[i]["key"] for i in indexes}:
                keep_indexes.add(next(i for i in indexes if records[i]["key"] == decision))
            else:
                raise MergeError("Choose which occurrence to keep for every duplicate group.")
        else:
            raise MergeError(f"Unsupported duplicate policy: {policy}")
    return [record for index, record in enumerate(records) if index in keep_indexes]


def _assign_splits(records: list[dict], ratios: dict, seed: int, group_sources: bool) -> dict[str, str]:
    split_names = [name for name in ("train", "val", "test") if int(ratios.get(name, 0)) > 0]
    total_percent = sum(int(ratios.get(name, 0)) for name in ("train", "val", "test"))
    if not split_names or total_percent != 100 or int(ratios.get("train", 0)) <= 0:
        raise MergeError("Train, validation, and test percentages must total 100, with train greater than zero.")
    # Connected components keep exact duplicates together; optional source grouping
    # keeps every selected image from a source dataset together as well.
    parent = list(range(len(records)))
    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    def union(left, right):
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a
    by_hash, by_source = {}, {}
    for index, record in enumerate(records):
        if record["image_hash"] in by_hash:
            union(index, by_hash[record["image_hash"]])
        else:
            by_hash[record["image_hash"]] = index
        if group_sources:
            key = record["source_root"]
            if key in by_source:
                union(index, by_source[key])
            else:
                by_source[key] = index
    components = defaultdict(list)
    for index in range(len(records)):
        components[find(index)].append(index)
    groups = list(components.values())
    if len(groups) < len(split_names):
        raise MergeError("There are too few independent image groups to populate every non-zero split. Reduce split count or turn off source grouping.")
    rng = random.Random(int(seed))
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)
    total_images = len(records)
    targets = {name: total_images * int(ratios.get(name, 0)) / 100 for name in split_names}
    counts = {name: 0 for name in split_names}
    assigned = {}
    for group_index, group in enumerate(groups):
        group_classes = {cid for index in group for cid in records[index]["class_ids"]}
        signature = tuple(sorted(group_classes))
        # Class-aware balance is the first tie breaker; global image deficit
        # determines the split while retaining deterministic results for a seed.
        group_size = len(group)
        # Minimize squared deviation from the requested image counts.
        # This fills small validation/test targets even for small datasets while
        # keeping connected duplicate/source groups intact.
        def objective(candidate):
            score = 0.0
            for name in split_names:
                projected = counts[name] + (group_size if name == candidate else 0)
                score += (projected - targets[name]) ** 2
            return score
        if group_index < len(split_names):
            # Each requested non-zero split receives an independent component
            # whenever enough components exist; small datasets otherwise tend
            # to leave validation/test empty despite a non-zero requested ratio.
            chosen = split_names[group_index]
        else:
            best = min(objective(name) for name in split_names)
            choices = [name for name in split_names if abs(objective(name) - best) < 1e-12]
            chosen = rng.choice(choices)
        for index in group:
            assigned[index] = chosen
            counts[chosen] += 1
    return {records[index]["key"]: assigned[index] for index in range(len(records))}


def finalize_merge(analysis: dict, output: Path, duplicate_policy: str, duplicate_decisions: dict,
                   ratios: dict, seed: int, group_sources: bool = False) -> dict:
    records = _select_duplicate_records(analysis, duplicate_policy, duplicate_decisions)
    if not records:
        raise MergeError("Duplicate handling left no images to merge.")
    split_map = _assign_splits(records, ratios, seed, group_sources)
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise MergeError(f"Output folder already exists: {output}")
    if not output.parent.is_dir():
        raise MergeError("Choose an existing parent folder for the merged dataset.")
    for source in analysis["sources"]:
        root = Path(source["root"]).resolve()
        if output == root or DatasetManager._inside(output, root) or DatasetManager._inside(root, output):
            raise MergeError("The output folder must not overlap a source dataset.")
    return {"output": str(output), "sources": analysis["sources"],
            "target_classes": analysis["target_classes"], "records": records,
            "duplicate_policy": duplicate_policy, "duplicate_decisions": duplicate_decisions,
            "duplicate_groups": analysis["duplicate_groups"], "ratios": ratios,
            "seed": int(seed), "group_sources": bool(group_sources), "split_map": split_map}


def execute_merge(plan: dict, progress: Optional[Callable[[int, int, str], None]] = None,
                  cancel_event: Optional[threading.Event] = None,
                  copy_file: Callable = shutil.copy2) -> dict:
    """Build to a sibling staging directory, validate, then atomically publish."""
    output = Path(plan["output"]).resolve()
    if output.exists():
        raise MergeError(f"Output folder already exists: {output}")
    if not output.parent.is_dir():
        raise MergeError("Output parent folder no longer exists.")
    staging = output.parent / f".{output.name}.merge-staging-{uuid.uuid4().hex}"
    total = len(plan["records"])
    done = 0
    output_owned = False
    completed_written = False
    log_lines = []
    provenance = []
    source_by_index = {int(source["source_index"]): source for source in plan["sources"]}
    final_classes = list(plan["target_classes"])
    try:
        staging.mkdir()
        for split in ("train", "val", "test"):
            (staging / "images" / split).mkdir(parents=True)
            (staging / "labels" / split).mkdir(parents=True)
        output_paths = set()
        for record in plan["records"]:
            if cancel_event and cancel_event.is_set():
                raise MergeCancelled("Merge cancelled before completion.")
            source = source_by_index[record["source_index"]]
            image = Path(record["image"])
            label = Path(record["label"])
            if not image.is_file() or _sha256(image) != record["image_hash"]:
                raise MergeError(f"Source image changed after preview: {image}")
            if record["label"] and (not label.is_file() or hashlib.sha256(label.read_text(encoding="utf-8").encode("utf-8")).hexdigest() != record["label_hash"]):
                raise MergeError(f"Source annotation changed after preview: {label}")
            relative = PurePosixPath(record["relative"])
            if relative.is_absolute() or ".." in relative.parts:
                raise MergeError(f"Unsafe source-relative image path: {relative}")
            source_key = re.sub(r"[^A-Za-z0-9_-]+", "_", source["name"]).strip("_")[:48] or "source"
            source_key += "_" + hashlib.sha256(source["root"].encode("utf-8")).hexdigest()[:8]
            split = plan["split_map"][record["key"]]
            relative_path = Path(source_key, *relative.parts)
            image_target = staging / "images" / split / relative_path
            label_target = staging / "labels" / split / relative_path.with_suffix(".txt")
            # YOLO labels are keyed by the image stem. Two image formats can
            # share that stem inside a source folder, so give the second label
            # a deterministic disambiguated stem rather than overwriting it.
            if label_target.exists():
                disambiguator = hashlib.sha256(record["key"].encode("utf-8")).hexdigest()[:8]
                image_target = image_target.with_name(f"{image_target.stem}_{disambiguator}{image_target.suffix}")
                label_target = label_target.with_name(f"{label_target.stem}_{disambiguator}.txt")
            rel_output = image_target.relative_to(staging).as_posix().casefold()
            if rel_output in output_paths:
                raise MergeError(f"Output filename collision for {image_target}")
            output_paths.add(rel_output)
            image_target.parent.mkdir(parents=True, exist_ok=True)
            label_target.parent.mkdir(parents=True, exist_ok=True)
            copy_file(image, image_target)
            label_lines = [f"{row['class_id']} {' '.join(row['coords'])}" for row in record["rows"]]
            label_target.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
            provenance.append({"output_image": image_target.relative_to(staging).as_posix(),
                               "output_annotation": label_target.relative_to(staging).as_posix(),
                               "source_dataset": source["root"], "source_image": record["image"],
                               "source_label": record["label"], "source_split": record["source_split"],
                               "output_split": split, "image_sha256": record["image_hash"],
                               "source_annotations": len(record["rows"])})
            done += 1
            log_lines.append(f"[{done}/{total}] {image} -> {image_target.relative_to(staging)}")
            if progress:
                progress(done, total, f"Copied {image.name}")
        for split, percent in (("train", plan["ratios"]["train"]),
                               ("val", plan["ratios"]["val"]),
                               ("test", plan["ratios"]["test"])):
            if percent == 0 and any(value == split for value in plan["split_map"].values()):
                raise MergeError(f"Images were assigned to disabled split {split}.")
        (staging / "classes.txt").write_text("\n".join(final_classes) + "\n", encoding="utf-8")
        data_yaml = {"path": str(output), "train": "images/train", "names": {i: name for i, name in enumerate(final_classes)},
                     "nc": len(final_classes)}
        if int(plan["ratios"]["val"]) > 0:
            data_yaml["val"] = "images/val"
        if int(plan["ratios"]["test"]) > 0:
            data_yaml["test"] = "images/test"
        (staging / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8")
        metadata = {
            "merge_id": uuid.uuid4().hex,
            "status": "validated_staging",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_datasets": plan["sources"],
            "source_validation": {source["root"]: source.get("validation", {}) for source in plan["sources"]},
            "unified_classes": [{"id": i, "name": name} for i, name in enumerate(final_classes)],
            "class_mappings": [{"source_dataset": source["root"], "source_classes": source["classes"],
                                "mapping": source["mapping"]} for source in plan["sources"]],
            "excluded_classes": [{"source_dataset": source["root"], "source_class_id": cid,
                                  "source_class": source["classes"][int(cid)]}
                                 for source in plan["sources"] for cid, target_id in source["mapping"].items()
                                 if target_id is None],
            "duplicate_handling": {"policy": plan["duplicate_policy"], "decisions": plan["duplicate_decisions"],
                                   "groups": plan["duplicate_groups"]},
            "split": {"ratios": plan["ratios"], "seed": plan["seed"],
                      "keep_source_datasets_together": plan["group_sources"]},
            "output_dataset": str(output), "images": provenance,
        }
        (staging / "merge_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        validation_manager = DatasetManager()
        validation_manager.open(staging)
        validation_errors = validation_manager.validate_labels()
        if validation_errors:
            raise MergeError("Output validation failed: " + validation_errors[0])
        if len(validation_manager.image_records()) != len(plan["records"]):
            raise MergeError("Output validation found an unexpected image count.")
        hash_splits = defaultdict(set)
        for item in provenance:
            hash_splits[item["image_sha256"]].add(item["output_split"])
        if any(len(splits) > 1 for splits in hash_splits.values()):
            raise MergeError("Duplicate image content leaked across output splits.")
        validation_summary = {"errors": 0, "warnings": 0, "images": len(provenance),
                              "annotations": sum(item["source_annotations"] for item in provenance),
                              "classes": len(final_classes)}
        metadata["validation"] = validation_summary
        (staging / "merge_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        log_lines.extend(["", f"Validation passed: {validation_summary}"])
        (staging / "merge.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        if cancel_event and cancel_event.is_set():
            raise MergeCancelled("Merge cancelled before publishing the output.")
        if output.exists():
            raise MergeError(f"Output folder appeared during merge: {output}")
        try:
            os.replace(staging, output)
            output_owned = True
        except OSError:
            # Windows/OneDrive can briefly deny directory rename while the
            # scanner indexes new files. A copy fallback remains safe because
            # the staging tree is validated and the output is removed on any
            # copy/finalization failure; metadata remains non-complete until
            # the final validation marker is written.
            if output.exists():
                raise
            try:
                shutil.copytree(staging, output)
                output_owned = True
            except Exception:
                shutil.rmtree(output, ignore_errors=True)
                raise
            shutil.rmtree(staging)
        metadata["status"] = "completed"
        try:
            (output / "merge_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
            completed_written = True
        except Exception:
            shutil.rmtree(output, ignore_errors=True)
            raise
        if progress:
            progress(total, total, "Merged dataset validated and published.")
        return {"output": str(output), "validation": validation_summary,
                "metadata": str(output / "merge_metadata.json"), "log": str(output / "merge.log")}
    except Exception:
        if output_owned and not completed_written:
            try:
                shutil.rmtree(output)
            except OSError:
                pass
        try:
            if staging.exists() and staging.parent == output.parent and staging.name.startswith(f".{output.name}.merge-staging-"):
                shutil.rmtree(staging)
        except OSError:
            pass
        raise
