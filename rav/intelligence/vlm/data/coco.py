"""COCO dataset inspection and manifest generation utilities."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from rav.intelligence.vlm.data.manifest import write_manifest
from rav.intelligence.vlm.data.schema import ManifestRecord, ObjectRecord, SplitName


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"COCO annotation file {path} must contain a JSON object")
    return data


class CocoDatasetInspector:
    """Inspect the actual COCO 2017 data layout and annotation coverage."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.annotations_dir = self.root / "annotations"

    @property
    def split_dirs(self) -> dict[str, Path]:
        return {
            "train": self.root / "train2017",
            "val": self.root / "val2017",
            "test": self.root / "test2017",
        }

    def inspect(self) -> dict[str, Any]:
        split_counts: dict[str, int] = {}
        image_dim_stats: dict[str, dict[str, list[int]]] = {}
        annotation_files = (
            sorted(self.annotations_dir.glob("*.json"))
            if self.annotations_dir.exists()
            else []
        )
        category_map: dict[str, set[str]] = {}
        captions_count = 0
        object_instances = 0

        for split_name, split_dir in self.split_dirs.items():
            files = list(split_dir.glob("*.jpg")) if split_dir.exists() else []
            split_counts[split_name] = len(files)
            image_dim_stats[split_name] = {"widths": [], "heights": []}

        for file_path in annotation_files:
            data = _load_json(file_path)
            if "categories" in data:
                category_map[file_path.name] = {category["name"] for category in data["categories"]}
            if "annotations" in data and file_path.name.startswith("captions_"):
                captions_count += len(data["annotations"])
            if "annotations" in data and file_path.name.startswith("instances_"):
                object_instances += len(data["annotations"])

            if "images" in data:
                for image in data["images"]:
                    split_name = (
                        file_path.name.replace("captions_", "")
                        .replace("instances_", "")
                        .replace("person_keypoints_", "")
                        .replace("2017.json", "")
                    )
                    if split_name not in self.split_dirs:
                        continue
                    image_dim_stats.setdefault(split_name, {"widths": [], "heights": []})
                    image_dim_stats[split_name]["widths"].append(int(image.get("width", 0)))
                    image_dim_stats[split_name]["heights"].append(int(image.get("height", 0)))

        missing = []
        for split_name, split_dir in self.split_dirs.items():
            if not split_dir.exists():
                missing.append(f"{split_name} folder missing at {split_dir}")

        return {
            "dataset_root": str(self.root),
            "train_images": split_counts.get("train", 0),
            "val_images": split_counts.get("val", 0),
            "test_images": split_counts.get("test", 0),
            "annotation_files": [path.name for path in annotation_files],
            "annotation_types": sorted({path.name.split("_")[0] for path in annotation_files}),
            "annotation_file_count": len(annotation_files),
            "categories": sorted(
                {category for categories in category_map.values() for category in categories}
            ),
            "category_count": len(
                {category for categories in category_map.values() for category in categories}
            ),
            "caption_count": captions_count,
            "object_instance_count": object_instances,
            "image_dimension_summary": image_dim_stats,
            "missing_or_inconsistent": missing,
        }


def build_manifest_for_split(
    root: str | Path,
    *,
    split: SplitName,
    output_path: str | Path | None = None,
    include_captions: bool = True,
) -> list[ManifestRecord]:
    """Build a manifest for a single split and return its records."""

    dataset_root = Path(root)
    split_dir = dataset_root / f"{split}2017"
    if not split_dir.exists():
        raise FileNotFoundError(f"Expected COCO split dir does not exist: {split_dir}")

    annotations_dir = dataset_root / "annotations"
    captions_path = annotations_dir / f"captions_{split}2017.json"
    instances_path = annotations_dir / f"instances_{split}2017.json"

    category_by_id: dict[int, str] = {}
    image_metadata: dict[int, dict[str, Any]] = {}
    caption_by_image: dict[int, list[str]] = {}
    object_by_image: dict[int, list[dict[str, Any]]] = {}

    if instances_path.exists():
        instance_data = _load_json(instances_path)
        for category in instance_data.get("categories", []):
            category_by_id[int(category["id"])] = str(category["name"])
        for image in instance_data.get("images", []):
            image_metadata[int(image["id"])] = {
                "width": int(image["width"]),
                "height": int(image["height"]),
                "file_name": str(image["file_name"]),
            }
        for annotation in instance_data.get("annotations", []):
            image_id = int(annotation["image_id"])
            obj = {
                "id": str(annotation.get("id", image_id)),
                "category": category_by_id.get(
                    int(annotation.get("category_id", -1)),
                    "unknown",
                ),
                "bbox": [
                    float(value)
                    for value in annotation.get("bbox", [0.0, 0.0, 0.0, 0.0])
                ],
                "area": float(annotation.get("area", 0.0)),
                "segmentation": annotation.get("segmentation"),
                "source": "coco2017",
            }
            object_by_image.setdefault(image_id, []).append(obj)

    if include_captions and captions_path.exists():
        caption_data = _load_json(captions_path)
        for image in caption_data.get("images", []):
            image_metadata.setdefault(
                int(image["id"]),
                {
                    "width": int(image.get("width", 0)),
                    "height": int(image.get("height", 0)),
                    "file_name": str(image.get("file_name", "")),
                },
            )
        for annotation in caption_data.get("annotations", []):
            image_id = int(annotation["image_id"])
            caption_by_image.setdefault(image_id, []).append(
                str(annotation.get("caption", ""))
            )

    records: list[ManifestRecord] = []
    image_files = sorted(split_dir.glob("*.jpg"))
    for image_path in image_files:
        image_id = int(image_path.stem)
        metadata = image_metadata.get(image_id, {})
        width = int(float(metadata.get("width", 0) or 0))
        height = int(float(metadata.get("height", 0) or 0))
        if width <= 0 or height <= 0:
            with Image.open(image_path) as opened:
                width, height = opened.size
        root_path = Path(root)
        base_path = root_path.resolve() if root_path.is_absolute() else root_path
        relative_under_root = image_path.relative_to(base_path)
        display_root = root_path
        if root_path.is_absolute() and root_path.is_relative_to(Path.cwd()):
            display_root = root_path.relative_to(Path.cwd())
        image_rel = str((display_root / relative_under_root).as_posix()).replace("\\", "/")
        if split == "test":
            captions: list[str] = []
            objects: list[ObjectRecord] = []
            category_names: list[str] = []
        else:
            captions = caption_by_image.get(image_id, [])
            objects: list[ObjectRecord] = []
            for obj in object_by_image.get(image_id, []):
                bbox = obj.get("bbox")
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                x, y, bbox_width, bbox_height = [float(value) for value in bbox]
                if bbox_width <= 0 or bbox_height <= 0:
                    continue
                objects.append(
                    ObjectRecord(
                        id=str(obj["id"]),
                        category=str(obj["category"]),
                        bbox=[x, y, bbox_width, bbox_height],
                        area=(
                            float(obj.get("area", 0.0))
                            if obj.get("area") is not None
                            else None
                        ),
                        segmentation=obj.get("segmentation"),
                        source=str(obj.get("source", "coco2017")),
                    )
                )
            category_names = sorted({obj.category for obj in objects})
        record = ManifestRecord(
            id=f"coco_{split}_{image_id}",
            image_id=image_id,
            source_dataset="coco2017",
            source="coco2017",
            split=split,
            image=image_rel,
            image_path=image_rel,
            width=width or 0,
            height=height or 0,
            captions=captions,
            objects=objects,
            category_names=category_names,
        )
        records.append(record)

    if output_path is not None:
        write_manifest(output_path, records)
    return records


def create_coco_statistics(
    root: str | Path,
    *,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Generate summary statistics for a COCO manifest or dataset."""

    dataset_root = Path(root)
    if manifest_path is None:
        manifest_path = dataset_root / "data" / "train" / "manifest.jsonl"
    records = [
        ManifestRecord.model_validate(json.loads(line))
        for line in Path(manifest_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    category_counter: Counter[str] = Counter()
    caption_counter = 0
    object_total = 0
    width_values: list[int] = []
    height_values: list[int] = []

    for record in records:
        width_values.append(record.width)
        height_values.append(record.height)
        caption_counter += len(record.captions)
        object_total += len(record.objects)
        category_counter.update(record.category_names)

    total = len(records)
    return {
        "records": total,
        "captions": caption_counter,
        "objects": object_total,
        "average_objects_per_image": (object_total / total) if total else 0.0,
        "average_captions_per_image": (caption_counter / total) if total else 0.0,
        "category_counts": dict(sorted(category_counter.items())),
        "image_widths": width_values,
        "image_heights": height_values,
        "missing_or_corrupt": 0,
    }
