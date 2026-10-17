"""Validate generated RAV JSONL manifests and image references."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from rav.intelligence.vlm.data.schema import validate_bounding_box


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a RAV JSONL dataset manifest.")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to the manifest JSONL file")
    return parser.parse_args()


def validate_record(payload: dict[str, object], *, manifest_path: Path) -> tuple[bool, list[str]]:
    errors: list[str] = []
    image = str(payload.get("image", ""))
    if not image:
        errors.append("missing image path")
    if not payload.get("id"):
        errors.append("missing id")
    if not payload.get("split"):
        errors.append("missing split")
    if not payload.get("width") or not payload.get("height"):
        errors.append("missing image dimensions")
    if isinstance(payload.get("captions", []), list):
        for index, caption in enumerate(payload["captions"]):
            if not isinstance(caption, str) or not caption.strip():
                errors.append(f"invalid caption at index {index}")
    if isinstance(payload.get("objects", []), list):
        for index, obj in enumerate(payload["objects"]):
            if not isinstance(obj, dict):
                errors.append(f"object at index {index} is not a JSON object")
                continue
            category = obj.get("category")
            if not isinstance(category, str) or not category.strip():
                errors.append(f"object at index {index} is missing category")
            bbox = obj.get("bbox")
            if isinstance(bbox, list):
                if not validate_bounding_box(bbox, int(payload.get("width", 0) or 0), int(payload.get("height", 0) or 0)):
                    errors.append(f"object at index {index} has an invalid bbox")
    image_value = Path(image)
    if image_value.is_absolute():
        image_path = image_value
    else:
        if image_value.parts and image_value.parts[0] in {"data", "coco2017"}:
            image_path = (Path.cwd() / image_value).resolve()
        else:
            image_path = (manifest_path.parent.parent / image_value).resolve()
    if not image_path.exists():
        errors.append(f"missing image: {image}")
    else:
        try:
            with Image.open(image_path) as opened:
                opened.verify()
        except (FileNotFoundError, OSError, UnidentifiedImageError):
            errors.append(f"cannot open image: {image}")
    return not errors, errors


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")

    records: list[dict[str, object]] = []
    invalid = 0
    missing_images = 0
    invalid_annotations = 0
    caption_records = 0
    object_count = 0
    seen_ids: set[str] = set()

    with manifest_path.open("r", encoding="utf-8") as handle:
        for row in handle:
            if not row.strip():
                continue
            try:
                payload = json.loads(row)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON in {manifest_path}: {exc}") from exc
            records.append(payload)

    for payload in records:
        payload_id = str(payload.get("id", ""))
        if payload_id in seen_ids:
            invalid += 1
            invalid_annotations += 1
            continue
        seen_ids.add(payload_id)

        valid, errors = validate_record(payload, manifest_path=manifest_path)
        if not valid:
            invalid += 1
            invalid_annotations += len(errors)
            missing_images += sum(1 for msg in errors if "missing image" in msg)
            continue

        captions = payload.get("captions", [])
        if isinstance(captions, list) and captions:
            caption_records += 1
        objects = payload.get("objects", [])
        if isinstance(objects, list):
            object_count += len(objects)

    summary = {
        "Dataset": "COCO 2017",
        "Split": manifest_path.parent.name,
        "Records": len(records),
        "Valid": len(records) - invalid,
        "Invalid": invalid,
        "Missing images": missing_images,
        "Invalid annotations": invalid_annotations,
        "Caption records": caption_records,
        "Object instances": object_count,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
