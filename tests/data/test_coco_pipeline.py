"""Tests for the COCO manifest and dataset pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from rav.intelligence.vlm.data.coco import build_manifest_for_split
from rav.intelligence.vlm.data.loader import RAVVisionDataset
from rav.intelligence.vlm.data.schema import validate_bounding_box


def test_build_manifest_for_split(tmp_path: Path) -> None:
    dataset_root = tmp_path / "coco2017"
    annotations_dir = dataset_root / "annotations"
    split_dir = dataset_root / "train2017"
    split_dir.mkdir(parents=True)
    annotations_dir.mkdir(parents=True)

    image_path = split_dir / "000000000001.jpg"
    Image.new("RGB", (64, 64), color="white").save(image_path)

    instance_payload = {
        "images": [
            {"id": 1, "file_name": "000000000001.jpg", "width": 64, "height": 64}
        ],
        "annotations": [
            {
                "id": 10,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 20, 15, 25],
                "area": 375,
            }
        ],
        "categories": [{"id": 1, "name": "person"}],
    }
    caption_payload = {
        "images": [
            {"id": 1, "file_name": "000000000001.jpg", "width": 64, "height": 64}
        ],
        "annotations": [{"image_id": 1, "caption": "A person standing in a hallway."}],
    }
    (annotations_dir / "instances_train2017.json").write_text(
        json.dumps(instance_payload),
        encoding="utf-8",
    )
    (annotations_dir / "captions_train2017.json").write_text(
        json.dumps(caption_payload),
        encoding="utf-8",
    )

    records = build_manifest_for_split(
        dataset_root,
        split="train",
        output_path=tmp_path / "train_manifest.jsonl",
    )

    assert len(records) == 1
    record = records[0]
    assert record.captions == ["A person standing in a hallway."]
    assert record.objects[0].category == "person"
    assert record.image.endswith("train2017/000000000001.jpg")


def test_validate_bounding_box() -> None:
    assert validate_bounding_box([10, 20, 15, 25], 64, 64) is True
    assert validate_bounding_box([50, 50, 20, 25], 64, 64) is False


def test_loader_reads_image_and_metadata(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (32, 32), color="black").save(image_path)
    record = {
        "id": "sample-1",
        "image_id": 1,
        "source_dataset": "coco2017",
        "source": "coco2017",
        "split": "val",
        "image": "sample.jpg",
        "image_path": "sample.jpg",
        "width": 32,
        "height": 32,
        "captions": ["A black square."],
        "objects": [
            {"category": "object", "bbox": [0, 0, 10, 10], "area": 100}
        ],
        "category_names": ["object"],
    }
    manifest_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    dataset = RAVVisionDataset(manifest_path, image_root=tmp_path)
    sample = dataset[0]

    assert sample["id"] == "sample-1"
    assert sample["image"].size == (32, 32)
    assert sample["captions"] == ["A black square."]
