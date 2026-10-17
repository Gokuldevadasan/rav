"""Dataset-adapter interfaces for pluggable source datasets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from rav.intelligence.vlm.data.schema import ManifestRecord


class BaseDataAdapter(ABC):
    """Common interface for converting raw source data into a RAV manifest record."""

    @abstractmethod
    def adapt(self, raw_record: dict[str, Any], *, split: str) -> ManifestRecord:
        raise NotImplementedError


class CocoDataAdapter(BaseDataAdapter):
    """Adapter for COCO-style annotations to the RAV unified schema."""

    def adapt(self, raw_record: dict[str, Any], *, split: str) -> ManifestRecord:
        image_path = str(Path(raw_record["image"]))
        object_list = [
            {
                "id": str(obj.get("id")),
                "category": obj.get("category", "unknown"),
                "bbox": obj.get("bbox"),
                "area": obj.get("area"),
                "segmentation": obj.get("segmentation"),
                "source": "coco2017",
            }
            for obj in raw_record.get("objects", [])
        ]
        return ManifestRecord(
            id=str(raw_record.get("id", f"coco_{split}_{raw_record.get('image_id')}") ),
            image_id=raw_record.get("image_id"),
            source_dataset="coco2017",
            source="coco2017",
            split=split,
            image=image_path,
            image_path=image_path,
            width=int(raw_record.get("width", 0)),
            height=int(raw_record.get("height", 0)),
            captions=list(raw_record.get("captions", [])),
            objects=object_list,
            category_names=list(raw_record.get("category_names", [])),
        )
