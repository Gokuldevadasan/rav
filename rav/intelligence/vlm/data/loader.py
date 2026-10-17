"""Lazy dataset loader for RAV JSONL manifest datasets."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset

from rav.intelligence.vlm.data.manifest import load_manifest


class RAVVisionDataset(Dataset):
    """A PyTorch-compatible dataset backed by a JSONL manifest."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        image_root: str | Path | None = None,
        transforms: Callable[[Image.Image], Any] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.image_root = (
            Path(image_root)
            if image_root is not None
            else self.manifest_path.parent.parent
        )
        self.transforms = transforms
        self.records = load_manifest(self.manifest_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image_value = Path(record.image)
        if image_value.is_absolute():
            image_path = image_value
        elif image_value.parts and image_value.parts[0] in {"data", "coco2017"}:
            image_path = (Path.cwd() / image_value).resolve()
        else:
            image_path = (self.image_root / image_value).resolve()
        image = Image.open(image_path).convert("RGB")
        sample: dict[str, Any] = {
            "id": record.id,
            "image_id": record.image_id,
            "source": record.source,
            "split": record.split,
            "image_path": str(image_path),
            "width": record.width,
            "height": record.height,
            "captions": record.captions,
            "objects": [obj.model_dump(mode="json") for obj in record.objects],
            "metadata": {
                "image": record.image,
                "source_dataset": record.source_dataset,
                "category_names": record.category_names,
            },
            "image": image,
        }
        if self.transforms is not None:
            sample["image"] = self.transforms(sample["image"])
        return sample
