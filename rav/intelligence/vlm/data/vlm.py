"""Lazy COCO-manifest adapter for image-to-caption VLM training."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from rav.intelligence.vlm.data.schema import ManifestRecord
from rav.intelligence.vlm.prompts import CAPTION_PROMPT, caption_messages


class ManifestCaptionDataset:
    """Index JSONL offsets only; records and image pixels are read on demand."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        max_samples: int | None = None,
        seed: int | None = None,
        prompt: str = CAPTION_PROMPT,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.prompt = prompt
        self._offsets: list[int] = []
        with self.manifest_path.open("rb") as handle:
            while True:
                offset = handle.tell()
                if not handle.readline():
                    break
                self._offsets.append(offset)
        if seed is not None:
            # Shuffle once before truncating: every smaller subset is a prefix of
            # every larger subset made with this manifest and seed.
            random.Random(seed).shuffle(self._offsets)
        if max_samples is not None:
            if max_samples < 1:
                raise ValueError("max_samples must be a positive integer when provided.")
            if max_samples > len(self._offsets):
                raise ValueError(
                    f"Requested {max_samples} samples but {self.manifest_path} contains "
                    f"only {len(self._offsets)} records."
                )
            self._offsets = self._offsets[:max_samples]

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        with self.manifest_path.open("rb") as handle:
            handle.seek(self._offsets[index])
            record = ManifestRecord.model_validate(json.loads(handle.readline()))
        if not record.captions:
            raise ValueError(f"Manifest record {record.id} has no COCO caption.")
        image_path = Path(record.image)
        if not image_path.is_absolute():
            image_path = (Path.cwd() / image_path).resolve()
        # Preserve all annotations as metadata; no annotation is invented.
        return {
            "id": record.id,
            "image_path": str(image_path),
            "caption": record.captions[index % len(record.captions)],
            "messages": caption_messages(
                record.captions[index % len(record.captions)], self.prompt
            ),
            "source": record.source,
            "metadata": {
                "image_id": record.image_id,
                "objects": [obj.model_dump(mode="json") for obj in record.objects],
                "category_names": record.category_names,
                "width": record.width,
                "height": record.height,
            },
        }
