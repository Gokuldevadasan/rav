"""RAV manifest and COCO data schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SplitName = Literal["train", "val", "test"]


class RAVBoundingBox(BaseModel):
    """A normalized image rectangle in pixels."""

    model_config = ConfigDict(frozen=True)

    x: float = Field(ge=0)
    y: float = Field(ge=0)
    width: float = Field(gt=0)
    height: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_box(self) -> RAVBoundingBox:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("bounding box width and height must be > 0")
        return self


class ObjectRecord(BaseModel):
    """A single object annotation from a COCO-style dataset."""

    id: str | None = None
    category: str = Field(min_length=1)
    bbox: list[float] | None = None
    area: float | None = None
    segmentation: Any | None = None
    source: str = "coco2017"

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, value: list[float] | None) -> list[float] | None:
        if value is None:
            return value
        if len(value) != 4:
            raise ValueError("bbox must contain exactly four values: x, y, width, height")
        x, y, width, height = value
        if width <= 0 or height <= 0:
            raise ValueError("bbox width and height must be > 0")
        return value


class RAVImageMetadata(BaseModel):
    """Image metadata attached to a manifest sample."""

    model_config = ConfigDict(frozen=True)

    image_id: int | str
    source: str = "coco2017"
    split: SplitName
    image: str = Field(min_length=1)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class ManifestRecord(BaseModel):
    """Unified data record used in a RAV JSONL manifest."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(min_length=1)
    image_id: int | str
    source_dataset: str = "coco2017"
    source: str = "coco2017"
    split: SplitName
    image: str = Field(min_length=1)
    image_path: str | None = None
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    captions: list[str] = Field(default_factory=list)
    objects: list[ObjectRecord] = Field(default_factory=list)
    category_names: list[str] = Field(default_factory=list)

    @field_validator("captions")
    @classmethod
    def validate_captions(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        for caption in value:
            if not isinstance(caption, str):
                raise ValueError("captions must contain strings")
            cleaned_caption = caption.strip()
            if cleaned_caption:
                cleaned.append(cleaned_caption)
        return cleaned

    @field_validator("category_names")
    @classmethod
    def validate_category_names(cls, value: list[str]) -> list[str]:
        return [category.strip() for category in value if category and category.strip()]

    @property
    def object_count(self) -> int:
        return len(self.objects)

    @property
    def caption_count(self) -> int:
        return len(self.captions)


def validate_bounding_box(box: list[float] | tuple[float, ...], width: int, height: int) -> bool:
    """Validate a COCO-style bbox against image bounds."""

    if len(box) != 4:
        return False
    x, y, w, h = [float(value) for value in box]
    if w <= 0 or h <= 0:
        return False
    x2 = x + w
    y2 = y + h
    return x >= 0 and y >= 0 and x2 <= width and y2 <= height


def box_to_rav_bbox(box: list[float] | tuple[float, ...]) -> RAVBoundingBox:
    """Convert a COCO bbox list into a validated RAV bbox."""

    if len(box) != 4:
        raise ValueError("bbox must include x, y, width, height")
    x, y, width, height = [float(value) for value in box]
    return RAVBoundingBox(x=x, y=y, width=width, height=height)
