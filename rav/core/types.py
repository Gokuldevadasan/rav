"""Typed, transport-safe representations shared by RAV subsystems."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SourceType(StrEnum):
    """Supported origins of a camera frame."""

    PHONE = "phone"
    WEBCAM = "webcam"
    SIMULATED = "simulated"
    ROS = "ros"
    VIDEO_FILE = "video_file"
    IMAGE_FILE = "image_file"


class BoundingBox(BaseModel):
    """A pixel-aligned object rectangle in an image."""

    model_config = ConfigDict(frozen=True)

    x: float = Field(ge=0)
    y: float = Field(ge=0)
    width: float = Field(gt=0)
    height: float = Field(gt=0)


class ObjectObservation(BaseModel):
    """One grounded object observation from a perception component."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    class_name: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    bbox: BoundingBox | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    graspable: bool | None = None
    source: str = Field(min_length=1)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CameraFrame(BaseModel):
    """A camera sample; pixel payload ownership stays with the camera adapter."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    id: str = Field(min_length=1)
    source: SourceType
    captured_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    payload: bytes | None = None


class Scene(BaseModel):
    """Current frame-level scene interpretation, before temporal fusion."""

    frame_id: str = Field(min_length=1)
    objects: list[ObjectObservation] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SystemHealth(BaseModel):
    """Non-sensitive health information for a future API endpoint."""

    status: str = "ok"
    system_name: str
    environment: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str) -> str:
        if value not in {"ok", "degraded", "unavailable"}:
            raise ValueError("status must be ok, degraded, or unavailable")
        return value
