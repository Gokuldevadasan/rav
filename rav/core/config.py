"""Configuration loading with explicit, validated schemas."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from rav.core.types import SourceType


class SystemConfig(BaseModel):
    name: str = Field(min_length=1)
    environment: str = Field(min_length=1)
    log_level: str = "INFO"


class APIConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)


class CameraConfig(BaseModel):
    source: SourceType = SourceType.PHONE
    max_fps: int = Field(default=10, ge=1, le=120)


class RAVConfig(BaseModel):
    system: SystemConfig
    api: APIConfig = Field(default_factory=APIConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)


def load_config(path: Path) -> RAVConfig:
    """Load and validate a YAML RAV configuration file."""
    with path.open("r", encoding="utf-8") as stream:
        raw_config = yaml.safe_load(stream)
    if not isinstance(raw_config, dict):
        raise ValueError(f"Configuration at {path} must contain a YAML mapping")
    return RAVConfig.model_validate(raw_config)
