"""Typed parsing of configs/vlm.yaml without machine-specific paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class LoRASettings:
    enabled: bool = True
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")


@dataclass(frozen=True)
class VLMSettings:
    model: dict[str, Any]
    training: dict[str, Any]
    lora: LoRASettings
    data: dict[str, Any]
    hardware: dict[str, Any]


def load_vlm_config(path: str | Path) -> VLMSettings:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    lora = payload.get("lora", {})
    return VLMSettings(
        model=payload.get("model", {}),
        training=payload.get("training", {}),
        data=payload.get("data", {}),
        hardware=payload.get("hardware", {}),
        lora=LoRASettings(
            enabled=lora.get("enabled", True),
            rank=lora.get("rank", 8),
            alpha=lora.get("alpha", 16),
            dropout=lora.get("dropout", 0.05),
            target_modules=tuple(lora.get("target_modules", LoRASettings.target_modules)),
        ),
    )
