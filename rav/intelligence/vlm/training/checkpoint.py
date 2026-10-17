"""Explicit adapter and full-model checkpoint layout."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def checkpoint_path(name: str, *, adapter: bool = True, root: str | Path = "models") -> Path:
    directory = Path(root) / ("adapters" if adapter else "vlm") / name
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_checkpoint(
    model: Any,
    processor: Any,
    name: str,
    *,
    adapter: bool = True,
    root: str | Path = "models",
) -> Path:
    destination = checkpoint_path(name, adapter=adapter, root=root)
    model.save_pretrained(destination)
    processor.save_pretrained(destination)
    return destination
