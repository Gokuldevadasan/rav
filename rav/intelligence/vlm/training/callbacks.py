"""Lightweight structured experiment recording without an external service."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def write_experiment_record(
    payload: dict[str, Any],
    *,
    root: str | Path = "experiments/results",
    filename: str | None = None,
) -> Path:
    """Persist configuration, hardware, metrics, and checkpoint provenance as JSON."""
    destination_root = Path(root)
    destination_root.mkdir(parents=True, exist_ok=True)
    record = dict(payload)
    record.setdefault("timestamp", datetime.now(UTC).isoformat())
    try:
        record.setdefault(
            "git_commit",
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip(),
        )
    except (OSError, subprocess.CalledProcessError):
        record.setdefault("git_commit", None)
    destination = destination_root / (filename or f"{record['timestamp'].replace(':', '-')}.json")
    destination.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return destination
