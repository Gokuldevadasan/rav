"""JSONL manifest helpers for RAV data records."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rav.intelligence.vlm.data.schema import ManifestRecord


def write_manifest(
    path: str | Path,
    records: Iterable[ManifestRecord | dict[str, Any]],
) -> int:
    """Write manifest records to a JSONL file and return the number of records."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = (
                record.model_dump(mode="json")
                if isinstance(record, ManifestRecord)
                else dict(record)
            )
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            written += 1
    return written


def load_manifest(path: str | Path) -> list[ManifestRecord]:
    """Load a manifest JSONL file into pydantic models."""

    manifest_path = Path(path)
    if not manifest_path.exists():
        return []

    records: list[ManifestRecord] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:  # pragma: no cover
                raise ValueError(
                    f"Malformed JSON in {manifest_path} at line {line_number}: {exc}"
                ) from exc
            records.append(ManifestRecord.model_validate(payload))
    return records
