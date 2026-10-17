"""Generate summary statistics for the RAV COCO manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rav.intelligence.vlm.data.coco import create_coco_statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate statistics for COCO-based RAV manifests.")
    parser.add_argument("--root", type=Path, required=True, help="Path to the COCO dataset root.")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional manifest JSONL path. Defaults to the train manifest.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.manifest is None:
        target_manifest = Path("data/train/manifest.jsonl")
    else:
        target_manifest = args.manifest
    statistics = create_coco_statistics(args.root, manifest_path=target_manifest)
    output_path = Path("data/processed/coco_statistics.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(statistics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(statistics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
