"""Build JSONL manifests for COCO train/val/test splits without copying image bytes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from rav.intelligence.vlm.data.coco import build_manifest_for_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RAV JSONL manifests from COCO annotations.")
    parser.add_argument("--root", type=Path, required=True, help="COCO root directory (e.g. data/coco2017)")
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"), help="Dataset configuration YAML")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    path_map = config.get("paths", {})
    output_paths = {
        "train": Path(path_map.get("train_manifest", "data/train/manifest.jsonl")),
        "val": Path(path_map.get("val_manifest", "data/val/manifest.jsonl")),
        "test": Path(path_map.get("test_manifest", "data/test/manifest.jsonl")),
    }

    for split in ("train", "val", "test"):
        records = build_manifest_for_split(args.root, split=split, output_path=output_paths[split])
        print(f"{split}: {len(records)} records written to {output_paths[split]}")


if __name__ == "__main__":
    sys.exit(main())
