"""Inspect the local COCO 2017 dataset and summarize its annotation coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rav.intelligence.vlm.data.coco import CocoDatasetInspector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a local COCO 2017 dataset.")
    parser.add_argument("--root", type=Path, required=True, help="Path to the COCO dataset root, e.g. data/coco2017")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root
    inspector = CocoDatasetInspector(root)
    summary = inspector.inspect()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
