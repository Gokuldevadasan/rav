"""Sample a few manifest records for human inspection."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from rav.intelligence.vlm.data.manifest import load_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a random sample from a RAV JSONL manifest.")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to the manifest JSONL file")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of records to inspect")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_manifest(args.manifest)
    if not records:
        raise ValueError(f"No records found in manifest: {args.manifest}")
    random.Random(args.seed).shuffle(records)
    selected = records[: min(args.num_samples, len(records))]

    output_dir = Path("data/processed/inspection")
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(selected, start=1):
        print(f"Sample {index}: {record.id}")
        print(f"  image: {record.image}")
        print(f"  captions: {record.captions[:3]}")
        print(f"  labels: {[obj.category for obj in record.objects[:5]]}")
        print(f"  bbox_count: {len(record.objects)}")
        print()

    html_path = output_dir / f"{args.manifest.stem}_sample.html"
    html_path.write_text(
        "<html><body><h1>RAV sample inspection</h1>" + "".join(
            f"<div><h2>{record.id}</h2><p>{record.image}</p><p>Captions: {'; '.join(record.captions[:3])}</p><p>Labels: {[obj.category for obj in record.objects[:5]]}</p></div>"
            for record in selected
        ) + "</body></html>",
        encoding="utf-8",
    )
    print(f"HTML preview written to {html_path}")


if __name__ == "__main__":
    main()
