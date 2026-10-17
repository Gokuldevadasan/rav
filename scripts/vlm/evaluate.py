"""Evaluate bounded VLM samples and preserve qualitative outputs as JSONL."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/val/manifest.jsonl")
    parser.add_argument("--model", default="HuggingFaceTB/SmolVLM-256M-Instruct")
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    import torch
    from PIL import Image
    from rav.intelligence.vlm.data.vlm import ManifestCaptionDataset
    from rav.intelligence.vlm.model import HuggingFaceVLM
    from rav.intelligence.vlm.training.trainer import trainable_parameter_count

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA requested but unavailable; pass --device cpu explicitly to override."
        )
    vlm = HuggingFaceVLM(args.model, device="auto" if args.device == "cuda" else args.device).load()
    dataset = ManifestCaptionDataset(args.manifest, max_samples=args.max_samples)
    started = time.perf_counter()
    outputs = []
    for sample in dataset:
        with Image.open(sample["image_path"]) as image:
            response = vlm.generate(
                image.convert("RGB"), "Describe this image accurately and concisely."
            )
        outputs.append(
            {"id": sample["id"], "reference_caption": sample["caption"], "generated": response}
        )
    elapsed = time.perf_counter() - started
    output_dir = Path("data/processed/vlm_baseline")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "timestamp": datetime.now(UTC).isoformat(),
        "model": args.model,
        "samples": len(outputs),
        "seconds": elapsed,
        "samples_per_second": len(outputs) / elapsed if elapsed else 0,
        "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 1024**2
        if torch.cuda.is_available()
        else None,
        "parameters": dict(
            zip(("trainable", "total"), trainable_parameter_count(vlm.model), strict=True)
        ),
        "outputs": outputs,
    }
    destination = output_dir / "evaluation.json"
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "outputs"}, indent=2))
    print(f"outputs={destination.resolve()}")


if __name__ == "__main__":
    main()
