"""Generate reproducible qualitative examples from a trained RAV adapter."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/val/manifest.jsonl")
    parser.add_argument("--model", default="HuggingFaceTB/SmolVLM-256M-Instruct")
    parser.add_argument("--adapter", default="models/adapters/rav_coco_1k_v1/best")
    parser.add_argument("--output-dir", default="data/processed/vlm_baseline/rav_coco_1k_v1")
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--quantization", choices=["none", "4bit"], default="4bit")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.max_samples < 1:
        parser.error("--max-samples must be positive.")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive.")

    import torch
    from PIL import Image
    from rav.intelligence.vlm.data.vlm import ManifestCaptionDataset
    from rav.intelligence.vlm.model import HuggingFaceVLM
    from rav.intelligence.vlm.prompts import CAPTION_PROMPT
    from rav.intelligence.vlm.training.trainer import trainable_parameter_count

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA requested but unavailable; pass --device cpu explicitly to override."
        )
    adapter = Path(args.adapter)
    if not adapter.is_dir():
        raise SystemExit(f"Trained adapter does not exist: {adapter}")
    vlm = HuggingFaceVLM(
        args.model,
        device="auto" if args.device == "cuda" else args.device,
        quantization=args.quantization,
    ).load()
    vlm.load_checkpoint(str(adapter))
    dataset = ManifestCaptionDataset(args.manifest, max_samples=args.max_samples, seed=args.seed)
    outputs: list[dict[str, str]] = []
    started = time.perf_counter()
    for sample in dataset:
        try:
            with Image.open(sample["image_path"]) as image:
                response = vlm.generate(
                    image.convert("RGB"),
                    CAPTION_PROMPT,
                    max_new_tokens=args.max_new_tokens,
                )
        except OSError as exc:
            raise RuntimeError(
                f"Could not open validation image {sample['image_path']}: {exc}"
            ) from exc
        outputs.append(
            {
                "id": sample["id"],
                "image_path": sample["image_path"],
                "prompt": CAPTION_PROMPT,
                "response": response,
                "reference_caption": sample["caption"],
            }
        )
    elapsed = time.perf_counter() - started
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "timestamp": datetime.now(UTC).isoformat(),
        "model": args.model,
        "adapter": str(adapter),
        "seed": args.seed,
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
        "scope_note": (
            "Small qualitative proof of concept; these examples do not establish "
            "general intelligence or broad visual competence."
        ),
    }
    evaluation_path = output_dir / "evaluation.json"
    evaluation_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    examples = [
        "# RAV 1K COCO qualitative examples",
        "",
        "Small proof-of-concept samples only; not evidence of general intelligence.",
        "",
    ]
    for index, output in enumerate(outputs, start=1):
        examples.extend(
            [
                f"## Example {index} — {output['id']}",
                "",
                f"- Image: `{output['image_path']}`",
                f"- Prompt: {output['prompt']}",
                f"- Model response: {output['response']}",
                "",
            ]
        )
    examples_path = output_dir / "examples.md"
    examples_path.write_text("\n".join(examples), encoding="utf-8")
    print(f"evaluation={evaluation_path.resolve()}")
    print(f"examples={examples_path.resolve()}")
    print(f"samples={len(outputs)}")


if __name__ == "__main__":
    main()
