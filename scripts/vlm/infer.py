"""Run configured local VLM inference on any PIL-compatible image file."""

from __future__ import annotations

import argparse
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", default="Describe this image accurately and concisely.")
    parser.add_argument("--model", default="HuggingFaceTB/SmolVLM-256M-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--quantization", choices=["none", "4bit"], default="none")
    parser.add_argument("--adapter", default=None, help="Optional PEFT adapter checkpoint path.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    import torch
    from PIL import Image
    from rav.intelligence.vlm.model import HuggingFaceVLM

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA was requested but is unavailable. "
            "Use --device cpu deliberately, or install CUDA PyTorch."
        )
    device = args.device if args.device != "cuda" else "auto"
    vlm = HuggingFaceVLM(args.model, device=device, quantization=args.quantization).load()
    if args.adapter:
        vlm.load_checkpoint(args.adapter)
    with Image.open(args.image) as image:
        started = time.perf_counter()
        answer = vlm.generate(image.convert("RGB"), args.prompt, max_new_tokens=args.max_new_tokens)
    elapsed = time.perf_counter() - started
    print(answer)
    print(f"inference_seconds={elapsed:.3f}")
    print(f"device={next(vlm.model.parameters()).device}")
    if torch.cuda.is_available():
        print(f"peak_gpu_memory_mib={torch.cuda.max_memory_allocated() / 1024**2:.1f}")


if __name__ == "__main__":
    main()
