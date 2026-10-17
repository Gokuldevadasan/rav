"""Train an isolated, reproducible COCO captioning experiment."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_EVAL_SAMPLES = 1_000


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vlm.yaml")
    parser.add_argument("--max-samples", type=int, required=True)
    parser.add_argument("--checkpoint-name", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--eval-samples",
        "--validation-samples",
        dest="eval_samples",
        type=int,
        default=DEFAULT_EVAL_SAMPLES,
    )
    parser.add_argument(
        "--training-mode", choices=("independent", "progressive"), default="independent"
    )
    parser.add_argument("--resume-from", default=None)
    parser.add_argument(
        "--device", default=None, help="Explicit override, e.g. cpu for diagnostics."
    )
    parser.add_argument("--quantization", choices=("none", "4bit"), default=None)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Require a complete cached base model and disable Hugging Face network access.",
    )
    args = parser.parse_args()
    if args.max_samples < 1:
        parser.error("--max-samples must be positive.")
    if args.eval_samples < 1:
        parser.error("--eval-samples must be positive.")
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive.")
    if Path(args.checkpoint_name).name != args.checkpoint_name:
        parser.error("--checkpoint-name must be a single directory name.")
    if args.training_mode == "progressive" and not args.resume_from:
        parser.error("--training-mode progressive requires --resume-from.")
    if args.training_mode == "independent" and args.resume_from:
        parser.error("--resume-from requires --training-mode progressive.")
    return args


def _gpu_information(torch: Any, device: str) -> dict[str, Any] | None:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return None
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "name": properties.name,
        "total_memory_bytes": properties.total_memory,
        "cuda_version": torch.version.cuda,
    }


def main() -> None:
    args = _arguments()
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch
    from rav.intelligence.vlm.data.vlm import ManifestCaptionDataset
    from rav.intelligence.vlm.model import HuggingFaceVLM
    from rav.intelligence.vlm.training.callbacks import write_experiment_record
    from rav.intelligence.vlm.training.checkpoint import save_checkpoint
    from rav.intelligence.vlm.training.collator import VLMDataCollator
    from rav.intelligence.vlm.training.config import load_vlm_config
    from rav.intelligence.vlm.training.trainer import (
        configure_lora,
        train_with_validation,
        trainable_parameter_count,
    )
    from torch.utils.data import DataLoader

    cfg = load_vlm_config(args.config)
    device = args.device or cfg.hardware.get("device", "cuda")
    quantization = args.quantization or cfg.hardware.get("quantization", "none")
    epochs = args.epochs if args.epochs is not None else cfg.training["epochs"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is required by configs/vlm.yaml but unavailable; refusing a silent CPU fallback."
        )
    experiment_root = (
        Path("models") / ("adapters" if cfg.lora.enabled else "vlm") / args.checkpoint_name
    )
    results_path = Path("experiments/results") / f"{args.checkpoint_name}.json"
    log_path = Path("experiments/logs") / f"{args.checkpoint_name}.jsonl"
    if experiment_root.exists() or results_path.exists() or log_path.exists():
        raise SystemExit(
            f"Experiment '{args.checkpoint_name}' already exists. "
            "Existing experiments are never overwritten."
        )
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(
        f"[1/5] Preparing seeded COCO subsets: "
        f"{args.max_samples:,} train / {args.eval_samples:,} validation",
        flush=True,
    )
    train_dataset = ManifestCaptionDataset(
        cfg.data["train_manifest"], max_samples=args.max_samples, seed=args.seed
    )
    validation_dataset = ManifestCaptionDataset(
        cfg.data["val_manifest"], max_samples=args.eval_samples, seed=args.seed
    )
    if len(train_dataset) != args.max_samples or len(validation_dataset) != args.eval_samples:
        raise RuntimeError("Dataset selection did not produce the requested fixed subset sizes.")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_event(event: str, **fields: Any) -> None:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps({"timestamp": datetime.now(UTC).isoformat(), "event": event, **fields})
                + "\n"
            )

    log_event(
        "training_started",
        experiment_name=args.checkpoint_name,
        model_name=cfg.model["name"],
        training_mode=args.training_mode,
        train_samples=len(train_dataset),
        eval_samples=len(validation_dataset),
        epochs=epochs,
        seed=args.seed,
        quantization=quantization,
        offline=args.offline,
    )
    model_reference = cfg.model["name"]
    if args.offline:
        print("[2/5] Resolving the cached pretrained model (offline mode)", flush=True)
        try:
            from huggingface_hub import snapshot_download

            model_reference = snapshot_download(
                repo_id=cfg.model["name"],
                revision=cfg.model.get("revision"),
                local_files_only=True,
            )
        except Exception as exc:
            raise SystemExit(
                f"Offline model cache is incomplete for {cfg.model['name']}. "
                "Run scripts/vlm/prefetch_model.py while internet access is available."
            ) from exc
        required_files = ("config.json", "model.safetensors", "preprocessor_config.json")
        missing_files = [
            name for name in required_files if not (Path(model_reference) / name).is_file()
        ]
        if missing_files:
            raise SystemExit(
                f"Cached model snapshot is missing required files: {', '.join(missing_files)}"
            )
        print(f"      Cached snapshot: {model_reference}", flush=True)

    print(
        f"[3/5] Loading pretrained model and processor "
        f"({quantization} quantization; CUDA device={device})",
        flush=True,
    )
    vlm = HuggingFaceVLM(
        model_reference,
        device="auto" if device.startswith("cuda") else device,
        revision=cfg.model.get("revision"),
        trust_remote_code=cfg.model.get("trust_remote_code", False),
        quantization=quantization,
    ).load()
    model = vlm.model
    if args.offline:
        model.config._name_or_path = model_reference
    if cfg.hardware.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    if cfg.lora.enabled:
        model = configure_lora(model, cfg.lora, qlora=quantization == "4bit")
        vlm.model = model
    if args.training_mode == "progressive":
        resume_from = Path(args.resume_from)
        if not resume_from.is_dir():
            raise SystemExit(f"Progressive checkpoint does not exist: {resume_from}")
        vlm.load_checkpoint(str(resume_from))
        model = vlm.model

    trainable, total = trainable_parameter_count(model)
    collator = VLMDataCollator(vlm.processor)
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.training["batch_size"], shuffle=False, collate_fn=collator
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=cfg.training["batch_size"],
        shuffle=False,
        collate_fn=collator,
    )
    model_device = next(model.parameters()).device
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    best_validation_loss = float("inf")
    best_checkpoint: Path | None = None
    print(
        f"[4/5] Ready: {epochs} epochs, {len(train_loader):,} batches/epoch, "
        f"{trainable:,} trainable parameters",
        flush=True,
    )

    def save_step(step: int) -> None:
        save_checkpoint(
            model,
            vlm.processor,
            f"{args.checkpoint_name}/checkpoint-{step}",
            adapter=cfg.lora.enabled,
        )

    def validate(epoch: int, validation_loss: float) -> None:
        nonlocal best_validation_loss, best_checkpoint
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_checkpoint = save_checkpoint(
                model, vlm.processor, f"{args.checkpoint_name}/best", adapter=cfg.lora.enabled
            )
        print(f"epoch={epoch} validation_loss={validation_loss:.6f}", flush=True)
        log_event("epoch_completed", epoch=epoch, validation_loss=validation_loss)

    started = time.perf_counter()
    print("[5/5] Training (live epoch and batch progress follows)", flush=True)
    final_training_loss, final_validation_loss, global_steps = train_with_validation(
        model,
        train_loader,
        validation_loader,
        device=model_device,
        learning_rate=cfg.training["learning_rate"],
        gradient_accumulation=cfg.training["gradient_accumulation"],
        epochs=epochs,
        save_steps=cfg.training["save_steps"],
        on_checkpoint=save_step,
        on_validation=validate,
    )
    training_seconds = time.perf_counter() - started
    latest_checkpoint = save_checkpoint(
        model, vlm.processor, f"{args.checkpoint_name}/latest", adapter=cfg.lora.enabled
    )
    peak_memory = torch.cuda.max_memory_allocated() if device.startswith("cuda") else None
    record = {
        "experiment_name": args.checkpoint_name,
        "training_mode": args.training_mode,
        "resume_from": args.resume_from,
        "dataset": {
            "train_manifest": str(cfg.data["train_manifest"]),
            "validation_manifest": str(cfg.data["val_manifest"]),
            "train_samples": len(train_dataset),
            "validation_samples": len(validation_dataset),
            "eval_samples": len(validation_dataset),
            "sampling": "seeded_permutation_prefix",
        },
        "random_seed": args.seed,
        "epochs": epochs,
        "global_steps": global_steps,
        "training_loss": final_training_loss,
        "validation_loss": final_validation_loss,
        "best_validation_loss": best_validation_loss,
        "final_validation_loss": final_validation_loss,
        "best_checkpoint": str(best_checkpoint.resolve()) if best_checkpoint else None,
        "final_checkpoint": str(latest_checkpoint.resolve()),
        "training_time_seconds": training_seconds,
        "samples_per_second": (len(train_dataset) * epochs) / training_seconds,
        "peak_gpu_memory_bytes": peak_memory,
        "gpu": _gpu_information(torch, device),
        "model_name": cfg.model["name"],
        "model_snapshot": str(model_reference),
        "offline": args.offline,
        "initialization": "base_model",
        "quantization": quantization,
        "lora": {
            "enabled": cfg.lora.enabled,
            "rank": cfg.lora.rank,
            "alpha": cfg.lora.alpha,
            "dropout": cfg.lora.dropout,
            "target_modules": list(cfg.lora.target_modules),
            "trainable_parameters": trainable,
            "total_parameters": total,
        },
    }
    record_path = write_experiment_record(record, filename=f"{args.checkpoint_name}.json")
    log_event(
        "training_completed",
        training_loss=final_training_loss,
        validation_loss=final_validation_loss,
        best_checkpoint=str(best_checkpoint.resolve()) if best_checkpoint else None,
        result=str(record_path.resolve()),
    )
    print(f"training_loss={final_training_loss:.6f}")
    print(f"final_validation_loss={final_validation_loss:.6f}")
    print(f"checkpoint={latest_checkpoint.resolve()}")
    print(f"best_checkpoint={best_checkpoint.resolve() if best_checkpoint else 'unavailable'}")
    print(f"experiment_record={record_path.resolve()}")
    print(f"training_log={log_path.resolve()}")


if __name__ == "__main__":
    main()
