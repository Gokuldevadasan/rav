"""Download and verify the complete pretrained VLM snapshot before offline training."""

from __future__ import annotations

import argparse
from pathlib import Path

REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="HuggingFaceTB/SmolVLM-256M-Instruct")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    print(f"Downloading/verifying model files: {args.model}", flush=True)
    try:
        snapshot = Path(
            snapshot_download(
                repo_id=args.model,
                revision=args.revision,
                local_files_only=args.local_files_only,
            )
        )
    except Exception as exc:
        if args.local_files_only:
            raise SystemExit(
                f"No complete cached snapshot for {args.model}. "
                "Remove --local-files-only while internet access is available."
            ) from exc
        raise SystemExit(
            f"Could not download {args.model}. Check DNS/internet access, then retry. "
            f"Details: {exc}"
        ) from exc

    missing = [name for name in REQUIRED_FILES if not (snapshot / name).is_file()]
    if missing:
        raise SystemExit(f"Downloaded snapshot is incomplete; missing: {', '.join(missing)}")
    total_bytes = sum(path.stat().st_size for path in snapshot.iterdir() if path.is_file())
    print(f"Verified local snapshot: {snapshot.resolve()}")
    print(f"Files: {len(list(snapshot.iterdir()))}; size: {total_bytes / 1024**2:.1f} MiB")
    print("Next: run scripts/vlm/train.py with --offline to prevent model network requests.")


if __name__ == "__main__":
    main()
