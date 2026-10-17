# RAV dataset documentation

This repository uses the original COCO 2017 dataset as the first source-of-truth dataset for the RAV foundation pipeline.

## Source

- Dataset: COCO 2017
- Source directory: `data/coco2017/`
- Supported annotation types: captions, instances, person keypoints
- Split directories: `train2017/`, `val2017/`, `test2017/`

## Expected directory structure

```text
data/
  coco2017/
    annotations/
    train2017/
    val2017/
    test2017/
  train/
    manifest.jsonl
  val/
    manifest.jsonl
  test/
    manifest.jsonl
  processed/
    coco_statistics.json
    inspection/
```

## Preprocessing workflow

1. Run `python scripts/data/inspect_coco.py --root data/coco2017`
2. Run `python scripts/data/build_coco_manifests.py --root data/coco2017 --config configs/data.yaml`
3. Run `python scripts/data/validate_rav_dataset.py --manifest data/train/manifest.jsonl`
4. Run `python scripts/data/coco_statistics.py --root data/coco2017`
5. Run `python scripts/data/sample_coco.py --manifest data/train/manifest.jsonl --num-samples 20`

## Manifest schema

The JSONL manifests record image-level metadata, source captions, and object annotations without storing binary image data inside the file. The schema is intentionally extensible for future grounding, reasoning, action, verification, and uncertainty fields.

## Reproducibility

The original COCO data is not modified. Manifests are generated from the source files and can be rebuilt at any time using the commands above. Generated files are kept under `data/` and should not be committed when they are large or machine-generated unless explicitly required.
