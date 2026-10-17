from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from rav.intelligence.vlm.data.vlm import ManifestCaptionDataset
from rav.intelligence.vlm.processor import inference_messages
from rav.intelligence.vlm.prompts import CAPTION_PROMPT, GROUNDING_INSTRUCTION
from rav.intelligence.vlm.registry import get_model_spec
from rav.intelligence.vlm.training.checkpoint import checkpoint_path
from rav.intelligence.vlm.training.config import load_vlm_config


def _manifest(path: Path, image: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "id": "coco_val_1",
                "image_id": 1,
                "source": "coco2017",
                "source_dataset": "coco2017",
                "split": "val",
                "image": str(image),
                "width": 8,
                "height": 8,
                "captions": ["A real COCO caption."],
                "objects": [{"category": "thing", "bbox": [0, 0, 1, 1]}],
                "category_names": ["thing"],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_manifest_caption_adapter_is_lazy_and_preserves_metadata(tmp_path: Path) -> None:
    image = tmp_path / "sample.jpg"
    Image.new("RGB", (8, 8)).save(image)
    manifest = tmp_path / "manifest.jsonl"
    _manifest(manifest, image)
    sample = ManifestCaptionDataset(manifest)[0]
    assert sample["caption"] == "A real COCO caption."
    assert sample["messages"][0]["content"][1]["text"] == CAPTION_PROMPT
    assert sample["metadata"]["objects"][0]["category"] == "thing"


def test_manifest_caption_subsets_are_seeded_and_cumulative(tmp_path: Path) -> None:
    image = tmp_path / "sample.jpg"
    Image.new("RGB", (8, 8)).save(image)
    manifest = tmp_path / "manifest.jsonl"
    records = []
    for index in range(8):
        records.append(
            {
                "id": f"coco_train_{index}",
                "image_id": index,
                "source": "coco2017",
                "source_dataset": "coco2017",
                "split": "train",
                "image": str(image),
                "width": 8,
                "height": 8,
                "captions": [f"caption {index}"],
                "objects": [],
                "category_names": [],
            }
        )
    manifest.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )
    first_stage = ManifestCaptionDataset(manifest, max_samples=3, seed=42)
    second_stage = ManifestCaptionDataset(manifest, max_samples=6, seed=42)
    repeated_stage = ManifestCaptionDataset(manifest, max_samples=3, seed=42)
    assert [first_stage[index]["id"] for index in range(3)] == [
        second_stage[index]["id"] for index in range(3)
    ]
    assert [first_stage[index]["id"] for index in range(3)] == [
        repeated_stage[index]["id"] for index in range(3)
    ]


def test_manifest_caption_rejects_unavailable_requested_size(tmp_path: Path) -> None:
    image = tmp_path / "sample.jpg"
    Image.new("RGB", (8, 8)).save(image)
    manifest = tmp_path / "manifest.jsonl"
    _manifest(manifest, image)
    try:
        ManifestCaptionDataset(manifest, max_samples=2, seed=42)
    except ValueError as error:
        assert "Requested 2 samples" in str(error)
    else:  # pragma: no cover
        raise AssertionError("Expected unavailable sample count to be rejected")


def test_config_and_model_registry() -> None:
    config = load_vlm_config(Path("configs/vlm.yaml"))
    assert config.lora.enabled
    assert config.lora.target_modules
    assert get_model_spec("smolvlm-256m").parameters == "256M"


def test_checkpoint_layout(tmp_path: Path) -> None:
    destination = checkpoint_path("smoke", root=tmp_path)
    assert destination == tmp_path / "adapters" / "smoke"
    assert destination.is_dir()


def test_inference_prompt_requests_grounded_uncertainty() -> None:
    messages = inference_messages("Where is the laptop?")
    prompt = messages[0]["content"][1]["text"]

    assert "Where is the laptop?" in prompt
    assert GROUNDING_INSTRUCTION in prompt
    assert "unsure" not in prompt.lower()
