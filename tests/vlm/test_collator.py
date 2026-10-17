from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from rav.intelligence.vlm.training.collator import VLMDataCollator


class FakeTokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        return {"input_ids": list(range(len(text.split())))}


class FakeProcessor:
    tokenizer = FakeTokenizer()

    def __init__(self) -> None:
        self.processor_kwargs: dict[str, object] = {}

    def apply_chat_template(self, messages: list[dict[str, object]], **_: object) -> str:
        return " ".join(str(item) for message in messages for item in message["content"])

    def __call__(
        self, *, text: list[str], images: list[Image.Image], **_: object
    ) -> dict[str, torch.Tensor]:
        self.processor_kwargs = _
        assert len(images) == len(text) == 1
        return {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0]]),
        }


def test_collator_constructs_padded_assistant_labels(tmp_path: Path) -> None:
    image = tmp_path / "image.jpg"
    Image.new("RGB", (8, 8)).save(image)
    feature = {
        "image_path": str(image),
        "messages": [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "describe"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "caption"}]},
        ],
    }
    batch = VLMDataCollator(FakeProcessor())([feature])
    assert batch["labels"].shape == (1, 4)
    assert batch["labels"][0, -1].item() == -100


def test_collator_does_not_truncate_expanded_image_tokens(tmp_path: Path) -> None:
    image = tmp_path / "image.jpg"
    Image.new("RGB", (8, 8)).save(image)
    feature = {
        "image_path": str(image),
        "messages": [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "describe"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "caption"}]},
        ],
    }
    processor = FakeProcessor()

    VLMDataCollator(processor)([feature])

    assert "truncation" not in processor.processor_kwargs
    assert "max_length" not in processor.processor_kwargs
