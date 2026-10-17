"""Lazy image-to-chat collator with assistant-only loss labels."""

from __future__ import annotations

from typing import Any


class VLMDataCollator:
    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "VLM collator requires Pillow. Install: pip install -e '.[vlm]'"
            ) from exc
        images = []
        texts = []
        prompt_lengths = []
        for feature in features:
            with Image.open(feature["image_path"]) as image:
                images.append(image.convert("RGB").copy())
            full = self.processor.apply_chat_template(
                feature["messages"], tokenize=False, add_generation_prompt=False
            )
            prompt = self.processor.apply_chat_template(
                feature["messages"][:1], tokenize=False, add_generation_prompt=True
            )
            texts.append(full)
            # Token counts are calculated after image-token expansion by the processor.
            prompt_lengths.append(
                len(self.processor.tokenizer(prompt, add_special_tokens=False)["input_ids"])
            )
        batch = self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        for row, prompt_length in enumerate(prompt_lengths):
            labels[row, : min(prompt_length, labels.shape[1])] = -100
        batch["labels"] = labels
        return batch
