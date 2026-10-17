"""Processor helpers kept independent from a particular VLM backend."""

from __future__ import annotations

from typing import Any

from rav.intelligence.vlm.prompts import GROUNDING_INSTRUCTION, caption_messages


def training_text(processor: Any, caption: str, prompt: str) -> str:
    """Render a full image-text conversation using the processor's template."""
    return processor.apply_chat_template(
        caption_messages(caption, prompt), tokenize=False, add_generation_prompt=False
    )


def inference_messages(prompt: str) -> list[dict[str, object]]:
    return caption_messages(None, f"{prompt.strip()}\n{GROUNDING_INSTRUCTION}")
