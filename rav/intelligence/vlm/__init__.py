"""Replaceable vision-language model contracts and local backends."""

from rav.intelligence.vlm.model import HuggingFaceVLM
from rav.intelligence.vlm.registry import create_vlm, register_backend

register_backend("huggingface", HuggingFaceVLM)

__all__ = ["HuggingFaceVLM", "create_vlm", "register_backend"]
