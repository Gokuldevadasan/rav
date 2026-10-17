"""Central VLM backend registry and baseline specifications."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from rav.intelligence.vlm.base import BaseVLM


@dataclass(frozen=True)
class VLMModelSpec:
    key: str
    model_id: str
    architecture: str
    parameters: str
    processor: str
    recommended_quantization: str
    training_method: str
    notes: str


_SPECS: dict[str, VLMModelSpec] = {
    "smolvlm-256m": VLMModelSpec(
        key="smolvlm-256m",
        model_id="HuggingFaceTB/SmolVLM-256M-Instruct",
        architecture="Idefics3 / SmolVLM",
        parameters="256M",
        processor="transformers.AutoProcessor",
        recommended_quantization="FP16 or 4-bit NF4",
        training_method="LoRA / QLoRA",
        notes=(
            "Hardware-conservative functional baseline for the current 4 GiB RTX 3050. "
            "Its limited capacity makes it a pipeline baseline, not a quality claim."
        ),
    ),
    "qwen2.5-vl-3b": VLMModelSpec(
        key="qwen2.5-vl-3b",
        model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        architecture="Qwen2.5-VL",
        parameters="3B",
        processor="transformers.AutoProcessor",
        recommended_quantization="4-bit NF4 (QLoRA for training)",
        training_method="LoRA / QLoRA",
        notes=(
            "Scale-up candidate for GPUs with at least roughly 6–8 GiB free VRAM when quantized; "
            "not suitable for the current 4 GiB GPU."
        ),
    ),
}
_BACKENDS: dict[str, Callable[..., BaseVLM]] = {}


def register_backend(name: str, factory: Callable[..., BaseVLM]) -> None:
    if name in _BACKENDS:
        raise ValueError(f"VLM backend already registered: {name}")
    _BACKENDS[name] = factory


def create_vlm(name: str = "huggingface", **kwargs: object) -> BaseVLM:
    try:
        return _BACKENDS[name](**kwargs)
    except KeyError as exc:
        raise ValueError(f"Unknown VLM backend {name!r}; available: {sorted(_BACKENDS)}") from exc


def get_model_spec(key: str) -> VLMModelSpec:
    try:
        return _SPECS[key]
    except KeyError as exc:
        raise ValueError(f"Unknown VLM model {key!r}; available: {sorted(_SPECS)}") from exc
