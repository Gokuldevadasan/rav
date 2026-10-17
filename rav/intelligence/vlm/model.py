"""Hugging Face local VLM implementation with optional QLoRA support."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

from rav.core.types import CameraFrame
from rav.intelligence.vlm.base import BaseVLM


def _dependencies() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "VLM dependencies are missing. Install them with: pip install -e '.[vlm]'"
        ) from exc
    return torch, AutoProcessor, AutoModelForImageTextToText


class HuggingFaceVLM(BaseVLM):
    """Local Transformers image-text backend; model ID remains configuration-driven."""

    def __init__(
        self,
        model_id: str,
        *,
        device: str = "auto",
        revision: str | None = None,
        trust_remote_code: bool = False,
        quantization: str = "none",
    ) -> None:
        self.model_id, self.device, self.revision = model_id, device, revision
        self.trust_remote_code, self.quantization = trust_remote_code, quantization
        self.model: Any | None = None
        self.processor: Any | None = None

    def load(self) -> HuggingFaceVLM:
        torch, AutoProcessor, Model = _dependencies()
        kwargs: dict[str, Any] = {
            "revision": self.revision,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.quantization == "4bit":
            try:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
            except ImportError as exc:
                raise RuntimeError("4-bit quantization requires bitsandbytes.") from exc
        elif self.device.startswith("cuda") and torch.cuda.is_available():
            kwargs["torch_dtype"] = torch.float16
        if self.device == "auto":
            kwargs["device_map"] = "auto"
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, revision=self.revision, trust_remote_code=self.trust_remote_code
        )
        self.model = Model.from_pretrained(self.model_id, **kwargs)
        if self.device not in {"auto", "cuda"}:
            self.model.to(self.device)
        return self

    def _require_loaded(self) -> tuple[Any, Any]:
        if self.model is None or self.processor is None:
            raise RuntimeError("VLM is not loaded; call load() first.")
        return self.model, self.processor

    def prepare_inputs(self, image: Any, prompt: str, **_kwargs: Any) -> Any:
        _, processor = self._require_loaded()
        from rav.intelligence.vlm.processor import inference_messages

        if isinstance(image, CameraFrame):
            if image.payload is None:
                raise ValueError("Camera frame does not contain an image payload.")
            from PIL import Image

            with Image.open(BytesIO(image.payload)) as decoded:
                image = decoded.convert("RGB")
        messages = inference_messages(prompt)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        batch = processor(text=[text], images=[image], padding=True, return_tensors="pt")
        model, _ = self._require_loaded()
        return batch.to(model.device)

    def generate(self, image: Any, prompt: str, **kwargs: Any) -> str:
        model, processor = self._require_loaded()
        inputs = self.prepare_inputs(image, prompt)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=int(kwargs.get("max_new_tokens", 128)),
            repetition_penalty=float(kwargs.get("repetition_penalty", 1.15)),
            no_repeat_ngram_size=int(kwargs.get("no_repeat_ngram_size", 3)),
        )
        trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, output_ids, strict=True)]
        return processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    def forward(self, **inputs: Any) -> Any:
        model, _ = self._require_loaded()
        return model(**inputs)

    def save(self, destination: str) -> None:
        model, processor = self._require_loaded()
        path = Path(destination)
        path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(path)
        processor.save_pretrained(path)

    def load_checkpoint(self, source: str) -> None:
        model, _ = self._require_loaded()
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("Loading an adapter requires peft.") from exc
        self.model = PeftModel.from_pretrained(model, source)
