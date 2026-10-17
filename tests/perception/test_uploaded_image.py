from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from PIL import Image
from rav.core.types import CameraFrame, SourceType
from rav.intelligence.vlm.model import HuggingFaceVLM
from rav.perception.camera.uploaded_image import UploadedImageCameraSource


def _image_bytes(image_format: str = "PNG", size: tuple[int, int] = (1400, 800)) -> bytes:
    image = Image.new("RGB", size, color="navy")
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


def test_uploaded_image_becomes_bounded_phone_camera_frame() -> None:
    frame = UploadedImageCameraSource(_image_bytes()).capture()

    assert isinstance(frame, CameraFrame)
    assert frame.source is SourceType.PHONE
    assert frame.width == 1024
    assert frame.height < 800
    assert frame.payload is not None
    with Image.open(BytesIO(frame.payload)) as image:
        assert image.format == "JPEG"
        assert image.size == (frame.width, frame.height)


def test_uploaded_image_rejects_corrupt_payload() -> None:
    with pytest.raises(ValueError, match="valid JPEG or PNG"):
        UploadedImageCameraSource(b"not an image").capture()


def test_uploaded_image_rejects_unsupported_format() -> None:
    with pytest.raises(ValueError, match="Only JPEG and PNG"):
        UploadedImageCameraSource(_image_bytes("GIF")).capture()


def test_uploaded_image_rejects_excessive_pixel_count() -> None:
    with pytest.raises(ValueError, match="pixel limit"):
        UploadedImageCameraSource(_image_bytes(size=(4500, 4500)), max_pixels=20_000_000).capture()


class _Batch(dict[str, Any]):
    def to(self, _device: Any) -> _Batch:
        return self

    @property
    def input_ids(self) -> torch.Tensor:
        return self["input_ids"]


class _Processor:
    def __init__(self) -> None:
        self.images: list[Image.Image] = []

    def apply_chat_template(self, _messages: Any, **_kwargs: Any) -> str:
        return "rendered prompt"

    def __call__(self, *, images: list[Image.Image], **_kwargs: Any) -> _Batch:
        self.images = images
        return _Batch(input_ids=torch.tensor([[1, 2]]))

    def batch_decode(self, _tokens: Any, **_kwargs: Any) -> list[str]:
        return ["A concise grounded response."]


class _GenerationModel:
    device = "cpu"

    def __init__(self) -> None:
        self.generation_options: dict[str, Any] = {}

    def generate(self, **kwargs: Any) -> torch.Tensor:
        self.generation_options = kwargs
        return torch.tensor([[1, 2, 3]])


def test_vlm_prepares_uploaded_camera_frame_as_rgb_image() -> None:
    vlm = HuggingFaceVLM("unused-test-model")
    processor = _Processor()
    vlm.model = SimpleNamespace(device="cpu")
    vlm.processor = processor
    frame = UploadedImageCameraSource(_image_bytes(size=(24, 16))).capture()

    vlm.prepare_inputs(frame, "What is visible?")

    assert len(processor.images) == 1
    assert processor.images[0].mode == "RGB"
    assert processor.images[0].size == (24, 16)


def test_vlm_uses_bounded_non_repeating_generation_defaults() -> None:
    vlm = HuggingFaceVLM("unused-test-model")
    processor = _Processor()
    model = _GenerationModel()
    vlm.model = model
    vlm.processor = processor
    image = Image.new("RGB", (24, 16))

    response = vlm.generate(image, "What do you see?")

    assert response == "A concise grounded response."
    assert model.generation_options["max_new_tokens"] == 128
    assert model.generation_options["repetition_penalty"] == 1.15
    assert model.generation_options["no_repeat_ngram_size"] == 3
