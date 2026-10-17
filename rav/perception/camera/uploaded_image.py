"""Camera source for one image uploaded by a browser or another HTTP client."""

from __future__ import annotations

from io import BytesIO
from uuid import uuid4

from PIL import Image, ImageOps, UnidentifiedImageError

from rav.core.types import CameraFrame, SourceType
from rav.perception.camera.base import CameraSource


class UploadedImageCameraSource(CameraSource):
    """Validate and normalize one JPEG/PNG upload into a transport-neutral frame."""

    def __init__(
        self,
        payload: bytes,
        *,
        max_dimension: int = 1024,
        max_pixels: int = 20_000_000,
    ) -> None:
        self._payload = payload
        self._max_dimension = max_dimension
        self._max_pixels = max_pixels

    def capture(self) -> CameraFrame:
        try:
            with Image.open(BytesIO(self._payload)) as image:
                if image.format not in {"JPEG", "PNG"}:
                    raise ValueError("Only JPEG and PNG camera images are supported.")
                width, height = image.size
                if width < 1 or height < 1 or width * height > self._max_pixels:
                    raise ValueError(
                        f"Image dimensions exceed the {self._max_pixels:,}-pixel limit."
                    )
                rgb = image.convert("RGB")
                if max(width, height) > self._max_dimension:
                    normalized = ImageOps.contain(
                        rgb,
                        (self._max_dimension, self._max_dimension),
                        method=Image.Resampling.LANCZOS,
                    )
                else:
                    normalized = rgb
        except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as exc:
            raise ValueError("Upload is not a valid JPEG or PNG image.") from exc

        sanitized = BytesIO()
        normalized.save(sanitized, format="JPEG", quality=90, optimize=True)
        return CameraFrame(
            id=uuid4().hex,
            source=SourceType.PHONE,
            width=normalized.width,
            height=normalized.height,
            payload=sanitized.getvalue(),
        )
