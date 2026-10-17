from __future__ import annotations

from typing import Any

import anyio
import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

pytest.importorskip("fastapi")

from rav.core.types import CameraFrame  # noqa: E402
from rav.intelligence.vlm.base import BaseVLM  # noqa: E402
from rav.interfaces.mobile_camera.server import _register_browser_open, create_app  # noqa: E402


class StubVLM(BaseVLM):
    def __init__(self, *, fail: bool = False) -> None:
        self.received_frame: CameraFrame | None = None
        self.received_prompt: str | None = None
        self.fail = fail

    def load(self) -> StubVLM:
        return self

    def prepare_inputs(self, _image: Any, _prompt: str, **_kwargs: Any) -> Any:
        return None

    def generate(self, image: Any, prompt: str, **_kwargs: Any) -> str:
        if self.fail:
            raise RuntimeError("test inference failure")
        assert isinstance(image, CameraFrame)
        self.received_frame = image
        self.received_prompt = prompt
        return "A test response grounded in the uploaded image."

    def forward(self, **_inputs: Any) -> Any:
        return None

    def save(self, _destination: str) -> None:
        return None

    def load_checkpoint(self, _source: str) -> None:
        return None


def _png() -> bytes:
    import io

    buffer = io.BytesIO()
    Image.new("RGB", (24, 16), color="orange").save(buffer, format="PNG")
    return buffer.getvalue()


def _post(app: Any, path: str, content: bytes, content_type: str) -> Any:
    async def send() -> Any:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            return await client.post(path, content=content, headers={"content-type": content_type})

    return anyio.run(send)


def test_camera_page_and_health_endpoint() -> None:
    async def check() -> tuple[Any, Any]:
        async with AsyncClient(
            transport=ASGITransport(app=create_app(StubVLM())),
            base_url="http://test",
        ) as client:
            return await client.get("/"), await client.get("/health")

    page, health = anyio.run(check)

    assert page.status_code == 200
    assert "RAV — Reasoning-Aware Vision" in page.text
    assert "getUserMedia" in page.text
    assert health.json() == {"status": "ok"}


def test_browser_opens_from_supported_startup_hook(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []

    def record_open(url: str) -> bool:
        opened.append(url)
        return True

    monkeypatch.setattr("rav.interfaces.mobile_camera.server.webbrowser.open_new_tab", record_open)
    app = create_app(StubVLM())

    _register_browser_open(app, "http://localhost:8000")
    for handler in app.router.on_startup:
        handler()

    assert opened == ["http://localhost:8000"]


def test_analyze_uses_phone_frame_and_user_prompt() -> None:
    vlm = StubVLM()
    result = _post(
        create_app(vlm),
        "/analyze?prompt=What%20is%20on%20the%20table%3F",
        _png(),
        "image/png",
    )

    assert result.status_code == 200
    assert result.json() == {"response": "A test response grounded in the uploaded image."}
    assert vlm.received_frame is not None
    assert vlm.received_frame.width == 24
    assert vlm.received_prompt == "What is on the table?"


def test_analyze_reports_invalid_content_and_oversized_uploads() -> None:
    app = create_app(StubVLM(), max_image_bytes=32)
    invalid_type = _post(app, "/analyze", b"x", "text/plain")
    oversized = _post(app, "/analyze", _png(), "image/png")

    assert invalid_type.status_code == 415
    assert oversized.status_code == 413


def test_analyze_returns_clear_model_error() -> None:
    result = _post(create_app(StubVLM(fail=True)), "/analyze", _png(), "image/png")

    assert result.status_code == 503
    assert "RAV inference failed" in result.json()["detail"]
