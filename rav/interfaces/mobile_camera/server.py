"""Serve the local RAV phone-camera question-answering prototype."""

from __future__ import annotations

import argparse
import logging
import os
import threading
import webbrowser
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from starlette.concurrency import run_in_threadpool

from rav.intelligence.vlm.base import BaseVLM
from rav.perception.camera.uploaded_image import UploadedImageCameraSource

logger = logging.getLogger("rav.mobile_camera")
DEFAULT_MAX_IMAGE_BYTES = 8 * 1024 * 1024


def create_app(vlm: BaseVLM, *, max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES) -> FastAPI:
    """Create the HTTP app around an already-loaded RAV VLM."""
    if max_image_bytes < 1:
        raise ValueError("max_image_bytes must be positive.")
    app = FastAPI(title="RAV Mobile Camera", docs_url=None, redoc_url=None)
    inference_lock = threading.Lock()

    @app.get("/", response_class=HTMLResponse)
    async def camera_page() -> str:
        from rav.interfaces.mobile_camera.page import CAMERA_PAGE

        return CAMERA_PAGE

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/analyze")
    async def analyze(
        request: Request,
        prompt: str = Query(default="Describe what you see around me.", max_length=500),
    ) -> dict[str, str]:
        content_type = request.headers.get("content-type", "").split(";", 1)[0].lower()
        if content_type not in {"image/jpeg", "image/png"}:
            raise HTTPException(
                status_code=415, detail="Send a JPEG or PNG image as the request body."
            )
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > max_image_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Image exceeds the {max_image_bytes:,}-byte upload limit.",
                    )
            except ValueError as exc:
                raise HTTPException(
                    status_code=400, detail="Invalid Content-Length header."
                ) from exc

        chunks: list[bytes] = []
        received = 0
        async for chunk in request.stream():
            received += len(chunk)
            if received > max_image_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"Image exceeds the {max_image_bytes:,}-byte upload limit.",
                )
            chunks.append(chunk)
        if not received:
            raise HTTPException(status_code=400, detail="Image request body is empty.")
        if not prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt must not be empty.")

        try:
            frame = UploadedImageCameraSource(b"".join(chunks)).capture()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        def infer() -> str:
            with inference_lock:
                return vlm.answer(
                    frame,
                    prompt.strip(),
                    max_new_tokens=64,
                    repetition_penalty=1.15,
                    no_repeat_ngram_size=3,
                )

        try:
            answer = await run_in_threadpool(infer)
        except Exception as exc:
            logger.exception("Local VLM inference failed for uploaded camera frame.")
            raise HTTPException(status_code=503, detail=f"RAV inference failed: {exc}") from exc
        return {"response": answer}

    return app


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="HuggingFaceTB/SmolVLM-256M-Instruct")
    parser.add_argument(
        "--adapter",
        default=None,
        help="Optional PEFT adapter directory; omit to use the original pretrained model.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--quantization", choices=["none", "4bit"], default="4bit")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Load the base model only from the local Hugging Face cache.",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open the camera interface in a browser after the server starts.",
    )
    parser.add_argument("--max-image-bytes", type=int, default=DEFAULT_MAX_IMAGE_BYTES)
    parser.add_argument("--ssl-certfile", default=None)
    parser.add_argument("--ssl-keyfile", default=None)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535.")
    if args.max_image_bytes < 1:
        parser.error("--max-image-bytes must be positive.")
    if bool(args.ssl_certfile) != bool(args.ssl_keyfile):
        parser.error("--ssl-certfile and --ssl-keyfile must be provided together.")
    for certificate in (args.ssl_certfile, args.ssl_keyfile):
        if certificate and not Path(certificate).is_file():
            parser.error(f"TLS file does not exist: {certificate}")
    return args


def _register_browser_open(app: FastAPI, url: str) -> None:
    def open_frontend() -> None:
        if not webbrowser.open_new_tab(url):
            logger.warning("Could not open a browser automatically; visit %s", url)

    app.router.add_event_handler("startup", open_frontend)


def main() -> None:
    args = _arguments()
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; use --device cpu only for diagnostics.")
    adapter = Path(args.adapter) if args.adapter else None
    if adapter is not None:
        if not adapter.is_dir():
            raise SystemExit(f"Trained RAV adapter does not exist: {adapter}")
        if not (adapter / "adapter_config.json").is_file() or not (
            adapter / "adapter_model.safetensors"
        ).is_file():
            raise SystemExit(f"Adapter checkpoint is incomplete: {adapter}")
    from rav.intelligence.vlm.model import HuggingFaceVLM

    model_reference = args.model
    if args.offline:
        from huggingface_hub import snapshot_download

        try:
            model_reference = snapshot_download(args.model, local_files_only=True)
        except Exception as exc:
            raise SystemExit(
                f"Base model {args.model} is not available in the local cache. "
                "Run scripts/vlm/prefetch_model.py while online first."
            ) from exc
        print(f"Using cached base model: {model_reference}", flush=True)

    vlm = HuggingFaceVLM(
        model_reference,
        device="auto" if args.device == "cuda" else args.device,
        quantization=args.quantization,
    ).load()
    if adapter is not None:
        vlm.load_checkpoint(str(adapter))
    app = create_app(vlm, max_image_bytes=args.max_image_bytes)
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "The camera server dependencies are missing. Install with: "
            "python -m pip install -e '.[mobile-camera]'"
        ) from exc

    scheme = "https" if args.ssl_certfile else "http"
    if args.open_browser:
        _register_browser_open(app, f"{scheme}://localhost:{args.port}")
    logging.basicConfig(level=logging.INFO)
    print(f"RAV camera server listening on {scheme}://{args.host}:{args.port}")
    if adapter is None:
        print("Using the original pretrained SmolVLM model (no adapter).")
    else:
        print(f"Loaded adapter: {adapter.resolve()}")
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        ssl_certfile=args.ssl_certfile,
        ssl_keyfile=args.ssl_keyfile,
    )


if __name__ == "__main__":
    main()
