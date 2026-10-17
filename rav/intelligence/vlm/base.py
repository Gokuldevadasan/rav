"""Provider-neutral VLM contract; implementations belong in separate modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from rav.core.types import CameraFrame


class BaseVLM(ABC):
    """A replaceable image-language model backend.

    Implementations may be local Hugging Face models, remote APIs, or exported
    runtimes.  Importantly, callers never need to know which one is in use.
    """

    @abstractmethod
    def load(self) -> BaseVLM:
        """Load model weights and processor resources."""

    @abstractmethod
    def prepare_inputs(self, image: Any, prompt: str, **kwargs: Any) -> Any:
        """Convert a generic image input and prompt into backend inputs."""

    @abstractmethod
    def generate(self, image: Any, prompt: str, **kwargs: Any) -> str:
        """Generate a response grounded in *image*."""

    @abstractmethod
    def forward(self, **inputs: Any) -> Any:
        """Run a training/evaluation forward pass."""

    @abstractmethod
    def save(self, destination: str) -> None:
        """Save the implementation's trainable state."""

    @abstractmethod
    def load_checkpoint(self, source: str) -> None:
        """Restore an implementation checkpoint or adapter."""

    def answer(self, frame: CameraFrame, prompt: str, **kwargs: Any) -> str:
        """Compatibility entrypoint for camera adapters.

        Camera payload decoding remains an adapter concern; local VLMs accept
        PIL, NumPy, Torch, file-path, and camera-frame image payloads directly.
        """
        return self.generate(frame, prompt, **kwargs)
