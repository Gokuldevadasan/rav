"""Camera input contract, independent of phone, ROS, or simulation transport."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rav.core.types import CameraFrame


class CameraSource(ABC):
    """Supplies frames without leaking capture-technology details upstream."""

    @abstractmethod
    def capture(self) -> CameraFrame:
        """Capture one frame or raise a camera-specific error."""
