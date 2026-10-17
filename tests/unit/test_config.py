from pathlib import Path

from rav.core.config import load_config
from rav.core.types import SourceType


def test_load_system_config() -> None:
    config = load_config(Path("configs/system.yaml"))

    assert config.system.name == "RAV"
    assert config.camera.source is SourceType.PHONE
