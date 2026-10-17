from pathlib import Path

from rav.core.system import RAVSystem


def test_system_returns_health() -> None:
    system = RAVSystem.from_yaml(Path("configs/system.yaml"))

    health = system.health()

    assert health.status == "ok"
    assert health.system_name == "RAV"
