"""Small composition root for the currently implemented RAV foundation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rav.core.config import RAVConfig, load_config
from rav.core.logging import configure_logging
from rav.core.types import SystemHealth


@dataclass(frozen=True, slots=True)
class RAVSystem:
    """Application composition root; future adapters are injected here."""

    config: RAVConfig

    @classmethod
    def from_yaml(cls, path: Path) -> RAVSystem:
        config = load_config(path)
        configure_logging(config.system.log_level)
        return cls(config=config)

    def health(self) -> SystemHealth:
        """Return local foundation health without probing models or cameras."""
        return SystemHealth(
            system_name=self.config.system.name,
            environment=self.config.system.environment,
        )
