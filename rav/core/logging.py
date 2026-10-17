"""Structured logging setup owned by the application boundary."""

from __future__ import annotations

import logging


def configure_logging(level: str) -> None:
    """Configure concise process-wide logging for local RAV processes."""
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )
