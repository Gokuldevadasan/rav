"""Minimal runnable entry point for the RAV foundation."""

from pathlib import Path

from rav.core.system import RAVSystem


def main() -> None:
    system = RAVSystem.from_yaml(Path("configs/system.yaml"))
    print(system.health())


if __name__ == "__main__":
    main()
