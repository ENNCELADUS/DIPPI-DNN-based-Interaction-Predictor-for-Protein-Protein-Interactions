"""Convenience entrypoint for running the embed CLI on HPC systems."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    # Allow `python src/embed/run.py ...` by ensuring the repo root is on sys.path.
    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.embed.cli.main import main as cli_main
else:  # pragma: no cover - exercised when run as a module
    from .cli.main import main as cli_main


def main(argv: list[str] | None = None) -> int:
    """Delegate to :mod:`src.embed.cli.main`."""

    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
