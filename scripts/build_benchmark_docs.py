"""Generate benchmark documentation from the tracked benchmark snapshot."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dasjax.benchmark_cli import DEFAULT_DOCS_OUTPUT, DEFAULT_SNAPSHOT, docs_command  # noqa: E402


def main() -> int:
    """Render the benchmark docs page."""
    args = type(
        "Args",
        (),
        {"snapshot": DEFAULT_SNAPSHOT, "output": DEFAULT_DOCS_OUTPUT},
    )()
    return docs_command(args)


if __name__ == "__main__":
    raise SystemExit(main())
