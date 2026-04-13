"""Keep README usage examples executable."""

from __future__ import annotations

from pathlib import Path

import pytest


README_PATH = Path(__file__).resolve().parents[1] / "README.md"


def _iter_usage_python_blocks(readme_text: str) -> list[str]:
    """Return Python fenced code blocks from the README usage section."""
    try:
        usage_start = readme_text.index("## Usage")
    except ValueError as exc:
        raise AssertionError("README is missing a Usage section.") from exc

    remaining = readme_text[usage_start:]
    next_section = remaining.find("\n## ", len("## Usage"))
    usage_text = remaining if next_section == -1 else remaining[:next_section]

    blocks: list[str] = []
    parts = usage_text.split("```python\n")
    for part in parts[1:]:
        code, _, _ = part.partition("\n```")
        blocks.append(code.strip())
    return blocks


def _readme_usage_blocks() -> list[str]:
    blocks = _iter_usage_python_blocks(README_PATH.read_text())
    assert blocks, "README Usage section must contain Python examples."
    return blocks


@pytest.mark.parametrize("code", _readme_usage_blocks())
def test_readme_usage_examples_execute(code: str) -> None:
    """Execute each README usage example without errors."""
    namespace = {"__name__": "__readme_example__"}
    exec(code, namespace)
