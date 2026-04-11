"""Test bootstrap for local namespace and pytree registration."""

from __future__ import annotations

import sys
from pathlib import Path

import dascore as dc
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import dasjax  # noqa: F401,E402

EXAMPLE_PATCH_NAMES = (
    "random_das",
    "patch_with_null",
    "random_patch_with_lat_lon",
    "random_patch_with_xyz",
    "wacky_dim_coords_patch",
    "chirp",
    "example_event_1",
    "deformation_rate_event_1",
    "nd_patch",
    "delta_patch",
)


@pytest.fixture(scope="session")
def example_patch_suite() -> tuple[tuple[str, dc.Patch], ...]:
    """Return a broad suite of local DASCore example patches."""
    return tuple((name, dc.get_example_patch(name)) for name in EXAMPLE_PATCH_NAMES)


@pytest.fixture(params=EXAMPLE_PATCH_NAMES, ids=EXAMPLE_PATCH_NAMES)
def example_patch(request) -> dc.Patch:
    """Return one example patch from the shared patch suite."""
    return dc.get_example_patch(request.param)
