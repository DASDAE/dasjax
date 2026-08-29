"""Parity tests for taper operations."""

from __future__ import annotations

import dascore as dc
import pytest
from dascore.exceptions import ParameterError

from dasjax import JaxPatchPipeline
from dasjax.operations.taper import _taper_coord_inds

from .helpers import assert_compiled_matches_dascore, assert_patch_close


CASES = (
    (
        "taper",
        (),
        {"time": 0.05, "window_type": "hann"},
        lambda p: p.taper(time=0.05, window_type="hann"),
    ),
    (
        "taper_range",
        (),
        {"time": (10, 20), "samples": True, "window_type": "hann"},
        lambda p: p.taper_range(time=(10, 20), samples=True, window_type="hann"),
    ),
)

# DASCore's own names for hann and triang, which scipy does not know.
DASCORE_WINDOW_NAMES = ("cos", "ramp")


@pytest.mark.parametrize("case", CASES, ids=lambda item: item[0])
def test_compiled_taper_operation_matches_dascore(case) -> None:
    """Match compiled taper operations against DASCore baselines."""
    assert_compiled_matches_dascore(case)


@pytest.mark.parametrize("window_type", DASCORE_WINDOW_NAMES)
def test_dascore_window_names_match_dascore(window_type) -> None:
    """Match DASCore's own window names, which scipy spells differently."""
    patch = dc.get_example_patch()
    taper = JaxPatchPipeline().taper(time=0.05, window_type=window_type)
    ranged = JaxPatchPipeline().taper_range(
        time=(10, 20), samples=True, window_type=window_type
    )

    assert_patch_close(
        taper.compile()(patch),
        patch.taper(time=0.05, window_type=window_type),
    )
    assert_patch_close(
        ranged.compile()(patch),
        patch.taper_range(time=(10, 20), samples=True, window_type=window_type),
    )


def test_taper_range_edges_and_errors() -> None:
    """Cover nested ranges, invert mode, and invalid taper values."""
    patch = dc.get_example_patch()
    pipeline = JaxPatchPipeline().taper_range(
        time=((10, 20), (30, 40)),
        samples=True,
        invert=True,
        window_type="hann",
    )
    expected = patch.taper_range(
        time=((10, 20), (30, 40)),
        samples=True,
        invert=True,
        window_type="hann",
    )

    assert_patch_close(pipeline.compile()(patch), expected)

    with pytest.raises(ParameterError, match="Cannot use"):
        JaxPatchPipeline().taper_range(time=(None, 10), samples=True).compile()(patch)
    with pytest.raises(ParameterError, match="len 2 or 4"):
        JaxPatchPipeline().taper_range(time=(1, 2, 3), samples=True).compile()(patch)
    with pytest.raises(ParameterError, match="len 2 or 4"):
        _taper_coord_inds(patch.get_coord("time"), 1, False, True)
    with pytest.raises(ParameterError, match="overlap"):
        JaxPatchPipeline().taper(time=(0.6, 0.6)).compile()(patch)

    coord = patch.get_coord("time")
    assert _taper_coord_inds(coord, (None, 2, 4, ...), False, True)[0][0] == 0
