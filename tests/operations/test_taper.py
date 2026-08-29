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


@pytest.mark.parametrize("case", CASES, ids=lambda item: item[0])
def test_compiled_taper_operation_matches_dascore(case) -> None:
    """Match compiled taper operations against DASCore baselines."""
    assert_compiled_matches_dascore(case)


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
