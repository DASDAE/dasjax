"""Parity tests for Hilbert operations."""

from __future__ import annotations

import dascore as dc
import numpy as np
import pytest

from dasjax import JaxPatchPipeline
from dasjax.kernels.spectral import _analytic_multiplier

from .helpers import assert_compiled_matches_dascore, even_patch


CASES = (
    ("hilbert", (), {"dim": "time"}, lambda p: p.hilbert(dim="time"), even_patch),
    ("envelope", (), {"dim": "time"}, lambda p: p.envelope(dim="time"), even_patch),
)


@pytest.mark.parametrize("case", CASES, ids=lambda item: item[0])
def test_compiled_hilbert_operation_matches_dascore(case) -> None:
    """Match compiled Hilbert operations against DASCore baselines."""
    assert_compiled_matches_dascore(case)


def test_hilbert_even_multiplier_and_error() -> None:
    """Cover even analytic multiplier and uneven-coordinate Hilbert errors."""
    assert _analytic_multiplier(4, "complex128").tolist() == [1, 2, 1, 0]

    patch = dc.get_example_patch()
    coord = patch.get_coord("time").values.astype("datetime64[ns]")
    coord[1000:] += np.timedelta64(1, "ms")
    uneven = patch.update(
        coords={
            "distance": patch.get_coord("distance").values,
            "time": coord,
        }
    )
    with pytest.raises(dc.exceptions.CoordError):
        JaxPatchPipeline().hilbert(dim="time").compile()(uneven)


def test_phase_weighted_stack_infers_transform_dim_and_errors() -> None:
    """Cover transform-dimension inference and uneven-coordinate validation."""
    patch = even_patch(dc.get_example_patch())

    assert np.allclose(
        JaxPatchPipeline().phase_weighted_stack("distance").compile()(patch).data,
        patch.phase_weighted_stack("distance").data,
        rtol=1e-5,
        atol=1e-6,
    )

    coord = patch.get_coord("time").values.astype("datetime64[ns]")
    coord[1000:] += np.timedelta64(1, "ms")
    uneven = patch.update(
        coords={
            "distance": patch.get_coord("distance").values,
            "time": coord,
        }
    )
    with pytest.raises(dc.exceptions.CoordError):
        JaxPatchPipeline().phase_weighted_stack("distance").compile()(uneven)

    patch_3d = dc.Patch(
        data=np.ones((2, 3, 4)),
        coords={"x": np.arange(2), "y": np.arange(3), "z": np.arange(4)},
        dims=("x", "y", "z"),
    )
    with pytest.raises(dc.exceptions.ParameterError, match="transform_dim"):
        JaxPatchPipeline().phase_weighted_stack("x").compile()(patch_3d)
