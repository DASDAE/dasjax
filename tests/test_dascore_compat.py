"""Parity on the cases that separate the DASCore versions dasjax supports.

The shared fixture suite in `conftest` happens not to discriminate here: no
example patch has both a slice whose most negative sample outweighs its most
positive one and a slice that is nothing but nulls. Those are exactly the
slices DASCore has changed its mind about, so they get their own patches.
"""

from __future__ import annotations

import dascore as dc
import numpy as np
import pytest

from dasjax import JaxPatchPipeline
from dasjax.operations import get_operation

NORMS = ("l1", "l2", "max", "bit")


def _patch(data: np.ndarray) -> dc.Patch:
    """Build a small two-dimensional patch with plain coordinates."""
    return dc.Patch(
        data=data,
        coords={"one": np.arange(data.shape[0]), "many": np.arange(data.shape[1])},
        dims=("one", "many"),
    )


@pytest.fixture(scope="module")
def negative_dominant_patch() -> dc.Patch:
    """A patch whose first slice peaks negative and second peaks positive."""
    return _patch(np.array([[-2.0, 1.0, 0.5], [3.0, -1.0, 2.0]]))


@pytest.fixture(scope="module")
def null_patch() -> dc.Patch:
    """A patch with a null-bearing slice, an all-zero slice, and zeros+null."""
    return _patch(
        np.array(
            [
                [np.nan, 3.0, 4.0],
                [0.0, 0.0, 0.0],
                [0.0, np.nan, 0.0],
                [-5.0, 2.0, np.nan],
            ]
        )
    )


def _assert_matches_dascore(patch: dc.Patch, norm: str) -> None:
    """Both dasjax paths must agree with DASCore's own normalize."""
    expected = patch.normalize("many", norm=norm).data
    eager = get_operation("normalize").patch_impl(patch, dim="many", norm=norm)
    compiled = JaxPatchPipeline().normalize(dim="many", norm=norm).compile()(patch)
    for name, out in (("eager", eager), ("compiled", compiled)):
        np.testing.assert_allclose(
            np.asarray(out.data),
            np.asarray(expected),
            rtol=1e-6,
            atol=1e-8,
            equal_nan=True,
            err_msg=f"{name} normalize(norm={norm!r}) disagrees with DASCore",
        )


@pytest.mark.parametrize("norm", NORMS)
def test_normalize_matches_dascore_when_peak_is_negative(negative_dominant_patch, norm):
    """Whichever peak DASCore divides by, dasjax divides by the same one."""
    _assert_matches_dascore(negative_dominant_patch, norm)


@pytest.mark.parametrize("norm", NORMS)
def test_normalize_matches_dascore_on_nulls_and_zeros(null_patch, norm):
    """Nulls and zero norms are handled the way this DASCore handles them."""
    _assert_matches_dascore(null_patch, norm)


@pytest.mark.parametrize("definite", [False, True])
def test_integrate_matches_dascore(negative_dominant_patch, definite):
    """Integrate survives DASCore moving coordinates out of the attributes."""
    patch = negative_dominant_patch
    expected = patch.integrate("many", definite=definite)
    compiled = (
        JaxPatchPipeline().integrate(dim="many", definite=definite).compile()(patch)
    )
    np.testing.assert_allclose(
        np.asarray(compiled.data),
        np.asarray(expected.data),
        rtol=1e-6,
        atol=1e-8,
        equal_nan=True,
    )
    assert compiled.dims == expected.dims
