"""Parity tests for differentiate operations."""

from __future__ import annotations

import dascore as dc
import numpy as np
import pytest
from dascore.exceptions import ParameterError

from dasjax import JaxPatchPipeline
from dasjax.kernels import differentiate_kernel

from .helpers import assert_compiled_matches_dascore, assert_patch_close


CASES = (("differentiate", (), {"dim": "time"}, lambda p: p.differentiate(dim="time")),)


@pytest.mark.parametrize("case", CASES, ids=lambda item: item[0])
def test_compiled_differentiate_operation_matches_dascore(case) -> None:
    """Match compiled differentiate operations against DASCore baselines."""
    assert_compiled_matches_dascore(case)


def test_differentiate_step_and_uneven_spacing_paths() -> None:
    """Cover stepped differentiation and uneven coordinate spacing."""
    patch = dc.get_example_patch()
    stepped = JaxPatchPipeline().differentiate(dim="time", step=2).compile()(patch)
    assert_patch_close(stepped, patch.differentiate(dim="time", step=2))

    data = np.arange(5.0) ** 2
    spacing = np.asarray([0.0, 0.5, 1.5, 3.0, 5.0])
    out = np.asarray(differentiate_kernel(data, axis=0, dx_or_spacing=spacing))
    assert out.shape == data.shape

    with pytest.raises(NotImplementedError, match="order=2"):
        differentiate_kernel(data, axis=0, dx_or_spacing=1.0, order=1)
    with pytest.raises(ValueError, match="too small"):
        differentiate_kernel(np.arange(2.0), axis=0, dx_or_spacing=1.0)
    with pytest.raises(ParameterError, match="only be used along one axis"):
        JaxPatchPipeline().differentiate(dim=("time", "distance"), step=2).compile()(
            patch
        )
