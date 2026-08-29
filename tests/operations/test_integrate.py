"""Parity tests for integrate operations."""

from __future__ import annotations

import dascore as dc
import numpy as np
import pytest

from dasjax import JaxPatchPipeline
from dasjax.kernels import integrate_kernel

from .helpers import assert_compiled_matches_dascore, assert_patch_close


CASES = (("integrate", (), {"dim": "time"}, lambda p: p.integrate(dim="time")),)


@pytest.mark.parametrize("case", CASES, ids=lambda item: item[0])
def test_compiled_integrate_operation_matches_dascore(case) -> None:
    """Match compiled integrate operations against DASCore baselines."""
    assert_compiled_matches_dascore(case)


def test_integrate_definite_path() -> None:
    """Cover definite integration."""
    patch = dc.get_example_patch()
    pipeline = JaxPatchPipeline().integrate(dim="time", definite=True)

    assert_patch_close(
        pipeline.compile()(patch),
        patch.integrate(dim="time", definite=True),
    )


def test_integrate_vector_spacing_kernel_path() -> None:
    """Cover vector-spacing integration kernel branches."""
    out = np.asarray(
        integrate_kernel(np.arange(5.0), axis=0, dx_or_spacing=np.arange(5.0) ** 2)
    )
    assert out.shape == (5,)
