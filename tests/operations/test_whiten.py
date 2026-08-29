"""Parity tests for whiten operations."""

from __future__ import annotations

import dascore as dc
import numpy as np
import pytest

from dasjax import JaxPatchPipeline
from dasjax.kernels import whiten_kernel

from .helpers import assert_compiled_matches_dascore, assert_patch_close, even_patch


CASES = (("whiten", (), {"time": None}, lambda p: p.whiten(time=None), even_patch),)


@pytest.mark.parametrize("case", CASES, ids=lambda item: item[0])
def test_compiled_whiten_operation_matches_dascore(case) -> None:
    """Match compiled whiten operations against DASCore baselines."""
    assert_compiled_matches_dascore(case)


def test_whiten_kernel_and_operation_smooth_paths() -> None:
    """Cover direct whitening branches and operation smooth-size binding."""
    data = np.sin(np.linspace(0, 2 * np.pi, 16))[None, :]
    complex_data = data.astype(np.complex128) + 1j * data.astype(np.complex128)
    weight = np.linspace(1.0, 2.0, 9)

    assert np.asarray(whiten_kernel(data, axis=1)).shape == data.shape
    assert (
        np.asarray(
            whiten_kernel(
                data, axis=1, window_len=3, water_level=0.1, freq_weight=weight
            )
        ).shape
        == data.shape
    )
    assert (
        np.asarray(whiten_kernel(complex_data, axis=1, window_len=1)).shape
        == data.shape
    )

    patch = dc.get_example_patch()
    pipeline = JaxPatchPipeline().whiten(time=None, smooth_size=1.0, water_level=0.1)
    assert_patch_close(
        pipeline.compile()(patch),
        patch.whiten(time=None, smooth_size=1.0, water_level=0.1),
    )
