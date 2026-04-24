"""Parity tests for spectral operations."""

from __future__ import annotations

import dascore as dc
import jax.numpy as jnp
import numpy as np
import pytest

from dasjax import JaxPatchPipeline
from dasjax.kernels.spectral import (
    _extract_zero_padded_frames,
    _linear_detrend_frames,
    banded_stft_kernel,
)

from .helpers import assert_compiled_matches_dascore, assert_patch_close, fbe_baseline


CASES = (
    (
        "fbe",
        (),
        {"time": 64, "samples": True, "overlap": 32, "fmin": 2.0, "fmax": 10.0},
        fbe_baseline,
        lambda p: dc.get_example_patch("chirp"),
    ),
)


@pytest.mark.parametrize("case", CASES, ids=lambda item: item[0])
def test_compiled_spectral_operation_matches_dascore(case) -> None:
    """Match compiled spectral operations against DASCore baselines."""
    assert_compiled_matches_dascore(case)


def test_banded_stft_edges() -> None:
    """Cover STFT detrending, NaN frame handling, and one-sample frame helper."""
    data = np.arange(8.0)[None, :]
    data[0, 0] = np.nan
    window = np.hanning(4)
    frame_starts = np.asarray([-1, 1, 6])
    out = np.asarray(
        banded_stft_kernel(
            data,
            axis=1,
            window=window,
            frame_starts=frame_starts,
            selected_bins=np.asarray([0, 1]),
            sample_step=0.1,
            detrend=True,
        )
    )

    assert out.shape == (1, 3)
    assert out[0, 0] == 0.0
    assert np.array_equal(
        np.asarray(_linear_detrend_frames(jnp.ones((1, 1)))),
        np.ones((1, 1)),
    )
    frames = np.asarray(
        _extract_zero_padded_frames(jnp.asarray([[1.0, 2.0]]), jnp.asarray([-1, 1]), 2)
    )
    assert frames.tolist() == [[[0.0, 1.0], [2.0, 0.0]]]


def test_fbe_detrend_and_nan_paths_match_dascore() -> None:
    """Cover FBE detrending and non-finite frame handling from the operation layer."""
    patch = dc.get_example_patch("chirp")
    pipeline = JaxPatchPipeline().fbe(
        time=64,
        samples=True,
        overlap=32,
        fmin=2.0,
        fmax=10.0,
        detrend=True,
    )

    expected = patch.stft(
        time=64,
        samples=True,
        overlap=32,
        detrend=True,
    ).abs()
    ft_dim = next(dim for dim in expected.dims if dim.startswith("ft_"))
    expected = expected.select(**{ft_dim: (2.0, 10.0)})
    expected = expected.sum(dim=ft_dim, dim_reduce="squeeze")

    assert_patch_close(pipeline.compile()(patch), expected)
