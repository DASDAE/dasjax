"""Parity tests for basic patch operations."""

from __future__ import annotations

import dascore as dc
import numpy as np
import pytest

from dasjax import JaxPatchPipeline
from dasjax.kernels import normalize_kernel

from .helpers import assert_compiled_matches_dascore, assert_patch_close, complex_patch


CASES = (
    ("identity", (), {}, lambda p: p),
    ("scale", (2.0,), {}, lambda p: p * 2.0),
    ("add", (2.0,), {}, lambda p: p + 2.0),
    ("abs", (), {}, lambda p: p.abs(), complex_patch),
    ("clip", (-0.25, 0.25), {}, lambda p: p.update(data=np.clip(p.data, -0.25, 0.25))),
    ("real", (), {}, lambda p: p.real(), complex_patch),
    ("imag", (), {}, lambda p: p.imag(), complex_patch),
    ("angle", (), {}, lambda p: p.angle(), complex_patch),
    ("conj", (), {}, lambda p: p.conj(), complex_patch),
    ("flip", ("time",), {}, lambda p: p.flip("time")),
    ("roll", (), {"time": 5, "samples": True}, lambda p: p.roll(time=5, samples=True)),
    ("standardize", (), {"dim": "time"}, lambda p: p.standardize(dim="time")),
    (
        "normalize",
        (),
        {"dim": "time", "norm": "l2"},
        lambda p: p.normalize(dim="time", norm="l2"),
    ),
    (
        "pad",
        (),
        {"time": (2, 3), "samples": True},
        lambda p: p.pad(time=(2, 3), samples=True),
    ),
)


@pytest.mark.parametrize("case", CASES, ids=lambda item: item[0])
def test_compiled_basic_operation_matches_dascore(case) -> None:
    """Match compiled basic operations against DASCore baselines."""
    assert_compiled_matches_dascore(case)


@pytest.mark.parametrize("norm", ["l1", "max", "bit"])
def test_normalize_modes_match_dascore(norm: str) -> None:
    """Cover non-default normalization modes."""
    patch = dc.get_example_patch()
    pipeline = JaxPatchPipeline().normalize(dim="time", norm=norm)

    assert_patch_close(
        pipeline.compile()(patch),
        patch.normalize(dim="time", norm=norm),
    )


def test_normalize_zero_and_invalid_mode() -> None:
    """Cover zero-norm handling and invalid normalization mode."""
    data = np.zeros((2, 3))

    assert np.array_equal(np.asarray(normalize_kernel(data, axis=1)), data)
    with pytest.raises(ValueError, match="Unsupported normalization"):
        normalize_kernel(data, axis=1, norm="bad")


def test_pad_fft_correlate_and_coordinate_padding() -> None:
    """Cover special and non-sample pad width branches."""
    patch = dc.get_example_patch()
    cases = (
        ({"time": "fft"}, lambda p: p.pad(time="fft")),
        ({"time": "correlate"}, lambda p: p.pad(time="correlate")),
        (
            {"time": np.timedelta64(1, "ms")},
            lambda p: p.pad(time=np.timedelta64(1, "ms")),
        ),
    )
    for kwargs, baseline in cases:
        out = JaxPatchPipeline().pad(**kwargs).compile()(patch)
        assert_patch_close(out, baseline(patch))


def test_roll_update_coord_is_rejected() -> None:
    """Cover unsupported roll coordinate updates."""
    patch = dc.get_example_patch()
    with pytest.raises(NotImplementedError, match="update_coord=False"):
        JaxPatchPipeline().roll(time=1, samples=True, update_coord=True).compile()(
            patch
        )


def test_aggregate_unsupported_options_are_rejected() -> None:
    """Cover aggregate validation for unsupported methods and reductions."""
    patch = dc.get_example_patch()

    with pytest.raises(NotImplementedError, match="Unsupported aggregate method"):
        JaxPatchPipeline().aggregate(dim="time", method=np.ptp).compile()(patch)
    with pytest.raises(NotImplementedError, match="dim_reduce"):
        JaxPatchPipeline().aggregate(
            dim="time",
            method="mean",
            dim_reduce="bad",
        ).compile()(patch)
