"""Parity tests for additional numeric patch operations."""

from __future__ import annotations

import dascore as dc
import numpy as np
import pytest

from dasjax import JaxPatchPipeline

from .helpers import assert_patch_close


ELEMENTWISE_CASES = (
    ("subtract", (1.5,), {}, lambda p: p.subtract(1.5)),
    ("multiply", (2.0,), {}, lambda p: p.multiply(2.0)),
    ("divide", (2.0,), {}, lambda p: p.divide(2.0)),
    ("maximum", (0.0,), {}, lambda p: p.maximum(0.0)),
    ("minimum", (0.0,), {}, lambda p: p.minimum(0.0)),
    ("exp", (), {}, lambda p: p.exp()),
    ("log", (), {}, lambda p: p.log()),
    ("log10", (), {}, lambda p: p.log10()),
    ("log2", (), {}, lambda p: p.log2()),
)


@pytest.mark.parametrize("name,args,kwargs,baseline", ELEMENTWISE_CASES)
def test_compiled_elementwise_numeric_operations_match_dascore(
    name,
    args,
    kwargs,
    baseline,
) -> None:
    """Match compiled elementwise numeric operations against DASCore."""
    patch = dc.get_example_patch()
    positive = patch.update(data=np.asarray(patch.data) + 1.0)
    pipeline = getattr(JaxPatchPipeline(), name)(*args, **kwargs)

    assert_patch_close(pipeline.compile()(positive), baseline(positive))


def test_compiled_null_and_boolean_numeric_operations_match_dascore() -> None:
    """Cover fill/mask numeric methods with NaN and infinite values."""
    base = dc.get_example_patch()
    data = np.asarray(base.data).copy()
    data[0, :3] = [0.0, np.nan, np.inf]
    patch = base.update(data=data)
    cond = np.ones(patch.shape, dtype=bool)
    cond[0, 1] = False

    cases = (
        (JaxPatchPipeline().fillna(-1.0), patch.fillna(-1.0)),
        (JaxPatchPipeline().is_finite(), patch.is_finite()),
        (JaxPatchPipeline().isinf(), patch.isinf()),
        (JaxPatchPipeline().isnan(), patch.isnan()),
        (
            JaxPatchPipeline().where(cond, other=-2.0),
            patch.where(cond, other=-2.0),
        ),
    )
    for pipeline, expected in cases:
        assert_patch_close(pipeline.compile()(patch), expected)


@pytest.mark.parametrize(
    "name",
    ["all", "any", "max", "mean", "median", "min", "std", "sum"],
)
def test_compiled_reductions_match_dascore(name: str) -> None:
    """Match named reductions along a dimension."""
    patch = dc.get_example_patch()
    if name in {"all", "any"}:
        patch = patch > 0.5
    pipeline = getattr(JaxPatchPipeline(), name)(dim="time")
    expected = getattr(patch, name)(dim="time")

    assert_patch_close(pipeline.compile()(patch), expected)


def test_compiled_aggregate_and_squeeze_reduction_match_dascore() -> None:
    """Cover aggregate and squeezed dimension metadata."""
    patch = dc.get_example_patch()

    assert_patch_close(
        JaxPatchPipeline().aggregate(dim="time", method="mean").compile()(patch),
        patch.aggregate(dim="time", method="mean"),
    )
    assert_patch_close(
        JaxPatchPipeline().sum(dim="time", dim_reduce="squeeze").compile()(patch),
        patch.sum(dim="time", dim_reduce="squeeze"),
    )


def test_compiled_callback_numeric_transforms_match_dascore() -> None:
    """Cover heavier numeric transforms which delegate to DASCore callbacks."""
    patch = dc.get_example_patch()
    interp_coord = patch.get_coord("time").values[::4]
    cases = (
        (
            JaxPatchPipeline().correlate(distance=10, samples=True),
            patch.correlate(distance=10, samples=True),
        ),
        (
            JaxPatchPipeline().decimate(time=2, filter_type=None),
            patch.decimate(time=2, filter_type=None),
        ),
        (
            JaxPatchPipeline().interpolate(time=interp_coord),
            patch.interpolate(time=interp_coord),
        ),
        (
            JaxPatchPipeline().resample(time=100, samples=True),
            patch.resample(time=100, samples=True),
        ),
        (
            JaxPatchPipeline().stft(time=64, samples=True, overlap=32),
            patch.stft(time=64, samples=True, overlap=32),
        ),
    )
    for pipeline, expected in cases:
        assert_patch_close(pipeline.compile()(patch), expected)


def test_compiled_istft_matches_dascore() -> None:
    """Cover inverse short-time Fourier transform callback."""
    patch = dc.get_example_patch().stft(time=64, samples=True, overlap=32)

    assert_patch_close(JaxPatchPipeline().istft().compile()(patch), patch.istft())
