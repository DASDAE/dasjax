"""Tests for compiled pipeline caching behavior."""

from __future__ import annotations

import dascore as dc
import jax
import numpy as np
import pytest

from dasjax import JaxPatchPipeline
from dasjax.operations import get_operation


def _run_operation(patch, name: str, *args, **kwargs):
    """Apply one internal operation through DASCore's pipe helper."""
    return patch.pipe(get_operation(name).patch_impl, *args, **kwargs)


def test_compile_reuses_cached_callable_for_same_pipeline() -> None:
    pipeline = JaxPatchPipeline().scale(2.0).add(1.0)

    compiled_1 = pipeline.compile()
    compiled_2 = pipeline.compile()

    assert compiled_1 is compiled_2


def test_compile_reuses_cached_callable_for_equivalent_pipeline_instances() -> None:
    pipeline_1 = JaxPatchPipeline().scale(2.0).add(1.0)
    pipeline_2 = JaxPatchPipeline().scale(2.0).add(1.0)

    compiled_1 = pipeline_1.compile()
    compiled_2 = pipeline_2.compile()

    assert compiled_1 is compiled_2


def test_compile_separates_distinct_pipeline_definitions() -> None:
    pipeline_1 = JaxPatchPipeline().scale(2.0)
    pipeline_2 = JaxPatchPipeline().scale(3.0)

    assert pipeline_1.compile() is not pipeline_2.compile()


def test_compile_separates_assert_no_fallback_flag() -> None:
    pipeline = JaxPatchPipeline().scale(2.0)

    compiled_default = pipeline.compile()
    compiled_checked = pipeline.compile(assert_no_fallback=True)

    assert compiled_default is not compiled_checked


def test_compile_rejects_device_and_backend_together() -> None:
    pipeline = JaxPatchPipeline().scale(2.0)

    with pytest.raises(ValueError, match="either device or backend"):
        pipeline.compile(device=jax.devices()[0], backend="cpu")


def test_assert_no_fallback_passes_for_native_pipeline() -> None:
    patch = dc.get_example_patch("chirp")
    pipeline = (
        JaxPatchPipeline()
        .scale(2.0)
        .hampel_filter(
            **{patch.dims[-1]: 3},
            samples=True,
            threshold=3.5,
            approximate=True,
        )
    )

    pipeline.assert_no_fallback(patch)


def test_assert_no_fallback_rejects_host_callback_pipeline() -> None:
    patch = dc.get_example_patch("chirp")
    pipeline = JaxPatchPipeline().gaussian_filter(**{patch.dims[-1]: 3}, samples=True)

    with pytest.raises(AssertionError, match="host fallbacks"):
        pipeline.assert_no_fallback(patch)


def test_assert_no_fallback_passes_for_fbe_pipeline() -> None:
    patch = dc.get_example_patch("chirp")
    pipeline = JaxPatchPipeline().fbe(
        time=64, samples=True, overlap=32, fmin=2.0, fmax=10.0
    )
    # fbe uses pure JAX (no host callbacks) — assert_no_fallback should not raise.
    pipeline.assert_no_fallback(patch)


def test_cached_pipeline_callable_still_matches_expected_output() -> None:
    patch = dc.get_example_patch("chirp")
    pipeline = (
        JaxPatchPipeline()
        .scale(2.0)
        .add(1.0)
        .detrend(
            dim=patch.dims[-1],
            type="constant",
        )
    )

    compiled = pipeline.compile()
    out = compiled(patch)
    expected = _run_operation(
        _run_operation(
            _run_operation(
                patch,
                "scale",
                2.0,
            ),
            "add",
            1.0,
        ),
        "detrend",
        dim=patch.dims[-1],
        type="constant",
    )

    assert out.equals(expected)


def test_compiled_pipeline_with_fbe_matches_eager() -> None:
    patch = dc.get_example_patch("chirp")
    pipeline = (
        JaxPatchPipeline()
        .scale(2.0)
        .fbe(time=64, samples=True, overlap=32, fmin=2.0, fmax=10.0)
    )
    out = pipeline.compile()(patch)
    expected = _run_operation(
        _run_operation(patch, "scale", 2.0),
        "fbe",
        time=64,
        samples=True,
        overlap=32,
        fmin=2.0,
        fmax=10.0,
    )
    assert np.allclose(out.data, expected.data, equal_nan=True, rtol=1e-5, atol=1e-6)
    assert out.coords == expected.coords


def test_compiled_pipeline_with_fbe_in_middle() -> None:
    patch = dc.get_example_patch("chirp")
    pipeline = (
        JaxPatchPipeline()
        .scale(2.0)
        .fbe(time=64, samples=True, overlap=32, fmin=2.0, fmax=10.0)
        .abs()
    )
    out = pipeline.compile()(patch)
    expected = _run_operation(
        _run_operation(
            _run_operation(patch, "scale", 2.0),
            "fbe",
            time=64,
            samples=True,
            overlap=32,
            fmin=2.0,
            fmax=10.0,
        ),
        "abs",
    )
    assert np.allclose(out.data, expected.data, equal_nan=True, rtol=1e-5, atol=1e-6)
    assert out.coords == expected.coords


def test_compile_accepts_explicit_device() -> None:
    patch = dc.get_example_patch("chirp")
    pipeline = JaxPatchPipeline().scale(2.0)

    compiled = pipeline.compile(device=jax.devices()[0])
    out = compiled(patch)

    assert out.equals(_run_operation(patch, "scale", 2.0))
