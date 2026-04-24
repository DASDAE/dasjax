"""Tests for compiled pipeline caching behavior."""

from __future__ import annotations

import dascore as dc
import jax
import numpy as np
import pytest

from dasjax import JaxPatchPipeline


def _fbe_baseline(patch):
    out = patch.stft(time=64, samples=True, overlap=32).abs()
    ft_dim = next(dim for dim in out.dims if dim.startswith("ft_"))
    out = out.select(**{ft_dim: (2.0, 10.0)})
    return out.sum(dim=ft_dim, dim_reduce="squeeze")


def test_compile_reuses_cached_callable_for_same_pipeline() -> None:
    """Reuse the compiled callable for repeated compile calls."""
    pipeline = JaxPatchPipeline().scale(2.0).add(1.0)

    compiled_1 = pipeline.compile()
    compiled_2 = pipeline.compile()

    assert compiled_1 is compiled_2


def test_compile_reuses_cached_callable_for_equivalent_pipeline_instances() -> None:
    """Reuse the compiled callable across equivalent pipeline instances."""
    pipeline_1 = JaxPatchPipeline().scale(2.0).add(1.0)
    pipeline_2 = JaxPatchPipeline().scale(2.0).add(1.0)

    compiled_1 = pipeline_1.compile()
    compiled_2 = pipeline_2.compile()

    assert compiled_1 is compiled_2


def test_compile_separates_distinct_pipeline_definitions() -> None:
    """Keep distinct pipeline definitions in separate cache entries."""
    pipeline_1 = JaxPatchPipeline().scale(2.0)
    pipeline_2 = JaxPatchPipeline().scale(3.0)

    assert pipeline_1.compile() is not pipeline_2.compile()


def test_compile_separates_assert_no_fallback_flag() -> None:
    """Separate cache entries when assert_no_fallback changes."""
    pipeline = JaxPatchPipeline().scale(2.0)

    compiled_default = pipeline.compile()
    compiled_checked = pipeline.compile(assert_no_fallback=True)

    assert compiled_default is not compiled_checked


def test_compile_rejects_device_and_backend_together() -> None:
    """Reject compile calls that pass both device and backend."""
    pipeline = JaxPatchPipeline().scale(2.0)

    with pytest.raises(ValueError, match="either device or backend"):
        pipeline.compile(device=jax.devices()[0], backend="cpu")


def test_assert_no_fallback_passes_for_native_pipeline() -> None:
    """Allow fallback assertions for fully native compiled pipelines."""
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


def test_assert_no_fallback_accepts_core_pipeline() -> None:
    """Core pipelines no longer have legacy fallback branches."""
    patch = dc.get_example_patch("chirp")
    pipeline = JaxPatchPipeline().gaussian_filter(**{patch.dims[-1]: 3}, samples=True)

    pipeline.assert_no_fallback(patch)


def test_assert_no_fallback_passes_for_fbe_pipeline() -> None:
    """Allow fallback assertions for pure-JAX FBE pipelines."""
    patch = dc.get_example_patch("chirp")
    pipeline = JaxPatchPipeline().fbe(
        time=64, samples=True, overlap=32, fmin=2.0, fmax=10.0
    )
    # fbe uses pure JAX (no host callbacks) — assert_no_fallback should not raise.
    pipeline.assert_no_fallback(patch)


def test_cached_pipeline_callable_still_matches_expected_output() -> None:
    """Preserve eager-equivalent results when serving cached callables."""
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
    expected = ((patch * 2.0) + 1.0).detrend(dim=patch.dims[-1], type="constant")

    assert out.equals(expected)


def test_compiled_pipeline_with_fbe_matches_eager() -> None:
    """Match eager behavior for compiled pipelines containing FBE."""
    patch = dc.get_example_patch("chirp")
    pipeline = (
        JaxPatchPipeline()
        .scale(2.0)
        .fbe(time=64, samples=True, overlap=32, fmin=2.0, fmax=10.0)
    )
    out = pipeline.compile()(patch)
    expected = _fbe_baseline(patch * 2.0)
    assert np.allclose(out.data, expected.data, equal_nan=True, rtol=1e-5, atol=1e-6)
    assert out.coords == expected.coords


def test_compiled_pipeline_with_fbe_in_middle() -> None:
    """Match eager behavior when FBE appears mid-pipeline."""
    patch = dc.get_example_patch("chirp")
    pipeline = (
        JaxPatchPipeline()
        .scale(2.0)
        .fbe(time=64, samples=True, overlap=32, fmin=2.0, fmax=10.0)
        .abs()
    )
    out = pipeline.compile()(patch)
    expected = _fbe_baseline(patch * 2.0).abs()
    assert np.allclose(out.data, expected.data, equal_nan=True, rtol=1e-5, atol=1e-6)
    assert out.coords == expected.coords


def test_compile_accepts_explicit_device() -> None:
    """Accept explicit device selection during compilation."""
    patch = dc.get_example_patch("chirp")
    pipeline = JaxPatchPipeline().scale(2.0)

    compiled = pipeline.compile(device=jax.devices()[0])
    out = compiled(patch)

    assert out.equals(patch * 2.0)


def test_compile_cache_is_bounded() -> None:
    """Evict older entries when the compiled cache exceeds its bound."""
    cache = JaxPatchPipeline._compiled_cache
    maxsize = cache.maxsize

    for idx in range(maxsize + 10):
        JaxPatchPipeline().scale(float(idx)).compile()

    assert len(cache) <= maxsize
