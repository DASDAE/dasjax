"""Tests for compiled pipeline caching behavior."""

from __future__ import annotations

from dataclasses import dataclass

import dascore as dc
import jax
import numpy as np
import pytest

from dasjax import JaxPatchPipeline, PatchBoundary, PatchOperation, PatchPyTree


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


def test_compile_rejects_device_and_backend_together() -> None:
    """Reject compile calls that pass both device and backend."""
    pipeline = JaxPatchPipeline().scale(2.0)

    with pytest.raises(ValueError, match="either device or backend"):
        pipeline.compile(device=jax.devices()[0], backend="cpu")


def test_pipeline_private_freezing_and_plan_edges() -> None:
    """Cover cache-key freezing for unhashable values and plan tail reporting."""
    patch = dc.get_example_patch()
    pipeline = JaxPatchPipeline().scale(np.asarray([2.0])).add({"value": 1.0})
    key = pipeline._compile_cache_key()
    set_key = JaxPatchPipeline().add({"a", "b"})._compile_cache_key()

    assert "ndarray" in repr(key)
    assert "(2.0,)" not in repr(key)
    assert "value" in repr(key)
    assert "a" in repr(set_key)
    assert JaxPatchPipeline().steps == ()

    class PlainOperation:
        value = []

        @classmethod
        def operation_name(cls):
            return "plain"

    from dasjax.pipeline import _operation_key

    assert _operation_key(PlainOperation())[0] == "plain"
    with pytest.raises(AttributeError, match="Unknown"):
        JaxPatchPipeline().not_registered

    plan = (
        JaxPatchPipeline().dft(dim="time", real=True).normalize("ft_time").plan(patch)
    )
    assert plan.fused_kernel_count == 1


@dataclass(frozen=True)
class _TailResultBoundary(PatchOperation):
    register = False

    def update_boundary_from_result(
        self,
        patch_tree: PatchPyTree,
        boundary: PatchBoundary,
    ) -> PatchBoundary:
        _ = patch_tree
        return boundary


def test_pipeline_plan_reports_unplanned_tail_from_execution_planner() -> None:
    """Cover the unplanned tail path in the public execution planner."""
    patch = dc.get_example_patch()
    tail_plan = JaxPatchPipeline()._plan_core_execution(
        patch,
        (_TailResultBoundary(), _TailResultBoundary()),
    )
    assert tail_plan.segments[-1].kind == "unplanned"


def test_pipeline_backend_and_result_boundary_compile_paths(monkeypatch) -> None:
    """Cover backend resolution and result-dependent execution hooks."""
    patch = dc.get_example_patch()

    with pytest.raises(RuntimeError):
        JaxPatchPipeline().scale(1.0).compile(backend="not-a-backend")
    out = JaxPatchPipeline().scale(1.0).compile(backend="cpu")(patch)
    assert out.equals(patch)
    monkeypatch.setattr("dasjax.pipeline.jax.local_devices", lambda backend: [])
    with pytest.raises(ValueError, match="No local JAX devices"):
        JaxPatchPipeline().scale(1.0).compile(backend="cpu")

    @dataclass(frozen=True)
    class _ResultBoundaryForCoverage(PatchOperation):
        register = False

        def update_boundary_from_result(
            self,
            patch_tree: PatchPyTree,
            boundary: PatchBoundary,
        ) -> PatchBoundary:
            _ = patch_tree
            return boundary

    PatchOperation._registry["result_boundary_for_coverage"] = (
        _ResultBoundaryForCoverage
    )
    try:
        out = JaxPatchPipeline().result_boundary_for_coverage().compile()(patch)
    finally:
        PatchOperation._registry.pop("result_boundary_for_coverage")
    assert out.equals(patch)


def test_compiled_pipeline_reuses_bound_plan_for_same_boundary() -> None:
    """Avoid rebinding operations on repeated calls with the same metadata."""
    patch = dc.get_example_patch()
    bind_count = {"value": 0}

    @dataclass(frozen=True)
    class _CountingBind(PatchOperation):
        register = False

        def bind(self, boundary: PatchBoundary):
            bind_count["value"] += 1
            _ = boundary
            return self

        def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
            return patch_tree

    PatchOperation._registry["counting_bind_for_cache"] = _CountingBind
    try:
        compiled = JaxPatchPipeline().counting_bind_for_cache().compile()
        compiled(patch)
        compiled(patch)
    finally:
        PatchOperation._registry.pop("counting_bind_for_cache")

    assert bind_count["value"] == 1


def test_native_hampel_pipeline_compiles() -> None:
    """Compile native approximate Hampel filtering."""
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

    out = pipeline.compile()(patch)

    assert out.shape == patch.shape


def test_gaussian_filter_pipeline_compiles() -> None:
    """Compile native Gaussian filtering."""
    patch = dc.get_example_patch("chirp")
    dim = patch.dims[-1]
    pipeline = JaxPatchPipeline().gaussian_filter(**{dim: 3}, samples=True)

    out = pipeline.compile()(patch)

    assert np.allclose(out.data, patch.gaussian_filter(**{dim: 3}, samples=True).data)


def test_exact_hampel_filter_is_not_implemented() -> None:
    """Reject exact Hampel filtering because it is not implemented in pure JAX."""
    patch = dc.get_example_patch("chirp")

    with pytest.raises(NotImplementedError, match="pure JAX"):
        JaxPatchPipeline().hampel_filter(
            **{patch.dims[-1]: 3},
            samples=True,
            approximate=False,
        ).compile()


def test_whiten_pipeline_compiles() -> None:
    """Compile native whitening."""
    patch = dc.get_example_patch("chirp")
    pipeline = JaxPatchPipeline().dft(dim=patch.dims[-1], real=True).whiten()

    out = pipeline.compile()(patch)
    expected = patch.dft(dim=patch.dims[-1], real=True).whiten()

    assert np.allclose(out.data, expected.data)


def test_fbe_pipeline_compiles() -> None:
    """Compile native FBE pipelines."""
    patch = dc.get_example_patch("chirp")
    pipeline = JaxPatchPipeline().fbe(
        time=64, samples=True, overlap=32, fmin=2.0, fmax=10.0
    )
    out = pipeline.compile()(patch)

    assert np.allclose(out.data, _fbe_baseline(patch).data)


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
