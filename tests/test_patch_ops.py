"""Tests for the unified PatchOp compiled execution path."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

import dascore as dc

from dasjax import JaxPatchPipeline
from dasjax.pipeline import _build_segments
from dasjax.operations import get_operation
from dasjax.operations.patch_ops import PatchOp, patch_to_state_spec


@dataclass(frozen=True)
class _SelectedAxisOp(PatchOp):
    dim: str

    def kernel(self, selected_axis):
        return {"data": np.asarray([selected_axis])}


@dataclass(frozen=True)
class _SelectedCoordOp(PatchOp):
    dim: str | None = None
    time: object | None = None
    distance: object | None = None

    def kernel(self, selected_coord, selected_size, selected_start, selected_stop):
        return {
            "data": np.asarray(
                [
                    len(np.asarray(selected_coord)),
                    selected_size,
                    selected_start,
                    selected_stop,
                ]
            )
        }


@dataclass(frozen=True)
class _SelectedStepOp(PatchOp):
    dim: str

    def kernel(self, selected_step):
        return {"data": np.asarray([selected_step])}


def test_scale_patch_op_uses_signature_inference() -> None:
    """Infer scale inputs directly from the PatchOp kernel signature."""
    patch = dc.get_example_patch()
    op = get_operation("scale").patch_op_cls.prepare(patch, 2.0)
    state, _ = patch_to_state_spec(patch)

    out = op.apply(state)

    assert np.allclose(np.asarray(out.data), patch.data * 2.0)
    assert out.coords.keys() == state.coords.keys()


def test_selected_dim_inference_uses_dim_kwarg() -> None:
    """Resolve selected bindings from an explicit dim kwarg."""
    patch = dc.get_example_patch()
    op = _SelectedAxisOp.prepare(patch, dim="time")
    state, _ = patch_to_state_spec(patch)

    out = op.apply(state)

    assert np.asarray(out.data).item() == patch.get_axis("time")


def test_selected_dim_inference_uses_single_dim_kwarg() -> None:
    """Resolve selected bindings from a single dim-matching kwarg."""
    patch = dc.get_example_patch()
    op = _SelectedCoordOp.prepare(patch, time=(2, 3))
    state, _ = patch_to_state_spec(patch)

    out = op.apply(state)
    time_coord = patch.get_coord("time")

    assert np.asarray(out.data)[0] == len(time_coord)
    assert np.asarray(out.data)[1] == len(time_coord)


def test_selected_dim_inference_rejects_missing_selection() -> None:
    """Reject selected bindings when no dimension can be inferred."""
    patch = dc.get_example_patch()

    with pytest.raises(ValueError, match="no selected dim"):
        _SelectedAxisOp.prepare(patch, dim=None)


def test_selected_dim_inference_rejects_ambiguous_selection() -> None:
    """Reject selected bindings when multiple dimensions match the call."""
    patch = dc.get_example_patch()

    with pytest.raises(ValueError, match="multiple dims"):
        _SelectedCoordOp.prepare(patch, time=1, distance=1)


def test_selected_step_requires_even_sampling() -> None:
    """Require evenly sampled coordinates for selected_step bindings."""
    patch = dc.get_example_patch("wacky_dim_coords_patch")
    dim = next(dim for dim in patch.dims if patch.get_coord(dim).step is None)

    with pytest.raises(ValueError, match="selected_step"):
        _SelectedStepOp.prepare(patch, dim=dim)


def test_flip_patch_op_matches_patch_impl() -> None:
    """Match the flip PatchOp path to the eager patch implementation."""
    patch = dc.get_example_patch()

    out = JaxPatchPipeline().flip("time").compile()(patch)
    expected = get_operation("flip").patch_impl(patch, "time")

    assert np.allclose(out.data, expected.data, equal_nan=True)
    assert out.coords == expected.coords


def test_pad_patch_op_matches_patch_impl() -> None:
    """Match the pad PatchOp path to the eager patch implementation."""
    patch = dc.get_example_patch()

    out = JaxPatchPipeline().pad(time=(2, 3), samples=True).compile()(patch)
    expected = get_operation("pad").patch_impl(patch, time=(2, 3), samples=True)

    assert np.allclose(out.data, expected.data, equal_nan=True)
    assert out.coords == expected.coords


def test_patch_op_segment_chain_matches_eager() -> None:
    """Match eager behavior for a mixed compiled PatchOp chain."""
    patch = dc.get_example_patch()
    pipeline = (
        JaxPatchPipeline()
        .scale(2.0)
        .pad(time=(2, 3), samples=True)
        .flip("time")
    )

    out = pipeline.compile()(patch)
    expected = (
        get_operation("flip").patch_impl(
            get_operation("pad").patch_impl(
                get_operation("scale").patch_impl(patch, 2.0),
                time=(2, 3),
                samples=True,
            ),
            "time",
        )
    )

    assert np.allclose(out.data, expected.data, equal_nan=True)
    assert out.coords == expected.coords


def test_patchop_segment_builder_keeps_pad_with_axis_only_followups() -> None:
    """Keep pad in the same PatchOp segment before simple axis-based ops."""
    resolved_steps = tuple(
        (
            get_operation(step.name),
            step.args,
            step.kwargs,
        )
        for step in JaxPatchPipeline()
        .scale(2.0)
        .pad(time=(2, 3), samples=True)
        .normalize(dim="time")
        .steps
    )

    segments = _build_segments(resolved_steps)

    assert [kind for kind, _ in segments] == ["patchop"]
    assert [item[0].name for item in segments[0][1]] == ["scale", "pad", "normalize"]


def test_detrend_patch_op_matches_patch_impl() -> None:
    """Match the detrend PatchOp path to the eager patch implementation."""
    patch = dc.get_example_patch()

    out = JaxPatchPipeline().detrend(dim="time", type="constant").compile()(patch)
    expected = get_operation("detrend").patch_impl(patch, dim="time", type="constant")

    assert np.allclose(out.data, expected.data, equal_nan=True)
    assert out.coords == expected.coords


def test_standardize_patch_op_matches_patch_impl() -> None:
    """Match the standardize PatchOp path to the eager patch implementation."""
    patch = dc.get_example_patch()

    out = JaxPatchPipeline().standardize(dim="time").compile()(patch)
    expected = get_operation("standardize").patch_impl(patch, dim="time")

    assert np.allclose(out.data, expected.data, equal_nan=True)
    assert out.coords == expected.coords


def test_differentiate_patch_op_matches_patch_impl() -> None:
    """Match the differentiate PatchOp path to the eager patch implementation."""
    patch = dc.get_example_patch()

    out = JaxPatchPipeline().differentiate(dim="time", step=2).compile()(patch)
    expected = get_operation("differentiate").patch_impl(patch, dim="time", step=2)

    assert np.allclose(out.data, expected.data, equal_nan=True, rtol=1e-5, atol=1e-6)
    assert out.coords == expected.coords


def test_integrate_indefinite_patch_op_matches_patch_impl() -> None:
    """Match indefinite integrate through the unified PatchOp runtime."""
    patch = dc.get_example_patch()

    out = JaxPatchPipeline().integrate(dim="time", definite=False).compile()(patch)
    expected = get_operation("integrate").patch_impl(patch, dim="time", definite=False)

    assert np.allclose(out.data, expected.data, equal_nan=True, rtol=1e-5, atol=1e-6)
    assert out.coords == expected.coords


def test_normalize_patch_op_matches_patch_impl() -> None:
    """Match the normalize PatchOp path to the eager patch implementation."""
    patch = dc.get_example_patch()

    out = JaxPatchPipeline().normalize(dim="time", norm="l2").compile()(patch)
    expected = get_operation("normalize").patch_impl(patch, dim="time", norm="l2")

    assert np.allclose(out.data, expected.data, equal_nan=True)
    assert out.coords == expected.coords


def test_roll_patch_op_matches_patch_impl() -> None:
    """Match the roll PatchOp path to the eager patch implementation."""
    patch = dc.get_example_patch()

    out = JaxPatchPipeline().roll(time=5, samples=True).compile()(patch)
    expected = get_operation("roll").patch_impl(patch, time=5, samples=True)

    assert np.allclose(out.data, expected.data, equal_nan=True)
    assert out.coords == expected.coords
