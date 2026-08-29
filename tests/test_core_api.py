"""Tests for the class-authored core operation API."""

from __future__ import annotations

from dataclasses import dataclass, replace
from operator import mul
from types import SimpleNamespace
from typing import ClassVar

import dascore as dc
import jax
import numpy as np
import pytest

from dasjax import JaxPatchPipeline, PatchBoundary, PatchOperation, PatchPyTree
from dasjax.core import (
    _coord_values_match,
    _encode_leaf,
    get_patch_operation,
    list_patch_operations,
)
from dasjax.operations.common import get_data_units_from_dims


@dataclass(frozen=True)
class RenameTimeToFtForTest(PatchOperation):
    """Test operation that statically changes the time dimension name."""

    register = False
    method_name: ClassVar[str | None] = "rename_time_to_ft_for_test"

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        dims = tuple("ft_time" if dim == "time" else dim for dim in patch_tree.dims)
        return patch_tree.new(dims=dims)

    def update_boundary(self, boundary: PatchBoundary) -> PatchBoundary:
        coords = boundary.coords.rename_coord(time="ft_time")
        coord_names = tuple(
            "ft_time" if name == "time" else name for name in boundary.coord_names
        )
        return boundary.new(coords=coords, coord_names=coord_names)


@dataclass(frozen=True)
class AddBoundAxisForTest(PatchOperation):
    """Test operation that must bind against the current boundary."""

    register = False
    method_name: ClassVar[str | None] = "add_bound_axis_for_test"
    dim: str
    axis: int | None = None

    def bind(self, boundary: PatchBoundary):
        return replace(self, axis=boundary.axis(self.dim))

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        assert self.axis is not None
        return patch_tree.new(data=patch_tree.data + self.axis)


@dataclass(frozen=True)
class ResultRenameTimeToFtForTest(PatchOperation):
    """Test operation that only knows its boundary after execution."""

    register = False
    method_name: ClassVar[str | None] = "result_rename_time_to_ft_for_test"

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        dims = tuple("ft_time" if dim == "time" else dim for dim in patch_tree.dims)
        return patch_tree.new(dims=dims)

    def update_boundary_from_result(
        self,
        patch_tree: PatchPyTree,
        boundary: PatchBoundary,
    ) -> PatchBoundary:
        _ = patch_tree
        coords = boundary.coords.rename_coord(time="ft_time")
        coord_names = tuple(
            "ft_time" if name == "time" else name for name in boundary.coord_names
        )
        return boundary.new(coords=coords, coord_names=coord_names)


def test_patch_pytree_roundtrip_restores_patch() -> None:
    """Round-trip a DASCore patch through the new PatchPyTree model."""
    patch = dc.get_example_patch()

    patch_tree, boundary = PatchPyTree.from_patch(patch)
    out = boundary.to_patch(patch_tree)

    assert out.equals(patch)


def test_patch_pytree_flattens_dynamic_coord_values_and_codes() -> None:
    """Expose ordered coord values and dtype codes while keeping only dims static."""
    patch = dc.get_example_patch()
    patch_tree, _ = PatchPyTree.from_patch(patch)

    leaves, tree = jax.tree_util.tree_flatten(patch_tree)
    out = jax.tree_util.tree_unflatten(tree, leaves)
    _, aux_data = patch_tree.tree_flatten()

    assert len(leaves) == 1 + 2 * len(patch.coords.coord_map)
    assert aux_data == {"dims": tuple(patch.dims)}
    assert out.dims == tuple(patch.dims)


def test_patch_pytree_restores_datetime_coords() -> None:
    """Restore datetime coordinates after dynamic JAX conversion."""
    patch = dc.get_example_patch()

    patch_tree, boundary = PatchPyTree.from_patch(patch)
    out = boundary.to_patch(patch_tree)

    assert out.get_coord("time").dtype == patch.get_coord("time").dtype
    assert np.array_equal(out.get_coord("time").values, patch.get_coord("time").values)


def test_coord_values_match_rejects_shape_mismatch() -> None:
    """Cover coordinate equality shape-mismatch rejection."""
    assert not _coord_values_match(np.arange(2), np.arange(3))


def test_encode_leaf_rejects_unsupported_dtype() -> None:
    """Reject coordinate dtypes that cannot be represented in JAX patches."""
    with pytest.raises(TypeError, match="Unsupported coordinate dtype"):
        _encode_leaf(np.asarray(["x"], dtype=object))
    with pytest.raises(TypeError, match="Unsupported coordinate dtype"):
        _encode_leaf(np.asarray([1], dtype=np.uint32))


def test_patch_boundary_swaps_dynamic_coord_values() -> None:
    """Rebuild coords by replacing values while preserving coord metadata."""
    patch = dc.get_example_patch()
    patch_tree, boundary = PatchPyTree.from_patch(patch)
    distance_idx = boundary.coord_names.index("distance")
    coord_values = list(patch_tree.coord_values)
    coord_values[distance_idx] = np.asarray(coord_values[distance_idx]) + 1

    out = boundary.to_patch(patch_tree.new(coords=tuple(coord_values)))

    assert np.array_equal(
        out.get_coord("distance").values,
        patch.get_coord("distance").values + 1,
    )
    assert out.get_coord("distance").units == patch.get_coord("distance").units


def test_patch_boundary_exposes_author_metadata_helpers() -> None:
    """Provide a compact metadata API for operation binding."""
    patch = dc.get_example_patch()
    _, boundary = PatchPyTree.from_patch(patch)

    assert boundary.dims == tuple(patch.dims)
    assert boundary.axis("time") == patch.get_axis("time")
    assert boundary.coord("time") == patch.get_coord("time")
    assert boundary.coord_dims("time") == ("time",)
    assert boundary.coord_index("time") == boundary.coord_names.index("time")


def test_patch_pytree_helpers_and_base_hooks() -> None:
    """Cover direct tree helpers and default operation hook behavior."""
    patch = dc.get_example_patch()
    patch_tree, boundary = PatchPyTree.from_patch(patch)

    with pytest.raises(RuntimeError, match="requires PatchBoundary"):
        patch_tree.to_patch(coerce_numpy=False)

    assert np.array_equal(patch_tree.coord(0), patch_tree.coord_values[0])
    replaced = patch_tree.replace_coord(0, np.asarray(patch_tree.coord_values[0]) + 1)
    assert np.array_equal(replaced[0], np.asarray(patch_tree.coord_values[0]) + 1)
    assert boundary.with_metadata(attrs=patch.attrs).attrs == patch.attrs

    op = PatchOperation()
    assert op.bind(boundary) is op
    assert op.kernel(patch_tree) is patch_tree
    assert op.update_boundary(boundary) is boundary
    assert op.update_boundary_from_result(patch_tree, boundary) is boundary


def test_patch_pytree_new_updates_data_and_ordered_coord_values() -> None:
    """Mirror Patch.new-style immutable updates for JAX patch trees."""
    patch = dc.get_example_patch()
    patch_tree, boundary = PatchPyTree.from_patch(patch)
    distance_idx = boundary.coord_names.index("distance")
    coord_values = list(patch_tree.coord_values)
    coord_values[distance_idx] = np.asarray(coord_values[distance_idx]) + 2

    out_tree = patch_tree.new(data=patch_tree.data + 1, coords=tuple(coord_values))
    out = boundary.to_patch(out_tree)

    assert np.array_equal(out.data, patch.data + 1)
    assert np.array_equal(
        out.get_coord("distance").values,
        patch.get_coord("distance").values + 2,
    )


def test_operation_subclasses_register_pipeline_names() -> None:
    """Register class-authored operations by snake-case class name."""
    assert "scale" in list_patch_operations()
    assert get_patch_operation("scale").operation_name() == "scale"


def test_operation_registration_validation() -> None:
    """Cover duplicate registration, suffix naming, and unknown lookups."""

    @dataclass(frozen=True)
    class ExampleOperation(PatchOperation):
        register = False

    assert ExampleOperation.operation_name() == "example"

    with pytest.raises(ValueError, match="Duplicate"):

        class Scale(PatchOperation):
            pass

    with pytest.raises(AttributeError, match="Unknown"):
        get_patch_operation("does_not_exist")


def test_unit_helper_handles_missing_and_multiple_units() -> None:
    """Cover unit helper paths for no units, skipped dims, and multiplication."""
    patch = dc.get_example_patch()
    _, boundary = PatchPyTree.from_patch(patch)
    attrs = boundary.attrs.update(data_units="m")
    coords = boundary.coords.set_units(time="s")
    unit_boundary = boundary.new(attrs=attrs, coords=coords)

    assert get_data_units_from_dims(boundary, ("time",), mul) is None
    assert (
        get_data_units_from_dims(unit_boundary, ("time", "distance"), mul) is not None
    )

    fake_boundary = SimpleNamespace(
        attrs=SimpleNamespace(data_units="m"),
        coord=lambda name: SimpleNamespace(units=None if name == "x" else "s"),
    )
    assert get_data_units_from_dims(fake_boundary, ("x", "time"), mul) is not None


def test_compiled_pipeline_uses_class_authored_basic_operations() -> None:
    """Run a fully ported class-authored chain through JaxPatchPipeline."""
    patch = dc.get_example_patch()
    pipeline = JaxPatchPipeline().scale(2.0).add(1.0).clip(-1.0, 1.0)

    out = pipeline.compile()(patch)
    expected = patch.update(data=np.clip(patch.data * 2.0 + 1.0, -1.0, 1.0))

    assert np.allclose(out.data, expected.data)
    assert out.coords == expected.coords


def test_core_planner_binds_against_static_boundary_updates() -> None:
    """Bind later ops against planned boundaries before running a JIT segment."""
    patch = dc.get_example_patch()
    pipeline = JaxPatchPipeline().dft(dim="time", real=True).normalize("ft_time")

    out = pipeline.compile()(patch)

    assert out.dims == ("distance", "ft_time")


def test_core_planner_keeps_static_boundary_updates_in_one_segment() -> None:
    """Static boundary changes should not force a core segment boundary."""
    patch = dc.get_example_patch()
    _, boundary = PatchPyTree.from_patch(patch)
    operations = (
        RenameTimeToFtForTest(),
        AddBoundAxisForTest("ft_time"),
    )

    segments = JaxPatchPipeline._plan_core_segments(operations, boundary)

    assert len(segments) == 1
    bound_ops, planned_boundary, needs_result_boundary = segments[0]
    assert len(bound_ops) == 2
    assert planned_boundary.dims == ("distance", "ft_time")
    assert not needs_result_boundary
    assert bound_ops[1].axis == 1


def test_pipeline_plan_reports_core_fused_segments() -> None:
    """Expose the planned core execution path without running kernels."""
    patch = dc.get_example_patch()
    pipeline = JaxPatchPipeline().dft(dim="time", real=True).normalize("ft_time")

    plan = pipeline.plan(patch)

    assert plan.uses_core
    assert plan.fused_kernel_count == 1
    assert len(plan.segments) == 1
    segment = plan.segments[0]
    assert segment.kind == "core"
    assert segment.operations == ("dft", "normalize")
    assert segment.fused
    assert segment.jitted
    assert segment.input_dims == ("distance", "time")
    assert segment.output_dims == ("distance", "ft_time")


def test_core_planner_restarts_after_result_boundary_updates() -> None:
    """Result-dependent boundary changes stop planning until after execution."""
    patch = dc.get_example_patch()
    patch_tree, boundary = PatchPyTree.from_patch(patch)
    op = ResultRenameTimeToFtForTest()
    bound_ops, planned_boundary, needs_result_boundary, op_index = (
        JaxPatchPipeline._plan_core_segment((op,), boundary, 0)
    )
    out_tree = bound_ops[0].kernel(patch_tree)
    out_boundary = bound_ops[0].update_boundary_from_result(out_tree, planned_boundary)
    out = out_boundary.to_patch(out_tree)

    assert out.dims == ("distance", "ft_time")
    assert needs_result_boundary
    assert op_index == 1


def test_pipeline_plan_reports_unplanned_result_boundary_tail() -> None:
    """Show where planning must pause for result-dependent metadata."""
    patch = dc.get_example_patch()
    _, boundary = PatchPyTree.from_patch(patch)

    segments = JaxPatchPipeline._plan_core_segments(
        (ResultRenameTimeToFtForTest(), AddBoundAxisForTest("ft_time")),
        boundary,
    )

    assert len(segments) == 1
    assert segments[0][2]


def test_pipeline_plan_reports_core_segments_for_all_operations() -> None:
    """Expose core execution segments for formerly registry-backed operations."""
    patch = dc.get_example_patch()
    pipeline = JaxPatchPipeline().detrend("time").normalize("time")

    plan = pipeline.plan(patch)

    assert plan.uses_core
    assert plan.fused_kernel_count == 1
    assert len(plan.segments) == 1
    assert plan.segments[0].kind == "core"
    assert plan.segments[0].operations == ("detrend", "normalize")
    assert plan.segments[0].fused
    assert plan.segments[0].jitted


def test_metadata_hook_uses_dascore_native_objects() -> None:
    """Pass native DASCore attrs and coord manager into metadata hooks."""

    @dataclass(frozen=True)
    class _MetadataProbe(PatchOperation):
        register = False

        def update_boundary(self, boundary):
            assert boundary.attrs.__class__.__name__ == "PatchAttrs"
            assert hasattr(boundary.coords, "coord_map")
            return boundary

    patch = dc.get_example_patch()
    patch_tree, boundary = PatchPyTree.from_patch(patch)
    metadata_op = _MetadataProbe()

    assert type(metadata_op).overrides("update_boundary")
    out = metadata_op.update_boundary(boundary).to_patch(metadata_op.kernel(patch_tree))

    assert out.equals(patch)
