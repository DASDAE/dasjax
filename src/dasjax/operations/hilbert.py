"""Hilbert transform patch operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

import dascore as dc
from dascore.exceptions import ParameterError

from dasjax.core import PatchBoundary, PatchOperation, PatchPyTree

from .. import kernels
from .common import dummy_patch, replace, tree_boundary_from_patch


@dataclass(frozen=True)
class Hilbert(PatchOperation):
    """Compute the analytic signal along one dimension."""

    dim: str
    axis: int | None = None

    def bind(self, boundary: PatchBoundary) -> Self:
        if boundary.coord(self.dim).step is None:
            raise dc.exceptions.CoordError(
                f"Coordinate {self.dim} is not evenly sampled as required by hilbert"
            )
        return replace(self, axis=boundary.axis(self.dim))

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        assert self.axis is not None
        return patch_tree.new(
            data=kernels.hilbert_kernel(patch_tree.data, axis=self.axis)
        )


@dataclass(frozen=True)
class Envelope(Hilbert):
    """Compute the signal envelope along one dimension."""

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        assert self.axis is not None
        return patch_tree.new(
            data=kernels.envelope_kernel(patch_tree.data, axis=self.axis)
        )


@dataclass(frozen=True)
class PhaseWeightedStack(PatchOperation):
    """Apply phase-weighted stacking over one dimension."""

    stack_dim: str
    transform_dim: str | None = None
    power: float = 2.0
    dim_reduce: str = "empty"
    stack_axis: int | None = None
    transform_axis: int | None = None
    out_boundary: PatchBoundary | None = None
    out_coords: tuple[Any, ...] | None = None
    out_dtype_codes: tuple[Any, ...] | None = None
    out_dims: tuple[str, ...] | None = None

    def bind(self, boundary: PatchBoundary) -> Self:
        transform_dim = self.transform_dim
        if transform_dim is None:
            other_dims = tuple(dim for dim in boundary.dims if dim != self.stack_dim)
            if len(other_dims) != 1:
                msg = "transform_dim must be provided for patches that are not 2D."
                raise ParameterError(msg)
            transform_dim = other_dims[0]
        if boundary.coord(transform_dim).step is None:
            raise dc.exceptions.CoordError(
                f"Coordinate {transform_dim} is not evenly sampled as required by hilbert"
            )
        out_patch = dummy_patch(boundary).phase_weighted_stack(
            self.stack_dim,
            transform_dim=transform_dim,
            power=self.power,
            dim_reduce=self.dim_reduce,
        )
        out_tree, out_boundary = tree_boundary_from_patch(out_patch)
        out_coords = out_tree.coord_values
        out_dtype_codes = out_tree.coord_dtype_codes
        if self.dim_reduce != "squeeze" and self.stack_dim in out_boundary.coord_names:
            coord_index = out_boundary.coord_index(self.stack_dim)
            out_boundary = out_boundary.new(
                coord_names=tuple(
                    name for name in out_boundary.coord_names if name != self.stack_dim
                )
            )
            out_coords = tuple(
                value for idx, value in enumerate(out_coords) if idx != coord_index
            )
            out_dtype_codes = tuple(
                value for idx, value in enumerate(out_dtype_codes) if idx != coord_index
            )
        return replace(
            self,
            transform_dim=transform_dim,
            stack_axis=boundary.axis(self.stack_dim),
            transform_axis=boundary.axis(transform_dim),
            out_boundary=out_boundary,
            out_coords=out_coords,
            out_dtype_codes=out_dtype_codes,
            out_dims=out_tree.dims,
        )

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        assert self.stack_axis is not None and self.transform_axis is not None
        data = kernels.phase_weighted_stack_kernel(
            patch_tree.data,
            stack_axis=self.stack_axis,
            transform_axis=self.transform_axis,
            power=self.power,
            squeeze=self.dim_reduce == "squeeze",
        )
        return patch_tree.new(
            data=data,
            coords=self.out_coords,
            coord_dtype_codes=self.out_dtype_codes,
            dims=self.out_dims,
        )

    def update_boundary(self, boundary: PatchBoundary) -> PatchBoundary:
        _ = boundary
        assert self.out_boundary is not None
        return self.out_boundary
