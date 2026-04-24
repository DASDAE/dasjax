"""Correlation patch operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

import dascore as dc
from dascore.proc.correlate import correlate_shift as dc_correlate_shift

from dasjax.core import PatchBoundary, PatchOperation, PatchPyTree

from .. import kernels
from .common import dummy_patch, replace, tree_boundary_from_patch


@dataclass(frozen=True)
class CorrelateShift(PatchOperation):
    """Apply the lag shift after frequency-domain correlation."""

    dim: str
    undo_weighting: bool = True
    axis: int | None = None
    step: float = 1.0
    out_boundary: PatchBoundary | None = None
    out_coords: tuple[Any, ...] | None = None
    out_dtype_codes: tuple[Any, ...] | None = None
    out_dims: tuple[str, ...] | None = None

    def bind(self, boundary: PatchBoundary) -> Self:
        patch = dummy_patch(boundary)
        coord = boundary.coord(self.dim)
        out = dc_correlate_shift.func(
            patch, dim=self.dim, undo_weighting=self.undo_weighting
        )
        out_tree, out_boundary = tree_boundary_from_patch(out)
        return replace(
            self,
            axis=boundary.axis(self.dim),
            step=float(dc.to_float(coord.step)),
            out_boundary=out_boundary,
            out_coords=out_tree.coord_values,
            out_dtype_codes=out_tree.coord_dtype_codes,
            out_dims=out_tree.dims,
        )

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        assert self.axis is not None
        return patch_tree.new(
            data=kernels.correlate_shift_kernel(
                patch_tree.data,
                axis=self.axis,
                step=self.step,
                undo_weighting=self.undo_weighting,
            ),
            coords=self.out_coords,
            coord_dtype_codes=self.out_dtype_codes,
            dims=self.out_dims,
        )

    def update_boundary(self, boundary: PatchBoundary) -> PatchBoundary:
        _ = boundary
        assert self.out_boundary is not None
        return self.out_boundary
