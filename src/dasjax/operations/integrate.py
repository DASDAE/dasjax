"""Integrate patch operation."""

from __future__ import annotations

from dataclasses import dataclass
from operator import mul
from typing import Any, Self

import dascore as dc
from dascore.utils.misc import iterate

from dasjax.core import PatchBoundary, PatchOperation, PatchPyTree

from .. import compat, kernels
from .common import (
    dummy_patch,
    get_data_units_from_dims,
    replace,
    tree_boundary_from_patch,
)


@dataclass(frozen=True)
class Integrate(PatchOperation):
    """Integrate patch data along one or more dimensions."""

    dim: str | tuple[str, ...] | None
    definite: bool = False
    axes: tuple[int, ...] = ()
    dxs_or_spacing: tuple[Any, ...] = ()
    attrs: Any = None
    out_boundary: PatchBoundary | None = None
    out_coords: tuple[Any, ...] | None = None
    out_dtype_codes: tuple[Any, ...] | None = None
    out_dims: tuple[str, ...] | None = None

    def bind(self, boundary: PatchBoundary) -> Self:
        dims = tuple(iterate(self.dim if self.dim is not None else boundary.dims))
        axes = []
        dxs = []
        for dim in dims:
            coord = dummy_patch(boundary).get_coord(dim, require_sorted=True)
            val = coord.step if coord.evenly_sampled else coord.data
            dxs.append(dc.to_float(val))
            axes.append(boundary.axis(dim))
        updates = {"data_units": get_data_units_from_dims(boundary, dims, mul)}
        # A definite integral reduces the dims away, so any coordinate
        # metadata the attributes still describe has to go with them. On
        # DASCore's dev branch the attributes no longer carry coordinates at
        # all, and there is nothing left to clear.
        if self.definite and compat.attrs_carry_coords(boundary.attrs):
            updates["coords"] = {}
        attrs = boundary.attrs.update(**updates)
        if self.definite:
            out_patch = dummy_patch(boundary).integrate(dim=dims, definite=True)
            out_tree, out_boundary = tree_boundary_from_patch(out_patch)
            return replace(
                self,
                axes=tuple(axes),
                dxs_or_spacing=tuple(dxs),
                attrs=attrs,
                out_boundary=out_boundary,
                out_coords=out_tree.coord_values,
                out_dtype_codes=out_tree.coord_dtype_codes,
                out_dims=out_tree.dims,
            )
        return replace(self, axes=tuple(axes), dxs_or_spacing=tuple(dxs), attrs=attrs)

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        data = patch_tree.data
        for axis, spacing in zip(self.axes, self.dxs_or_spacing, strict=True):
            data = kernels.integrate_kernel(
                data, axis=axis, dx_or_spacing=spacing, definite=self.definite
            )
        if self.definite:
            return patch_tree.new(
                data=data,
                coords=self.out_coords,
                dims=self.out_dims,
                coord_dtype_codes=self.out_dtype_codes,
            )
        return patch_tree.new(data=data)

    def update_boundary(self, boundary: PatchBoundary) -> PatchBoundary:
        return (
            self.out_boundary
            if self.definite and self.out_boundary
            else boundary.new(attrs=self.attrs)
        )
