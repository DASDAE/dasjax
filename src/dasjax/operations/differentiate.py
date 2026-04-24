"""Differentiate patch operation."""

from __future__ import annotations

from dataclasses import dataclass
from operator import truediv
from typing import Any, Self

import dascore as dc
from dascore.exceptions import ParameterError
from dascore.utils.misc import iterate

from dasjax.core import PatchBoundary, PatchOperation, PatchPyTree

from .. import kernels
from .common import dummy_patch, get_data_units_from_dims, replace


@dataclass(frozen=True)
class Differentiate(PatchOperation):
    """Differentiate patch data along one or more dimensions."""

    dim: str | tuple[str, ...] | None
    order: int = 2
    step: int = 1
    axes: tuple[int, ...] = ()
    dxs_or_spacing: tuple[Any, ...] = ()
    attrs: Any = None

    def bind(self, boundary: PatchBoundary) -> Self:
        dims = tuple(iterate(self.dim if self.dim is not None else boundary.dims))
        if self.step > 1 and len(dims) > 1:
            raise ParameterError(
                "Step in patch.differentiate can only be used along one axis."
            )
        axes = []
        dxs = []
        for dim in dims:
            coord = dummy_patch(boundary).get_coord(dim, require_sorted=True)
            val = coord.step if coord.evenly_sampled else coord.data
            dxs.append(dc.to_float(val))
            axes.append(boundary.axis(dim))
        attrs = boundary.attrs.update(
            data_units=get_data_units_from_dims(boundary, dims, truediv)
        )
        return replace(self, axes=tuple(axes), dxs_or_spacing=tuple(dxs), attrs=attrs)

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        data = patch_tree.data
        for axis, spacing in zip(self.axes, self.dxs_or_spacing, strict=True):
            data = kernels.differentiate_kernel(
                data, axis=axis, dx_or_spacing=spacing, order=self.order, step=self.step
            )
        return patch_tree.new(data=data)

    def update_boundary(self, boundary: PatchBoundary) -> PatchBoundary:
        return boundary.new(attrs=self.attrs)
