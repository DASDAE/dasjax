"""Mute patch operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

import numpy as np
import dascore as dc
from dascore.exceptions import ParameterError
from dascore.proc.mute import line_mute as dc_line_mute

from dasjax.core import PatchBoundary, PatchOperation, PatchPyTree

from .. import kernels
from .common import dummy_patch, replace


def _ones_patch(boundary: PatchBoundary):
    """Create a metadata-only patch with ones for mask construction."""
    patch = dummy_patch(boundary)
    return patch.update(data=np.ones(patch.shape, dtype=np.float64))


@dataclass(frozen=True)
class LineMute(PatchOperation):
    """Mute data in a region bounded by one or two lines."""

    smooth: Any = None
    invert: bool = False
    relative: bool = True
    kwargs: dict[str, Any] | None = None
    mask: Any = None

    def __init__(self, *, smooth=None, invert=False, relative=True, **kwargs):
        object.__setattr__(self, "smooth", smooth)
        object.__setattr__(self, "invert", bool(invert))
        object.__setattr__(self, "relative", bool(relative))
        object.__setattr__(self, "kwargs", dict(kwargs))
        object.__setattr__(self, "mask", None)

    def bind(self, boundary: PatchBoundary) -> Self:
        mask_patch = dc_line_mute.func(
            _ones_patch(boundary),
            smooth=self.smooth,
            invert=self.invert,
            relative=self.relative,
            **(self.kwargs or {}),
        )
        return replace(self, mask=np.asarray(mask_patch.data))

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        return patch_tree.new(
            data=kernels.apply_mask_kernel(patch_tree.data, self.mask)
        )


@dataclass(frozen=True)
class SlopeMute(PatchOperation):
    """Mute data between two slope boundaries."""

    slopes: Any
    dims: tuple[str, str] = ("distance", "time")
    smooth: Any = None
    invert: bool = False
    mask: Any = None

    def bind(self, boundary: PatchBoundary) -> Self:
        slopes = np.asarray(self.slopes)
        if slopes.shape != (2,):
            msg = "slopes must be a sequence of length 2"
            raise ParameterError(msg)
        if np.any(slopes < 0):
            msg = "slopes must be positive."
            raise ParameterError(msg)
        patch = _ones_patch(boundary)
        coord_x, coord_y = (patch.get_coord(x, require_sorted=True) for x in self.dims)
        origin = (float(dc.to_float(coord_x[0])), float(dc.to_float(coord_y[0])))
        range_x = float(dc.to_float(coord_x[-1] - coord_x[0]))
        dim0_vals = [[origin[0], origin[0] + range_x], [origin[0], origin[0] + range_x]]
        dim1_vals = [[origin[1], None], [origin[1], None]]
        for num, slope in enumerate(slopes):
            if np.isclose(slope, 0):
                dim0_vals[num][1] = origin[0]
                dim1_vals[num][1] = origin[1] + 1
            elif np.isinf(slope):
                dim0_vals[num][1] = origin[0] + 1
                dim1_vals[num][1] = origin[1]
            else:
                dim1_vals[num][1] = origin[1] + range_x / slope
        mask_patch = dc_line_mute.func(
            patch,
            **{
                self.dims[0]: dim0_vals,
                self.dims[1]: dim1_vals,
                "smooth": self.smooth,
                "invert": self.invert,
                "relative": False,
            },
        )
        return replace(self, mask=np.asarray(mask_patch.data))

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        return patch_tree.new(
            data=kernels.apply_mask_kernel(patch_tree.data, self.mask)
        )
