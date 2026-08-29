"""Taper patch operations."""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from operator import add
from typing import Any, Self

import dascore as dc
import numpy as np
from dascore.exceptions import ParameterError
from dasjax.compat import taper_ramp, taper_range_ramp
from dasjax.core import PatchBoundary, PatchOperation, PatchPyTree

from .. import kernels
from .common import dummy_patch, replace


def _taper_slices(boundary: PatchBoundary, kwargs: dict[str, Any]):
    dim = next(key for key in kwargs if key in boundary.dims)
    axis = boundary.axis(dim)
    value = kwargs[dim]
    coord = boundary.coord(dim)
    if isinstance(value, (tuple, list, np.ndarray)):
        start, stop = value
    else:
        start, stop = value, value
    dur = coord.coord_range(extend=False)
    clses = (dc.units.Quantity, np.timedelta64)
    start = start if isinstance(start, clses) or start is None else start * dur
    stop = stop if isinstance(stop, clses) or stop is None else stop * dur
    stop = -stop if stop is not None else stop
    _, inds_1 = coord.select((None, start), relative=True)
    _, inds_2 = coord.select((stop, None), relative=True)
    return axis, inds_1, inds_2


@dataclass(frozen=True)
class Taper(PatchOperation):
    """Apply edge tapers to patch data along one dimension."""

    window_type: str = "hann"
    kwargs: dict[str, Any] | None = None
    axis: int | None = None
    weight: Any = None

    def __init__(self, window_type: str = "hann", **kwargs):
        object.__setattr__(self, "window_type", window_type)
        object.__setattr__(self, "kwargs", dict(kwargs))
        object.__setattr__(self, "axis", None)
        object.__setattr__(self, "weight", None)

    def bind(self, boundary: PatchBoundary) -> Self:
        axis, start_slice, end_slice = _taper_slices(boundary, self.kwargs or {})
        length = boundary.coords.shape[axis]
        start_len = start_slice.stop
        end_len = length - end_slice.start
        if (
            start_len is not None
            and end_len is not None
            and start_len > end_slice.start
        ):
            raise ParameterError("Taper windows cannot overlap")
        weight = np.ones(length, dtype=np.float64)
        if start_len is not None and start_len > 0:
            weight[:start_len] = taper_ramp(self.window_type, start_len)
        if end_slice.start is not None and end_slice.start < length:
            weight[end_slice.start :] = taper_ramp(self.window_type, end_len)[::-1]
        return replace(self, axis=axis, weight=weight)

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        assert self.axis is not None
        return patch_tree.new(
            data=kernels.apply_1d_weight_kernel(patch_tree.data, self.axis, self.weight)
        )


def _taper_coord_inds(coord, values, relative, samples):
    error_msg = "A len 2 or 4 sequence is required for taper values"
    if not isinstance(values, (tuple, list, np.ndarray)) or not len(values):
        raise ParameterError(error_msg)
    if isinstance(values[0], (tuple, list, np.ndarray)):
        return reduce(
            add, (_taper_coord_inds(coord, item, relative, samples) for item in values)
        )
    if len(values) not in {2, 4}:
        raise ParameterError(error_msg)
    out = [None] * len(values)
    for idx, value in enumerate(values):
        if value is None or value == ...:
            if len(values) == 2:
                raise ParameterError(
                    "Cannot use ... or None when only two values provided"
                )
            out[idx] = 0 if (idx / len(out)) < 0.5 else len(coord) - 1
        else:
            out[idx] = coord.get_next_index(value, samples=samples, relative=relative)
    return [[0, *out, len(coord)]] if len(out) == 2 else [out]


def _taper_curve(coord, ind_1, ind_2, window_type, reverse=False):
    taper = taper_range_ramp(window_type, ind_2 - ind_1)
    return taper[::-1] if reverse else taper


@dataclass(frozen=True)
class TaperRange(PatchOperation):
    """Apply tapers around selected coordinate ranges."""

    window_type: str = "hann"
    invert: bool = False
    relative: bool = False
    samples: bool = False
    kwargs: dict[str, Any] | None = None
    axis: int | None = None
    weight: Any = None

    def __init__(
        self,
        window_type: str = "hann",
        invert=False,
        relative=False,
        samples=False,
        **kwargs,
    ):
        object.__setattr__(self, "window_type", window_type)
        object.__setattr__(self, "invert", invert)
        object.__setattr__(self, "relative", relative)
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "kwargs", dict(kwargs))
        object.__setattr__(self, "axis", None)
        object.__setattr__(self, "weight", None)

    def bind(self, boundary: PatchBoundary) -> Self:
        kwargs = self.kwargs or {}
        dim = next(key for key in kwargs if key in boundary.dims)
        axis = boundary.axis(dim)
        coord = dummy_patch(boundary).get_coord(dim, require_sorted=True)
        inds = _taper_coord_inds(coord, kwargs[dim], self.relative, self.samples)
        weight = np.zeros(len(coord))
        for i1, i2, i3, i4 in inds:
            weight[i1:i2] += _taper_curve(coord, i1, i2, self.window_type)
            weight[i3:i4] += _taper_curve(coord, i3, i4, self.window_type, reverse=True)
            weight[i2:i3] += 1
        if self.invert:
            weight = np.abs(weight - np.max(weight))
        return replace(self, axis=axis, weight=np.asarray(weight))

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        assert self.axis is not None
        return patch_tree.new(
            data=kernels.apply_1d_weight_kernel(patch_tree.data, self.axis, self.weight)
        )
