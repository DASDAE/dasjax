"""Whiten patch operation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

import dascore as dc
from dascore.transform.fourier import dft as dc_dft
from dascore.utils.transformatter import FourierTransformatter
import jax.numpy as jnp
from scipy.fft import next_fast_len

from dasjax.core import PatchBoundary, PatchOperation, PatchPyTree

from .. import kernels
from .common import dummy_patch, replace


@dataclass(frozen=True)
class Whiten(PatchOperation):
    """Whiten patch data along a frequency-domain dimension."""

    smooth_size: float | None = None
    water_level: float | None = None
    kwargs: dict[str, Any] | None = None
    axis: int | None = None
    window_len: int | None = None
    pad_after: int = 0
    sample_step: float = 1.0
    input_is_fft: bool = False

    def __init__(self, smooth_size=None, water_level=None, **kwargs):
        object.__setattr__(self, "smooth_size", smooth_size)
        object.__setattr__(self, "water_level", water_level)
        object.__setattr__(self, "kwargs", dict(kwargs))
        object.__setattr__(self, "axis", None)
        object.__setattr__(self, "window_len", None)
        object.__setattr__(self, "pad_after", 0)
        object.__setattr__(self, "sample_step", 1.0)
        object.__setattr__(self, "input_is_fft", False)

    def bind(self, boundary: PatchBoundary) -> Self:
        kwargs = self.kwargs or {}
        dim = next(iter(kwargs), "time" if "time" in boundary.dims else "time")
        fft_dim = FourierTransformatter().rename_dims(dim)[0]
        input_is_fft = dim not in boundary.dims and fft_dim in boundary.dims
        axis_dim = fft_dim if input_is_fft else dim
        axis = boundary.axis(axis_dim)
        sample_step = (
            1.0
            if input_is_fft
            else abs(float(dc.to_float(boundary.coord(dim).step)))
        )
        axis_len = len(boundary.coord(axis_dim))
        pad_after = 0 if input_is_fft else next_fast_len(axis_len) - axis_len
        window_len = None
        if self.smooth_size is not None:
            coord = boundary.coord(fft_dim) if input_is_fft else None
            if coord is None:
                fft_patch = dc_dft.func(dummy_patch(boundary), dim=dim, real=True)
                coord = fft_patch.get_coord(fft_dim)
            window_len = coord.get_sample_count(self.smooth_size, enforce_lt_coord=True)
        return replace(
            self,
            axis=axis,
            window_len=window_len,
            pad_after=pad_after,
            sample_step=sample_step,
            input_is_fft=input_is_fft,
        )

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        assert self.axis is not None
        data = patch_tree.data
        if self.input_is_fft:
            data = kernels.whiten_spectrum_kernel(
                data,
                axis=self.axis,
                window_len=self.window_len,
                water_level=self.water_level,
            )
            return patch_tree.new(data=data)
        if self.pad_after:
            pad_width = [(0, 0)] * len(patch_tree.dims)
            pad_width[self.axis] = (0, self.pad_after)
            data = jnp.pad(data, tuple(pad_width))
        data = kernels.whiten_kernel(
            data,
            axis=self.axis,
            window_len=self.window_len,
            water_level=self.water_level,
        ) / self.sample_step
        if self.pad_after:
            slices = [slice(None)] * data.ndim
            slices[self.axis] = slice(0, data.shape[self.axis] - self.pad_after)
            data = data[tuple(slices)]
        return patch_tree.new(
            data=data,
        )
