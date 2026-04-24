"""Filter patch operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

import numpy as np
import dascore as dc
from dascore.exceptions import FilterValueError, ParameterError
from dascore.proc.filter import get_inverted_quant
from dascore.units import get_filter_units
from dascore.utils.patch import (
    get_dim_axis_value,
    get_dim_sampling_rate,
    get_patch_window_size,
)
from scipy.fft import next_fast_len

from dasjax.core import PatchBoundary, PatchOperation, PatchPyTree

from .. import kernels
from .common import dummy_patch, replace


def _size_and_axes(
    boundary: PatchBoundary,
    kwargs: dict[str, Any],
    samples: bool,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Resolve DASCore dimension window kwargs into static size and axes."""
    dimfo = get_dim_axis_value(
        dummy_patch(boundary), kwargs=kwargs, allow_multiple=True
    )
    axes = [axis for _dim, axis, _value in dimfo]
    size = [1] * len(boundary.dims)
    for dim, axis, value in dimfo:
        size[axis] = boundary.coord(dim).get_sample_count(value, samples=samples)
    return tuple(int(x) for x in size), tuple(int(x) for x in axes)


@dataclass(frozen=True)
class GaussianFilter(PatchOperation):
    """Apply a Gaussian smoothing filter to patch data."""

    samples: bool = False
    mode: str = "reflect"
    cval: float = 0.0
    truncate: float = 4.0
    kwargs: dict[str, Any] | None = None
    sigma: tuple[float, ...] = ()
    axes: tuple[int, ...] = ()

    def __init__(self, samples=False, mode="reflect", cval=0.0, truncate=4.0, **kwargs):
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "cval", float(cval))
        object.__setattr__(self, "truncate", float(truncate))
        object.__setattr__(self, "kwargs", dict(kwargs))
        object.__setattr__(self, "sigma", ())
        object.__setattr__(self, "axes", ())

    def bind(self, boundary: PatchBoundary) -> Self:
        dimfo = get_dim_axis_value(
            dummy_patch(boundary), kwargs=self.kwargs or {}, allow_multiple=True
        )
        axes = []
        sigma = []
        for dim, axis, value in dimfo:
            sigma.append(
                float(boundary.coord(dim).get_sample_count(value, samples=self.samples))
            )
            axes.append(axis)
        return replace(self, axes=tuple(axes), sigma=tuple(sigma))

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        return patch_tree.new(
            data=kernels.gaussian_filter_kernel(
                patch_tree.data,
                sigma=self.sigma,
                axes=self.axes,
                mode=self.mode,
                cval=self.cval,
                truncate=self.truncate,
            )
        )


@dataclass(frozen=True)
class HampelFilter(PatchOperation):
    """Replace outliers with a Hampel filter."""

    threshold: float = 10.0
    samples: bool = False
    approximate: bool = True
    kwargs: dict[str, Any] | None = None
    size: tuple[int, ...] = ()

    def __init__(self, *, threshold=10.0, samples=False, approximate=True, **kwargs):
        if not approximate:
            msg = "Exact Hampel median filtering is not implemented in pure JAX."
            raise NotImplementedError(msg)
        object.__setattr__(self, "threshold", float(threshold))
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "approximate", approximate)
        object.__setattr__(self, "kwargs", dict(kwargs))
        object.__setattr__(self, "size", ())

    def bind(self, boundary: PatchBoundary) -> Self:
        if self.threshold <= 0 or not np.isfinite(self.threshold):
            raise ParameterError(
                "hampel_filter threshold must be finite and greater than zero"
            )
        size = get_patch_window_size(
            dummy_patch(boundary),
            self.kwargs or {},
            self.samples,
            require_odd=True,
            warn_above=10,
            min_samples=3,
        )
        return replace(self, size=tuple(int(x) for x in size))

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        return patch_tree.new(
            data=kernels.hampel_filter_kernel(
                patch_tree.data,
                size=self.size,
                threshold=self.threshold,
                approximate=self.approximate,
            )
        )


@dataclass(frozen=True)
class MedianFilter(PatchOperation):
    """Apply an exact median filter to patch data."""

    samples: bool = False
    mode: str = "reflect"
    cval: float = 0.0
    kwargs: dict[str, Any] | None = None
    size: tuple[int, ...] = ()

    def __init__(self, samples=False, mode="reflect", cval=0.0, **kwargs):
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "cval", float(cval))
        object.__setattr__(self, "kwargs", dict(kwargs))
        object.__setattr__(self, "size", ())

    def bind(self, boundary: PatchBoundary) -> Self:
        size, _axes = _size_and_axes(boundary, self.kwargs or {}, self.samples)
        return replace(self, size=size)

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        return patch_tree.new(
            data=kernels.median_filter_kernel(
                patch_tree.data,
                size=self.size,
                mode=self.mode,
                cval=self.cval,
            )
        )


@dataclass(frozen=True)
class SobelFilter(PatchOperation):
    """Apply a Sobel filter along one dimension."""

    dim: str
    mode: str = "reflect"
    cval: float = 0.0
    axis: int | None = None

    def bind(self, boundary: PatchBoundary) -> Self:
        if not isinstance(self.dim, str):
            msg = "dim parameter should be a string."
            raise FilterValueError(msg)
        return replace(self, axis=boundary.axis(self.dim))

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        assert self.axis is not None
        return patch_tree.new(
            data=kernels.sobel_filter_kernel(
                patch_tree.data,
                axis=self.axis,
                mode=self.mode,
                cval=self.cval,
            )
        )


@dataclass(frozen=True)
class SavgolFilter(PatchOperation):
    """Apply Savitzky-Golay filtering along one or more dimensions."""

    polyorder: int
    samples: bool = False
    mode: str = "interp"
    cval: float = 0.0
    kwargs: dict[str, Any] | None = None
    size: tuple[int, ...] = ()
    axes: tuple[int, ...] = ()
    coeffs: tuple[Any, ...] = ()
    left_coeffs: tuple[Any, ...] = ()
    right_coeffs: tuple[Any, ...] = ()

    def __init__(self, polyorder, samples=False, mode="interp", cval=0.0, **kwargs):
        object.__setattr__(self, "polyorder", int(polyorder))
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "cval", float(cval))
        object.__setattr__(self, "kwargs", dict(kwargs))
        object.__setattr__(self, "size", ())
        object.__setattr__(self, "axes", ())
        object.__setattr__(self, "coeffs", ())
        object.__setattr__(self, "left_coeffs", ())
        object.__setattr__(self, "right_coeffs", ())

    def bind(self, boundary: PatchBoundary) -> Self:
        size, axes = _size_and_axes(boundary, self.kwargs or {}, self.samples)
        coeffs = []
        left_coeffs = []
        right_coeffs = []
        for axis in axes:
            window = size[axis]
            coeff, left, right = kernels.savgol_coefficients(window, self.polyorder)
            coeffs.append(coeff)
            left_coeffs.append(left)
            right_coeffs.append(right)
        return replace(
            self,
            size=size,
            axes=axes,
            coeffs=tuple(coeffs),
            left_coeffs=tuple(left_coeffs),
            right_coeffs=tuple(right_coeffs),
        )

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        return patch_tree.new(
            data=kernels.savgol_filter_kernel(
                patch_tree.data,
                size=self.size,
                axes=self.axes,
                coeffs=self.coeffs,
                left_coeffs=self.left_coeffs,
                right_coeffs=self.right_coeffs,
                mode=self.mode,
                cval=self.cval,
            )
        )


@dataclass(frozen=True)
class NotchFilter(PatchOperation):
    """Apply one or more narrow notch filters."""

    q: float
    kwargs: dict[str, Any] | None = None
    axes: tuple[int, ...] = ()
    b: tuple[Any, ...] = ()
    a: tuple[Any, ...] = ()
    zi: tuple[Any, ...] = ()
    padlen: tuple[int, ...] = ()

    def __init__(self, q, **kwargs):
        object.__setattr__(self, "q", float(q))
        object.__setattr__(self, "kwargs", dict(kwargs))
        object.__setattr__(self, "axes", ())
        object.__setattr__(self, "b", ())
        object.__setattr__(self, "a", ())
        object.__setattr__(self, "zi", ())
        object.__setattr__(self, "padlen", ())

    def bind(self, boundary: PatchBoundary) -> Self:
        dimfo = get_dim_axis_value(
            dummy_patch(boundary), kwargs=self.kwargs or {}, allow_multiple=True
        )
        axes = []
        b_values = []
        a_values = []
        zi_values = []
        padlens = []
        for dim, axis, value in dimfo:
            coord = boundary.coord(dim)
            if isinstance(value, dc.units.Quantity) and coord.units is not None:
                value, _ = get_inverted_quant(value, coord.units)
            w0 = float(dc.to_float(value))
            sr = get_dim_sampling_rate(dummy_patch(boundary), dim)
            b, a, zi, padlen = kernels.design_notch_filter(sr, w0, self.q)
            axes.append(axis)
            b_values.append(b)
            a_values.append(a)
            zi_values.append(zi)
            padlens.append(padlen)
        return replace(
            self,
            axes=tuple(int(x) for x in axes),
            b=tuple(b_values),
            a=tuple(a_values),
            zi=tuple(zi_values),
            padlen=tuple(int(x) for x in padlens),
        )

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        data = patch_tree.data
        for axis, b, a, zi, padlen in zip(
            self.axes, self.b, self.a, self.zi, self.padlen, strict=True
        ):
            data = kernels.notch_filter_kernel(
                data,
                b=b,
                a=a,
                zi=zi,
                padlen=padlen,
                axis=axis,
            )
        return patch_tree.new(data=data)


@dataclass(frozen=True)
class WienerFilter(PatchOperation):
    """Apply a local-statistics Wiener filter."""

    noise: float | None = None
    samples: bool = False
    kwargs: dict[str, Any] | None = None
    size: tuple[int, ...] = ()

    def __init__(self, *, noise=None, samples=False, **kwargs):
        object.__setattr__(self, "noise", noise)
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "kwargs", dict(kwargs))
        object.__setattr__(self, "size", ())

    def bind(self, boundary: PatchBoundary) -> Self:
        if not self.kwargs:
            msg = (
                "To use wiener_filter you must specify dimension-specific window "
                "sizes via kwargs (e.g., time=5, distance=3)"
            )
            raise ParameterError(msg)
        size = get_patch_window_size(
            dummy_patch(boundary), self.kwargs or {}, self.samples, min_samples=1
        )
        return replace(self, size=tuple(int(x) for x in size))

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        return patch_tree.new(
            data=kernels.wiener_filter_kernel(
                patch_tree.data, size=self.size, noise=self.noise
            )
        )


@dataclass(frozen=True)
class SlopeFilter(PatchOperation):
    """Filter over slopes in the 2-D Fourier domain."""

    filt: Any
    dims: tuple[str, str] = ("distance", "time")
    directional: bool = False
    notch: bool | None = None
    invert: bool = False
    axes: tuple[int, int] = ()
    dxs: tuple[float, float] = ()
    mask: Any = None
    steps: tuple[float, float] = ()
    sizes: tuple[int, int] = ()
    pad_width: tuple[tuple[int, int], ...] = ()
    attrs: Any = None

    def bind(self, boundary: PatchBoundary) -> Self:
        filt = np.asarray(self.filt, dtype=np.float64)
        if not (len(filt) == 4 and np.all(filt[:-1] <= filt[1:])):
            msg = f"filt must be a sorted length 4 sequence. You passed {self.filt}"
            raise ParameterError(msg)
        patch = dummy_patch(boundary)
        pad_width = [(0, 0)] * len(boundary.dims)
        for dim in self.dims:
            axis = patch.get_axis(dim)
            target = next_fast_len(len(patch.get_coord(dim)))
            pad_width[axis] = (0, target - len(patch.get_coord(dim)))
        work_patch = patch.pad.func(patch, **{dim: "fft" for dim in self.dims})
        dft_patch = work_patch.dft.func(work_patch, self.dims, pad=False)
        freq_dims = tuple(f"ft_{x}" for x in self.dims)
        dim1, dim2 = freq_dims[-1], freq_dims[-2]
        coord1 = dft_patch.get_array(dim1)
        coord2 = dft_patch.get_array(dim2) + np.finfo(float).eps
        ax1 = dft_patch.dims.index(dim1)
        ax2 = dft_patch.dims.index(dim2)
        shape_1 = [None] * dft_patch.ndim
        shape_2 = [None] * dft_patch.ndim
        shape_1[ax1] = slice(None)
        shape_2[ax2] = slice(None)
        slope = coord1[tuple(shape_1)] / coord2[tuple(shape_2)]
        if not self.directional:
            slope = np.abs(slope)
        invert = self.invert if self.notch is None else self.notch
        fac = np.where(
            (slope >= filt[0]) & (slope <= filt[1]),
            1.0 - np.sin(0.5 * np.pi * (slope - filt[0]) / (filt[1] - filt[0])),
            1.0,
        )
        fac = np.where((slope >= filt[1]) & (slope <= filt[2]), 0.0, fac)
        fac = np.where(
            (slope >= filt[2]) & (slope <= filt[3]),
            np.sin(0.5 * np.pi * (slope - filt[2]) / (filt[3] - filt[2])),
            fac,
        )
        mask = fac if invert else 1.0 - fac
        axes = tuple(work_patch.get_axis(dim) for dim in self.dims)
        dxs = tuple(float(dc.to_float(work_patch.get_coord(dim).step)) for dim in self.dims)
        return replace(
            self,
            axes=axes,
            dxs=dxs,
            mask=np.asarray(mask),
            steps=dxs,
            sizes=tuple(len(work_patch.get_coord(dim)) for dim in self.dims),
            pad_width=tuple(pad_width),
            attrs=boundary.attrs,
        )

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        return patch_tree.new(
            data=kernels.slope_filter_kernel(
                patch_tree.data,
                axes=self.axes,
                dxs=self.dxs,
                mask=self.mask,
                steps=self.steps,
                sizes=self.sizes,
                pad_width=self.pad_width,
            )
        )

    def update_boundary(self, boundary: PatchBoundary) -> PatchBoundary:
        return boundary.new(attrs=self.attrs)


@dataclass(frozen=True)
class PassFilter(PatchOperation):
    """Apply a band, low, or high pass filter along one dimension."""

    corners: int = 4
    zerophase: bool = True
    kwargs: dict[str, Any] | None = None
    sos: Any = None
    zi: Any = None
    padlen: int | None = None
    axis: int | None = None

    def __init__(self, corners=4, zerophase=True, **kwargs):
        object.__setattr__(self, "corners", int(corners))
        object.__setattr__(self, "zerophase", bool(zerophase))
        object.__setattr__(self, "kwargs", dict(kwargs))
        object.__setattr__(self, "sos", None)
        object.__setattr__(self, "zi", None)
        object.__setattr__(self, "padlen", None)
        object.__setattr__(self, "axis", None)

    def bind(self, boundary: PatchBoundary) -> Self:
        from dascore.utils.misc import check_filter_kwargs

        dim, (arg1, arg2) = check_filter_kwargs(self.kwargs or {})
        filt_min, filt_max = get_filter_units(
            arg1, arg2, to_unit=boundary.coord(dim).units, dim=dim
        )
        sr = get_dim_sampling_rate(dummy_patch(boundary), dim)
        sos = np.asarray(
            kernels.design_pass_filter_sos(sr, filt_min, filt_max, self.corners)
        )
        return replace(
            self,
            sos=sos,
            zi=kernels.pass_filter_initial_state(sos),
            padlen=kernels.pass_filter_default_padlen(sos),
            axis=boundary.axis(dim),
        )

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        assert self.axis is not None and self.padlen is not None
        return patch_tree.new(
            data=kernels.pass_filter_kernel(
                patch_tree.data,
                sos=self.sos,
                zi=self.zi,
                padlen=self.padlen,
                axis=self.axis,
                zerophase=self.zerophase,
            )
        )
