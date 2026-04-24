"""Core PatchOperation ports for the public pipeline operation surface."""

from __future__ import annotations

from dataclasses import dataclass, fields
from functools import reduce
from operator import add, mul, truediv
from typing import Any, ClassVar

import dascore as dc
import jax
import jax.numpy as jnp
import numpy as np
from dascore.exceptions import ParameterError
from dascore.proc.basic import pad as dc_pad
from dascore.transform.fourier import (
    _get_idft_dims_steps_axis,
    dft as dc_dft,
    idft as dc_idft,
)
from dascore.units import get_filter_units, get_quantity
from dascore.utils.misc import iterate
from dascore.utils.patch import get_dim_axis_value, get_dim_sampling_rate, get_patch_window_size
from dascore.utils.transformatter import FourierTransformatter
from scipy.fft import next_fast_len
from scipy.signal import ShortTimeFFT, get_window

from . import kernels
from .core import PatchBoundary, PatchOperation, PatchPyTree


def replace(obj, **changes):
    """Copy frozen operation instances without calling custom constructors."""
    out = object.__new__(type(obj))
    for field in fields(obj):
        object.__setattr__(out, field.name, changes.get(field.name, getattr(obj, field.name)))
    return out


def _dummy_patch(boundary: PatchBoundary, dtype=np.float64) -> dc.Patch:
    return dc.Patch(
        data=np.zeros(boundary.coords.shape, dtype=dtype),
        coords=boundary.coords,
        dims=boundary.dims,
        attrs=boundary.attrs,
    )


def _tree_boundary_from_patch(patch: dc.Patch) -> tuple[PatchPyTree, PatchBoundary]:
    return PatchPyTree.from_patch(patch)


def _get_data_units_from_dims(
    boundary: PatchBoundary,
    dims: tuple[str, ...],
    operator,
):
    if (data_units := get_quantity(boundary.attrs.data_units)) is None:
        return None
    dim_units = None
    for dim_name in dims:
        dim_unit = get_quantity(boundary.coord(dim_name).units)
        if dim_unit is None:
            continue
        dim_units = dim_unit if dim_units is None else dim_unit * dim_units
    return operator(data_units, dim_units) if dim_units is not None else data_units


@dataclass(frozen=True)
class Flip(PatchOperation):
    """Flip data and, optionally, associated coordinate values."""

    dims: tuple[str, ...] = ()
    flip_coords: bool = True
    axes: tuple[int, ...] = ()
    coord_axes: tuple[tuple[int, tuple[int, ...]], ...] = ()

    def __init__(self, *dims: str, flip_coords: bool = True):
        object.__setattr__(self, "dims", tuple(dims))
        object.__setattr__(self, "flip_coords", flip_coords)
        object.__setattr__(self, "axes", ())
        object.__setattr__(self, "coord_axes", ())

    def bind(self, boundary: PatchBoundary):
        dims = self.dims or boundary.dims
        axes = tuple(boundary.axis(dim) for dim in dims)
        coord_axes = []
        if self.flip_coords:
            for name in boundary.coord_names:
                axes_for_coord = tuple(
                    idx
                    for idx, coord_dim in enumerate(boundary.coord_dims(name))
                    if coord_dim in dims
                )
                if axes_for_coord:
                    coord_axes.append((boundary.coord_index(name), axes_for_coord))
        return replace(self, dims=dims, axes=axes, coord_axes=tuple(coord_axes))

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        data = kernels.flip_kernel(patch_tree.data, self.axes)
        coords = patch_tree.coord_values
        if self.flip_coords:
            for index, axes in self.coord_axes:
                coords = tuple(
                    jnp.flip(value, axis=axes) if idx == index else value
                    for idx, value in enumerate(coords)
                )
        return patch_tree.new(data=data, coords=coords)


@dataclass(frozen=True)
class Roll(PatchOperation):
    """Roll data along one dimension."""

    samples: bool = False
    update_coord: bool = False
    kwargs: dict[str, Any] | None = None
    axis: int | None = None
    shift: int | None = None

    def __init__(
        self,
        samples: bool = False,
        update_coord: bool = False,
        **kwargs,
    ):
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "update_coord", update_coord)
        object.__setattr__(self, "kwargs", dict(kwargs))
        object.__setattr__(self, "axis", None)
        object.__setattr__(self, "shift", None)

    def bind(self, boundary: PatchBoundary):
        if self.update_coord:
            raise NotImplementedError("Compiled roll currently requires update_coord=False.")
        kwargs = self.kwargs or {}
        dim = next(key for key in kwargs if key in boundary.dims)
        coord = boundary.coord(dim)
        return replace(
            self,
            axis=boundary.axis(dim),
            shift=int(coord.get_sample_count(kwargs[dim], samples=self.samples)),
        )

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        assert self.axis is not None and self.shift is not None
        return patch_tree.new(data=kernels.roll_kernel(patch_tree.data, self.shift, self.axis))


@dataclass(frozen=True)
class Standardize(PatchOperation):
    dim: str
    axis: int | None = None

    def bind(self, boundary: PatchBoundary):
        return replace(self, axis=boundary.axis(self.dim))

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        assert self.axis is not None
        return patch_tree.new(data=kernels.standardize_kernel(patch_tree.data, self.axis))


@dataclass(frozen=True)
class Detrend(PatchOperation):
    dim: str
    type: str = "linear"
    axis: int | None = None

    def bind(self, boundary: PatchBoundary):
        kernels.validate_detrend_type(self.type)
        return replace(self, axis=boundary.axis(self.dim))

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        assert self.axis is not None
        return patch_tree.new(
            data=kernels.detrend_kernel(patch_tree.data, axis=self.axis, type=self.type)
        )


@dataclass(frozen=True)
class Normalize(PatchOperation):
    dim: str
    norm: str = "l2"
    axis: int | None = None

    def bind(self, boundary: PatchBoundary):
        return replace(self, axis=boundary.axis(self.dim))

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        assert self.axis is not None
        return patch_tree.new(
            data=kernels.normalize_kernel(patch_tree.data, axis=self.axis, norm=self.norm)
        )


@dataclass(frozen=True)
class Differentiate(PatchOperation):
    dim: str | tuple[str, ...] | None
    order: int = 2
    step: int = 1
    axes: tuple[int, ...] = ()
    dxs_or_spacing: tuple[Any, ...] = ()
    attrs: Any = None

    def bind(self, boundary: PatchBoundary):
        dims = tuple(iterate(self.dim if self.dim is not None else boundary.dims))
        if self.step > 1 and len(dims) > 1:
            raise ParameterError("Step in patch.differentiate can only be used along one axis.")
        axes = []
        dxs = []
        for dim in dims:
            coord = _dummy_patch(boundary).get_coord(dim, require_sorted=True)
            val = coord.step if coord.evenly_sampled else coord.data
            dxs.append(dc.to_float(val))
            axes.append(boundary.axis(dim))
        attrs = boundary.attrs.update(
            data_units=_get_data_units_from_dims(boundary, dims, truediv)
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


@dataclass(frozen=True)
class Integrate(PatchOperation):
    dim: str | tuple[str, ...] | None
    definite: bool = False
    axes: tuple[int, ...] = ()
    dxs_or_spacing: tuple[Any, ...] = ()
    attrs: Any = None
    out_boundary: PatchBoundary | None = None
    out_coords: tuple[Any, ...] | None = None
    out_dtype_codes: tuple[Any, ...] | None = None
    out_dims: tuple[str, ...] | None = None

    def bind(self, boundary: PatchBoundary):
        dims = tuple(iterate(self.dim if self.dim is not None else boundary.dims))
        axes = []
        dxs = []
        for dim in dims:
            coord = _dummy_patch(boundary).get_coord(dim, require_sorted=True)
            val = coord.step if coord.evenly_sampled else coord.data
            dxs.append(dc.to_float(val))
            axes.append(boundary.axis(dim))
        attrs = boundary.attrs.update(
            data_units=_get_data_units_from_dims(boundary, dims, mul),
            coords={} if self.definite else boundary.attrs.coords,
        )
        if self.definite:
            out_patch = _dummy_patch(boundary).integrate(dim=dims, definite=True)
            out_tree, out_boundary = _tree_boundary_from_patch(out_patch)
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
        return self.out_boundary if self.definite and self.out_boundary else boundary.new(attrs=self.attrs)


def _taper_window(window_type: str, size: int) -> np.ndarray:
    return np.asarray(get_window(window_type, size, fftbins=False), dtype=np.float64)


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
    window_type: str = "hann"
    kwargs: dict[str, Any] | None = None
    axis: int | None = None
    weight: Any = None

    def __init__(self, window_type: str = "hann", **kwargs):
        object.__setattr__(self, "window_type", window_type)
        object.__setattr__(self, "kwargs", dict(kwargs))
        object.__setattr__(self, "axis", None)
        object.__setattr__(self, "weight", None)

    def bind(self, boundary: PatchBoundary):
        axis, start_slice, end_slice = _taper_slices(boundary, self.kwargs or {})
        length = boundary.coords.shape[axis]
        start_len = start_slice.stop
        end_len = length - end_slice.start
        if start_len is not None and end_len is not None and start_len > end_slice.start:
            raise ParameterError("Taper windows cannot overlap")
        weight = np.ones(length, dtype=np.float64)
        if start_len is not None and start_len > 0:
            weight[:start_len] = _taper_window(self.window_type, 2 * start_len)[:start_len]
        if end_slice.start is not None and end_slice.start < length:
            weight[end_slice.start:] = _taper_window(self.window_type, 2 * end_len)[end_len:]
        return replace(self, axis=axis, weight=weight)

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        assert self.axis is not None
        return patch_tree.new(data=kernels.apply_1d_weight_kernel(patch_tree.data, self.axis, self.weight))


def _taper_coord_inds(coord, values, relative, samples):
    error_msg = "A len 2 or 4 sequence is required for taper values"
    if not isinstance(values, (tuple, list, np.ndarray)) or not len(values):
        raise ParameterError(error_msg)
    if isinstance(values[0], (tuple, list, np.ndarray)):
        return reduce(add, (_taper_coord_inds(coord, item, relative, samples) for item in values))
    if len(values) not in {2, 4}:
        raise ParameterError(error_msg)
    out = [None] * len(values)
    for idx, value in enumerate(values):
        if value is None or value == ...:
            if len(values) == 2:
                raise ParameterError("Cannot use ... or None when only two values provided")
            out[idx] = 0 if (idx / len(out)) < 0.5 else len(coord) - 1
        else:
            out[idx] = coord.get_next_index(value, samples=samples, relative=relative)
    return [[0, *out, len(coord)]] if len(out) == 2 else [out]


def _taper_curve(coord, ind_1, ind_2, window_type, reverse=False):
    taper = _taper_window(window_type, (ind_2 - ind_1) * 2 + 1)[: ind_2 - ind_1]
    return taper[::-1] if reverse else taper


@dataclass(frozen=True)
class TaperRange(PatchOperation):
    window_type: str = "hann"
    invert: bool = False
    relative: bool = False
    samples: bool = False
    kwargs: dict[str, Any] | None = None
    axis: int | None = None
    weight: Any = None

    def __init__(self, window_type: str = "hann", invert=False, relative=False, samples=False, **kwargs):
        object.__setattr__(self, "window_type", window_type)
        object.__setattr__(self, "invert", invert)
        object.__setattr__(self, "relative", relative)
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "kwargs", dict(kwargs))
        object.__setattr__(self, "axis", None)
        object.__setattr__(self, "weight", None)

    def bind(self, boundary: PatchBoundary):
        kwargs = self.kwargs or {}
        dim = next(key for key in kwargs if key in boundary.dims)
        axis = boundary.axis(dim)
        coord = _dummy_patch(boundary).get_coord(dim, require_sorted=True)
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
        return patch_tree.new(data=kernels.apply_1d_weight_kernel(patch_tree.data, self.axis, self.weight))


@dataclass(frozen=True)
class GaussianFilter(PatchOperation):
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

    def bind(self, boundary: PatchBoundary):
        dimfo = get_dim_axis_value(_dummy_patch(boundary), kwargs=self.kwargs or {}, allow_multiple=True)
        axes = []
        sigma = []
        for dim, axis, value in dimfo:
            sigma.append(float(boundary.coord(dim).get_sample_count(value, samples=self.samples)))
            axes.append(axis)
        return replace(self, axes=tuple(axes), sigma=tuple(sigma))

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        return patch_tree.new(
            data=kernels.gaussian_filter_kernel(
                patch_tree.data, sigma=self.sigma, axes=self.axes, mode=self.mode, cval=self.cval, truncate=self.truncate
            )
        )


@dataclass(frozen=True)
class HampelFilter(PatchOperation):
    threshold: float = 10.0
    samples: bool = False
    approximate: bool = True
    kwargs: dict[str, Any] | None = None
    size: tuple[int, ...] = ()

    def __init__(self, *, threshold=10.0, samples=False, approximate=True, **kwargs):
        object.__setattr__(self, "threshold", float(threshold))
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "approximate", approximate)
        object.__setattr__(self, "kwargs", dict(kwargs))
        object.__setattr__(self, "size", ())

    def bind(self, boundary: PatchBoundary):
        if self.threshold <= 0 or not np.isfinite(self.threshold):
            raise ParameterError("hampel_filter threshold must be finite and greater than zero")
        size = get_patch_window_size(_dummy_patch(boundary), self.kwargs or {}, self.samples, require_odd=True, warn_above=10, min_samples=3)
        return replace(self, size=tuple(int(x) for x in size))

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        return patch_tree.new(
            data=kernels.hampel_filter_kernel(
                patch_tree.data, size=self.size, threshold=self.threshold, approximate=self.approximate
            )
        )


@dataclass(frozen=True)
class PassFilter(PatchOperation):
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

    def bind(self, boundary: PatchBoundary):
        from dascore.utils.misc import check_filter_kwargs

        dim, (arg1, arg2) = check_filter_kwargs(self.kwargs or {})
        filt_min, filt_max = get_filter_units(arg1, arg2, to_unit=boundary.coord(dim).units, dim=dim)
        sr = get_dim_sampling_rate(_dummy_patch(boundary), dim)
        sos = np.asarray(kernels.design_pass_filter_sos(sr, filt_min, filt_max, self.corners))
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
                patch_tree.data, sos=self.sos, zi=self.zi, padlen=self.padlen, axis=self.axis, zerophase=self.zerophase
            )
        )


@dataclass(frozen=True)
class Pad(PatchOperation):
    mode: str = "constant"
    constant_values: Any = 0
    expand_coords: bool = True
    samples: bool = False
    kwargs: dict[str, Any] | None = None
    data_pad_width: tuple[tuple[int, int], ...] = ()
    out_boundary: PatchBoundary | None = None
    out_coords: tuple[Any, ...] | None = None
    out_dtype_codes: tuple[Any, ...] | None = None
    out_dims: tuple[str, ...] | None = None

    def __init__(self, mode="constant", constant_values=0, expand_coords=True, samples=False, **kwargs):
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "constant_values", constant_values)
        object.__setattr__(self, "expand_coords", expand_coords)
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "kwargs", dict(kwargs))
        object.__setattr__(self, "data_pad_width", ())
        object.__setattr__(self, "out_boundary", None)
        object.__setattr__(self, "out_coords", None)
        object.__setattr__(self, "out_dtype_codes", None)
        object.__setattr__(self, "out_dims", None)

    def bind(self, boundary: PatchBoundary):
        patch = _dummy_patch(boundary)
        pad_width = [(0, 0)] * len(patch.shape)
        dimfo = get_dim_axis_value(patch, kwargs=self.kwargs or {}, allow_multiple=True)
        for dim, axis, value in dimfo:
            coord = patch.get_coord(dim, require_evenly_sampled=False)
            if value in {"fft", "correlate"}:
                target_length = len(coord) if value == "fft" else 2 * len(coord) - 1
                pad_tuple = (0, next_fast_len(target_length) - len(coord))
            else:
                if not isinstance(value, (tuple, list)):
                    value = (value, value)
                pad_tuple = (
                    tuple(int(coord.get_sample_count(x)) for x in value)
                    if not self.samples
                    else tuple(int(x) for x in value)
                )
            pad_width[axis] = tuple(pad_tuple)
        out = dc_pad.func(
            patch, mode=self.mode, constant_values=self.constant_values, expand_coords=self.expand_coords, samples=self.samples, **(self.kwargs or {})
        )
        out_tree, out_boundary = _tree_boundary_from_patch(out)
        return replace(
            self,
            data_pad_width=tuple(pad_width),
            out_boundary=out_boundary,
            out_coords=out_tree.coord_values,
            out_dtype_codes=out_tree.coord_dtype_codes,
            out_dims=out_tree.dims,
        )

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        return patch_tree.new(
            data=jnp.pad(patch_tree.data, self.data_pad_width, mode=self.mode, constant_values=self.constant_values),
            coords=self.out_coords,
            coord_dtype_codes=self.out_dtype_codes,
            dims=self.out_dims,
        )

    def update_boundary(self, boundary: PatchBoundary) -> PatchBoundary:
        _ = boundary
        assert self.out_boundary is not None
        return self.out_boundary


@dataclass(frozen=True)
class Hilbert(PatchOperation):
    dim: str
    axis: int | None = None

    def bind(self, boundary: PatchBoundary):
        if boundary.coord(self.dim).step is None:
            raise dc.exceptions.CoordError(f"Coordinate {self.dim} is not evenly sampled as required by hilbert")
        return replace(self, axis=boundary.axis(self.dim))

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        assert self.axis is not None
        return patch_tree.new(data=kernels.hilbert_kernel(patch_tree.data, axis=self.axis))


@dataclass(frozen=True)
class Envelope(Hilbert):
    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        assert self.axis is not None
        return patch_tree.new(data=kernels.envelope_kernel(patch_tree.data, axis=self.axis))


@dataclass(frozen=True)
class Dft(PatchOperation):
    method_name: ClassVar[str | None] = "dft"
    dim: str | None | tuple[str, ...]
    real: str | bool | None = None
    pad: bool = True
    axes: tuple[int, ...] = ()
    dxs: tuple[float, ...] = ()
    real_axis: int | None = None
    pad_width: tuple[tuple[int, int], ...] = ()
    out_boundary: PatchBoundary | None = None
    out_coords: tuple[Any, ...] | None = None
    out_dtype_codes: tuple[Any, ...] | None = None
    out_dims: tuple[str, ...] | None = None

    def bind(self, boundary: PatchBoundary):
        patch = _dummy_patch(boundary)
        dims = list(iterate(self.dim if self.dim is not None else boundary.dims))
        real = dims[-1] if self.real is True else self.real
        if isinstance(real, str) and real in dims:
            dims.append(dims.pop(dims.index(real)))
        pad_width = [(0, 0)] * len(boundary.dims)
        work_patch = patch
        if self.pad:
            for dim in dims:
                axis = work_patch.get_axis(dim)
                target = next_fast_len(len(work_patch.get_coord(dim)))
                pad_width[axis] = (0, target - len(work_patch.get_coord(dim)))
            work_patch = dc_pad.func(work_patch, **{dim: "fft" for dim in dims})
        axes = tuple(work_patch.get_axis(dim) for dim in dims)
        dxs = tuple(float(dc.to_float(work_patch.get_coord(dim).step)) for dim in dims)
        out = dc_dft.func(patch, dim=self.dim, real=self.real, pad=self.pad)
        out_tree, out_boundary = _tree_boundary_from_patch(out)
        return replace(
            self,
            axes=axes,
            dxs=dxs,
            real_axis=work_patch.get_axis(real) if isinstance(real, str) and real in dims else None,
            pad_width=tuple(pad_width),
            out_boundary=out_boundary,
            out_coords=out_tree.coord_values,
            out_dtype_codes=out_tree.coord_dtype_codes,
            out_dims=out_tree.dims,
        )

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        data = patch_tree.data
        if any(before or after for before, after in self.pad_width):
            data = jnp.pad(data, self.pad_width)
        data = kernels.dft_kernel(data, axes=self.axes, dxs=self.dxs, real_axis=self.real_axis)
        return patch_tree.new(data=data, coords=self.out_coords, coord_dtype_codes=self.out_dtype_codes, dims=self.out_dims)

    def update_boundary(self, boundary: PatchBoundary) -> PatchBoundary:
        _ = boundary
        assert self.out_boundary is not None
        return self.out_boundary


@dataclass(frozen=True)
class Idft(PatchOperation):
    method_name: ClassVar[str | None] = "idft"
    dim: str | None | tuple[str, ...] = None
    axes: tuple[int, ...] = ()
    steps: tuple[float, ...] = ()
    sizes: tuple[int, ...] | None = None
    real: bool = False
    out_boundary: PatchBoundary | None = None
    out_coords: tuple[Any, ...] | None = None
    out_dtype_codes: tuple[Any, ...] | None = None
    out_dims: tuple[str, ...] | None = None

    def bind(self, boundary: PatchBoundary):
        patch = _dummy_patch(boundary, dtype=np.complex128)
        dims, _steps, axes, real = _get_idft_dims_steps_axis(patch, self.dim)
        out = dc_idft.func(patch, dim=self.dim)
        out_tree, out_boundary = _tree_boundary_from_patch(out)
        sizes = tuple(out.shape[axis] for axis in axes)
        new_dims = FourierTransformatter().rename_dims(dims, forward=False)
        out_steps = tuple(
            float(dc.to_float(out.get_coord(dim).step)) for dim in new_dims
        )
        return replace(
            self,
            axes=tuple(axes),
            steps=out_steps,
            sizes=sizes,
            real=real,
            out_boundary=out_boundary,
            out_coords=out_tree.coord_values,
            out_dtype_codes=out_tree.coord_dtype_codes,
            out_dims=out_tree.dims,
        )

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        data = kernels.idft_kernel(patch_tree.data, axes=self.axes, new_steps=self.steps, sizes=self.sizes, real=self.real)
        return patch_tree.new(data=data, coords=self.out_coords, coord_dtype_codes=self.out_dtype_codes, dims=self.out_dims)

    def update_boundary(self, boundary: PatchBoundary) -> PatchBoundary:
        _ = boundary
        assert self.out_boundary is not None
        return self.out_boundary


@dataclass(frozen=True)
class Whiten(PatchOperation):
    smooth_size: float | None = None
    water_level: float | None = None
    kwargs: dict[str, Any] | None = None
    axis: int | None = None
    window_len: int | None = None
    sample_step: float = 1.0
    patch_template: Any = None

    def __init__(self, smooth_size=None, water_level=None, **kwargs):
        object.__setattr__(self, "smooth_size", smooth_size)
        object.__setattr__(self, "water_level", water_level)
        object.__setattr__(self, "kwargs", dict(kwargs))
        object.__setattr__(self, "axis", None)
        object.__setattr__(self, "window_len", None)
        object.__setattr__(self, "sample_step", 1.0)
        object.__setattr__(self, "patch_template", None)

    def bind(self, boundary: PatchBoundary):
        kwargs = self.kwargs or {}
        dim = next(iter(kwargs), "time" if "time" in boundary.dims else boundary.dims[-1])
        axis = boundary.axis(dim)
        sample_step = abs(float(dc.to_float(boundary.coord(dim).step)))
        window_len = None
        if self.smooth_size is not None:
            fft_dim = FourierTransformatter().rename_dims(dim)[0]
            fft_patch = dc_dft.func(_dummy_patch(boundary), dim=dim, real=True)
            coord = fft_patch.get_coord(fft_dim)
            window_len = coord.get_sample_count(self.smooth_size, enforce_lt_coord=True)
        return replace(
            self,
            axis=axis,
            window_len=window_len,
            sample_step=sample_step,
            patch_template=_dummy_patch(boundary),
        )

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        assert self.axis is not None
        source = jnp.asarray(patch_tree.data)
        shape_dtype = jax.ShapeDtypeStruct(source.shape, source.dtype)

        def _callback(data):
            patch = self.patch_template.new(data=np.asarray(data))
            kwargs = self.kwargs or {}
            return np.asarray(
                patch.whiten.func(
                    patch,
                    smooth_size=self.smooth_size,
                    water_level=self.water_level,
                    **kwargs,
                ).data
            )

        return patch_tree.new(data=jax.pure_callback(_callback, shape_dtype, source))


@dataclass(frozen=True)
class Fbe(PatchOperation):
    overlap: Any = 0
    samples: bool = False
    detrend: bool = False
    taper_window: str | np.ndarray | tuple = "hann"
    fmin: float | None = None
    fmax: float | None = None
    kwargs: dict[str, Any] | None = None
    prepared: dict[str, Any] | None = None
    out_boundary: PatchBoundary | None = None
    out_coords: tuple[Any, ...] | None = None
    out_dtype_codes: tuple[Any, ...] | None = None
    out_dims: tuple[str, ...] | None = None

    def __init__(self, overlap=0, samples=False, detrend=False, taper_window="hann", fmin=None, fmax=None, **kwargs):
        object.__setattr__(self, "overlap", overlap)
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "detrend", detrend)
        object.__setattr__(self, "taper_window", taper_window)
        object.__setattr__(self, "fmin", fmin)
        object.__setattr__(self, "fmax", fmax)
        object.__setattr__(self, "kwargs", dict(kwargs))
        object.__setattr__(self, "prepared", None)
        object.__setattr__(self, "out_boundary", None)
        object.__setattr__(self, "out_coords", None)
        object.__setattr__(self, "out_dtype_codes", None)
        object.__setattr__(self, "out_dims", None)

    def bind(self, boundary: PatchBoundary):
        patch = _dummy_patch(boundary)
        call_kwargs = dict(self.kwargs or {})
        dim, axis, val = get_dim_axis_value(patch, kwargs=call_kwargs)[0]
        coord = patch.get_coord(dim, require_evenly_sampled=True)
        window_samples = coord.get_sample_count(val, samples=self.samples, enforce_lt_coord=True)
        step = dc.to_float(coord.step)
        sampling_rate = 1 / abs(step)
        window = self.taper_window if isinstance(self.taper_window, np.ndarray) else get_window(self.taper_window, window_samples, fftbins=False)
        overlap = coord[:window_samples].get_sample_count(self.overlap, samples=self.samples, enforce_lt_coord=True) if self.overlap is not None else 0
        hop = window_samples - overlap
        stft = ShortTimeFFT(win=window, hop=hop, fs=sampling_rate, fft_mode="onesided", mfft=window_samples)
        frame_times = np.asarray(stft.t(len(coord)))
        frame_starts = np.rint(frame_times * sampling_rate).astype(np.int64) - stft.m_num_mid
        frequencies = np.asarray(stft.f, dtype=np.float64)
        mask = np.ones(len(frequencies), dtype=bool)
        if self.fmin is not None:
            mask &= frequencies >= self.fmin
        if self.fmax is not None:
            mask &= frequencies <= self.fmax
        prepared = {
            "axis": axis,
            "window": np.asarray(window),
            "frame_starts": frame_starts,
            "selected_bins": np.flatnonzero(mask).astype(np.int64),
            "sample_step": step,
            "detrend": self.detrend,
        }
        # Reuse the existing DASCore STFT metadata construction by running on zeros.
        out = patch.stft(**{dim: val}, overlap=self.overlap, samples=self.samples, detrend=self.detrend, taper_window=self.taper_window).abs()
        ft_dim = next(dim_name for dim_name in out.dims if dim_name.startswith("ft_"))
        if self.fmin is not None or self.fmax is not None:
            out = out.select(**{ft_dim: (self.fmin, self.fmax)})
        out = out.sum(dim=ft_dim, dim_reduce="squeeze")
        out_tree, out_boundary = _tree_boundary_from_patch(out)
        return replace(
            self,
            prepared=prepared,
            out_boundary=out_boundary,
            out_coords=out_tree.coord_values,
            out_dtype_codes=out_tree.coord_dtype_codes,
            out_dims=out_tree.dims,
        )

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        assert self.prepared is not None
        data = kernels.banded_stft_kernel(patch_tree.data, **self.prepared)
        return patch_tree.new(data=data, coords=self.out_coords, coord_dtype_codes=self.out_dtype_codes, dims=self.out_dims)

    def update_boundary(self, boundary: PatchBoundary) -> PatchBoundary:
        _ = boundary
        assert self.out_boundary is not None
        return self.out_boundary
