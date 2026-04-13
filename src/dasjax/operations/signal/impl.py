"""Executable implementations for signal operations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import reduce
from operator import add
from typing import Any
from operator import mul, truediv

import dascore as dc
import numpy as np
from dascore.constants import PatchType
from dascore.exceptions import ParameterError
from dascore.proc.taper import taper as dc_taper
from dascore.proc.taper import taper_range as dc_taper_range
from dascore.transform.differentiate import differentiate as dc_differentiate
from dascore.transform.integrate import integrate as dc_integrate
from dascore.units import get_quantity
from dascore.units import Quantity
from dascore.utils.misc import iterate
from scipy.signal import get_window

from dasjax import kernels

from ..common import get_axis, get_axis_from_dims, update_patch
from ..patch_ops import MetaDelta, PatchOp, PatchSpec


@dataclass(frozen=True)
class DetrendOp(PatchOp):
    """Compiled detrend operation using selected-axis inference."""

    dim: str
    type: str = "linear"

    def kernel(self, data, selected_axis):
        return {
            "data": kernels.detrend_kernel(data, axis=selected_axis, type=self.type)
        }


@dataclass(frozen=True)
class NormalizeOp(PatchOp):
    """Compiled normalize operation using selected-axis inference."""

    dim: str
    norm: str = "l2"

    def kernel(self, data, selected_axis):
        return {
            "data": kernels.normalize_kernel(data, axis=selected_axis, norm=self.norm)
        }


@dataclass(frozen=True)
class DifferentiateOp(PatchOp):
    """Compiled differentiate operation with metadata updates."""

    axes: tuple[int, ...]
    dxs_or_spacing: tuple[Any, ...]
    order: int = 2
    step: int = 1
    dims: tuple[str, ...] = ()
    new_attrs: Any = None

    @classmethod
    def prepare(
        cls,
        patch: PatchType,
        dim: str | tuple[str, ...] | None,
        order: int = 2,
        step: int = 1,
    ) -> "DifferentiateOp":
        dims = tuple(iterate(dim if dim is not None else patch.dims))
        dxs_or_spacing = []
        axes = []
        for dim_name in dims:
            coord = patch.get_coord(dim_name, require_sorted=True)
            val = coord.step if coord.evenly_sampled else coord.data
            dxs_or_spacing.append(dc.to_float(val))
            axes.append(patch.get_axis(dim_name))
        if step > 1 and len(axes) > 1:
            msg = "Step in patch.differentiate can only be used along one axis."
            raise ParameterError(msg)
        new_attrs = patch.attrs.update(
            data_units=_get_data_units_from_dims_local(patch, dims, truediv)
        )
        return cls(
            axes=tuple(axes),
            dxs_or_spacing=tuple(dxs_or_spacing),
            order=order,
            step=step,
            dims=dims,
            new_attrs=new_attrs,
        )

    def kernel(self, data):
        out = data
        for axis, dx_or_spacing in zip(self.axes, self.dxs_or_spacing, strict=True):
            out = kernels.differentiate_kernel(
                out,
                axis=axis,
                dx_or_spacing=dx_or_spacing,
                order=self.order,
                step=self.step,
            )
        return {"data": out}

    def meta_delta(self, spec: PatchSpec) -> MetaDelta | None:
        _ = spec
        return MetaDelta(attrs=self.new_attrs)


@dataclass(frozen=True)
class IntegrateOp(PatchOp):
    """Compiled integrate operation with eager fallback for definite integrals."""

    axes: tuple[int, ...]
    dxs_or_spacing: tuple[Any, ...]
    definite: bool = False
    dims: tuple[str, ...] = ()
    new_attrs: Any = None
    mutates_spec = True
    requires_materialized_patch_after = True
    compile_category = "compiled_boundary"

    @classmethod
    def prepare(
        cls,
        patch: PatchType,
        dim: tuple[str, ...] | str | None,
        definite: bool = False,
    ) -> "IntegrateOp":
        dims = tuple(iterate(dim if dim is not None else patch.dims))
        dxs_or_spacing = []
        axes = []
        for dim_name in dims:
            coord = patch.get_coord(dim_name, require_sorted=True)
            val = coord.step if coord.evenly_sampled else coord.data
            dxs_or_spacing.append(dc.to_float(val))
            axes.append(patch.get_axis(dim_name))
        new_attrs = patch.attrs.update(
            data_units=_get_data_units_from_dims_local(patch, dims, mul),
            coords={} if definite else patch.attrs.coords,
        )
        return cls(
            axes=tuple(axes),
            dxs_or_spacing=tuple(dxs_or_spacing),
            definite=definite,
            dims=dims,
            new_attrs=new_attrs,
        )

    def kernel(self, data):
        out = data
        for axis, dx_or_spacing in zip(self.axes, self.dxs_or_spacing, strict=True):
            out = kernels.integrate_kernel(
                out,
                axis=axis,
                dx_or_spacing=dx_or_spacing,
                definite=self.definite,
            )
        return {"data": out}

    def meta_delta(self, spec: PatchSpec) -> MetaDelta | None:
        if self.definite:
            _ = spec
            return None
        return MetaDelta(attrs=self.new_attrs)

    def reconstruct(
        self,
        previous_patch: PatchType,
        spec: PatchSpec,
        state,
    ) -> PatchType:
        if self.definite:
            return dc_integrate.func(previous_patch, dim=self.dims, definite=True)
        return super().reconstruct(previous_patch, spec, state)


@dataclass(frozen=True)
class TaperOp(PatchOp):
    """Compiled taper operation."""

    axis: int
    weight: np.ndarray
    requires_materialized_patch_for_prepare = True

    @classmethod
    def prepare(
        cls,
        patch: PatchType,
        window_type: str = "hann",
        **kwargs,
    ) -> "TaperOp":
        _, prepared = prepare_taper_call(
            patch,
            (),
            {"window_type": window_type, **kwargs},
        )
        return cls(axis=prepared["axis"], weight=prepared["weight"])

    def kernel(self, data):
        return {
            "data": kernels.apply_1d_weight_kernel(
                data, axis=self.axis, weight=self.weight
            )
        }


@dataclass(frozen=True)
class TaperRangeOp(PatchOp):
    """Compiled taper_range operation."""

    axis: int
    weight: np.ndarray
    requires_materialized_patch_for_prepare = True

    @classmethod
    def prepare(
        cls,
        patch: PatchType,
        window_type: str = "hann",
        invert: bool = False,
        relative: bool = False,
        samples: bool = False,
        **kwargs,
    ) -> "TaperRangeOp":
        _, prepared = prepare_taper_range_call(
            patch,
            (),
            {
                "window_type": window_type,
                "invert": invert,
                "relative": relative,
                "samples": samples,
                **kwargs,
            },
        )
        return cls(axis=prepared["axis"], weight=prepared["weight"])

    def kernel(self, data):
        return {
            "data": kernels.apply_1d_weight_kernel(
                data, axis=self.axis, weight=self.weight
            )
        }


def window_function(window_type: str, size: int) -> np.ndarray:
    return np.asarray(get_window(window_type, size, fftbins=False), dtype=np.float64)


def _get_data_units_from_dims_local(
    patch: PatchType,
    dims: tuple[str, ...],
    operator,
):
    """Derive data units for dimension-wise math without eager patch execution."""
    if (data_units := get_quantity(patch.attrs.data_units)) is None:
        return None
    dim_units = None
    for dim_name in dims:
        dim_unit = get_quantity(patch.get_coord(dim_name).units)
        if dim_unit is None:
            continue
        dim_units = dim_unit if dim_units is None else dim_unit * dim_units
    if dim_units is not None:
        data_units = operator(data_units, dim_units)
    return data_units


def get_taper_slices_local(patch: PatchType, kwargs: dict[str, Any]):
    dim = next(key for key in kwargs if key in patch.dims)
    axis = patch.get_axis(dim)
    value = kwargs[dim]
    coord = patch.coords.coord_map[dim]
    if isinstance(value, Sequence | np.ndarray):
        if len(value) != 2:
            raise AssertionError("Length 2 sequence required.")
        start, stop = value[0], value[1]
    else:
        start, stop = value, value
    dur = coord.coord_range(extend=False)
    clses = (Quantity, np.timedelta64)
    start = start if isinstance(start, clses) or start is None else start * dur
    stop = stop if isinstance(stop, clses) or stop is None else stop * dur
    stop = -stop if stop is not None else stop
    _, inds_1 = coord.select((None, start), relative=True)
    _, inds_2 = coord.select((stop, None), relative=True)
    return axis, (start, stop), inds_1, inds_2


def validate_windows_local(samps, start_slice, end_slice, shape, axis):
    max_len = shape[axis]
    start_ind = start_slice.stop
    end_ind = end_slice.start
    bad_start = samps[0] is not None and (start_ind is None or start_ind < 0)
    bad_end = samps[1] is not None and (end_ind is None or end_ind > max_len)
    if bad_start or bad_end:
        raise ParameterError("Total taper lengths exceed total dim length")
    if start_ind is None or end_ind is None:
        return
    if start_ind > end_ind:
        raise ParameterError("Taper windows cannot overlap")


def get_taper_coord_inds_local(coord, values, relative, samples):
    error_msg = "A len 2 or 4 sequence is required for taper values"
    if not isinstance(values, (Sequence | np.ndarray)) or not len(values):
        raise ParameterError(error_msg)
    if isinstance(values[0], (Sequence | np.ndarray)):
        out = [
            get_taper_coord_inds_local(coord, item, relative, samples)
            for item in values
        ]
        return reduce(add, out)
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
    if len(out) == 2:
        out = [0, *out, len(coord)]
    return [out]


def get_taper_curve_local(coord, ind_1, ind_2, window_type, reverse=False):
    taper = window_function(window_type, (ind_2 - ind_1) * 2 + 1)[: ind_2 - ind_1]
    if reverse:
        taper = taper[::-1]
    if not coord.evenly_sampled:
        old_coord = coord.select((ind_1, ind_2), samples=True)[0]
        new_coord = old_coord.snap().change_length(len(old_coord))
        old_x = dc.to_float(old_coord.values)
        new_x = dc.to_float(new_coord.values)
        taper = np.interp(new_x, old_x, taper)
    return taper


def get_range_envelope_local(coord, inds, window_type, invert):
    out = np.zeros(len(coord))
    for ind_set in inds:
        i1, i2, i3, i4 = ind_set
        left_taper = get_taper_curve_local(coord, i1, i2, window_type)
        right_taper = get_taper_curve_local(coord, i3, i4, window_type, reverse=True)
        out[i1:i2] += left_taper
        out[i3:i4] += right_taper
        out[i2:i3] += 1
    if invert:
        out = np.abs(out - np.max(out))
    return out


def validate_detrend_patch_input(
    patch: PatchType, dim: str, type: str = "linear"
) -> None:
    get_axis(patch, dim)
    detrend_type = kernels.validate_detrend_type(type)
    if detrend_type == "linear" and not kernels.is_finite_array(patch.data):
        raise ValueError("array must not contain infs or NaNs")


def detrend_patch(patch: PatchType, dim: str, type: str = "linear") -> PatchType:
    validate_detrend_patch_input(patch, dim=dim, type=type)
    axis = get_axis(patch, dim)
    return update_patch(patch, kernels.detrend_kernel(patch.data, axis=axis, type=type))


def detrend_leaves(
    data: Any,
    coord_leaves: tuple[Any, ...],
    *,
    dims: tuple[str, ...],
    dim: str,
    type: str = "linear",
) -> tuple[Any, tuple[Any, ...]]:
    axis = get_axis_from_dims(dims, dim)
    return kernels.detrend_kernel(data, axis=axis, type=type), coord_leaves


def normalize_patch(patch: PatchType, dim: str, norm: str = "l2") -> PatchType:
    axis = get_axis(patch, dim)
    return update_patch(
        patch, kernels.normalize_kernel(patch.data, axis=axis, norm=norm)
    )


def normalize_leaves(
    data: Any,
    coord_leaves: tuple[Any, ...],
    *,
    dims: tuple[str, ...],
    dim: str,
    norm: str = "l2",
) -> tuple[Any, tuple[Any, ...]]:
    axis = get_axis_from_dims(dims, dim)
    return kernels.normalize_kernel(data, axis=axis, norm=norm), coord_leaves


def differentiate_patch(
    patch: PatchType,
    dim: str | tuple[str, ...] | None,
    order: int = 2,
    step: int = 1,
) -> PatchType:
    return dc_differentiate.func(patch, dim=dim, order=order, step=step)


def integrate_patch(
    patch: PatchType,
    dim: tuple[str, ...] | str | None,
    definite: bool = False,
) -> PatchType:
    return dc_integrate.func(patch, dim=dim, definite=definite)


def prepare_taper_call(
    patch: PatchType,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    _ = args
    call_kwargs = dict(kwargs)
    window_type = call_kwargs.pop("window_type", "hann")
    axis, _, start_slice, end_slice = get_taper_slices_local(patch, call_kwargs)
    shape = patch.shape
    validate_windows_local(
        (start_slice.stop, shape[axis] - end_slice.start),
        start_slice,
        end_slice,
        shape,
        axis,
    )
    weight = np.ones(shape[axis], dtype=np.float64)
    if start_slice.stop is not None and start_slice.stop > 0:
        start_len = start_slice.stop
        weight[:start_len] = window_function(window_type, 2 * start_len)[:start_len]
    if end_slice.start is not None and end_slice.start < shape[axis]:
        end_len = shape[axis] - end_slice.start
        weight[end_slice.start :] = window_function(window_type, 2 * end_len)[end_len:]
    return (), {"axis": axis, "weight": weight}


def taper_patch(patch: PatchType, window_type: str = "hann", **kwargs) -> PatchType:
    return dc_taper.func(patch, window_type=window_type, **kwargs)


def taper_leaves(
    data: Any,
    coord_leaves: tuple[Any, ...],
    *,
    dims: tuple[str, ...],
    axis: int,
    weight: np.ndarray,
) -> tuple[Any, tuple[Any, ...]]:
    _ = dims
    return kernels.apply_1d_weight_kernel(data, axis=axis, weight=weight), coord_leaves


def prepare_taper_range_call(
    patch: PatchType,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    _ = args
    call_kwargs = dict(kwargs)
    window_type = call_kwargs.pop("window_type", "hann")
    invert = bool(call_kwargs.pop("invert", False))
    relative = bool(call_kwargs.pop("relative", False))
    samples = bool(call_kwargs.pop("samples", False))
    dim = next(key for key in call_kwargs if key in patch.dims)
    axis = patch.get_axis(dim)
    coord = patch.get_coord(dim, require_sorted=True)
    inds = get_taper_coord_inds_local(coord, call_kwargs[dim], relative, samples)
    weight = get_range_envelope_local(coord, inds, window_type, invert)
    return (), {"axis": axis, "weight": np.asarray(weight)}


def taper_range_patch(
    patch: PatchType,
    window_type: str = "hann",
    invert: bool = False,
    relative: bool = False,
    samples: bool = False,
    **kwargs,
) -> PatchType:
    return dc_taper_range.func(
        patch,
        window_type=window_type,
        invert=invert,
        relative=relative,
        samples=samples,
        **kwargs,
    )


def taper_range_leaves(
    data: Any,
    coord_leaves: tuple[Any, ...],
    *,
    dims: tuple[str, ...],
    axis: int,
    weight: np.ndarray,
) -> tuple[Any, tuple[Any, ...]]:
    _ = dims
    return kernels.apply_1d_weight_kernel(data, axis=axis, weight=weight), coord_leaves
