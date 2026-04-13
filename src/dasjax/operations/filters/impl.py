"""Executable implementations for filter operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from dascore.constants import PatchType
from dascore.exceptions import ParameterError
from dascore.units import get_filter_units
from dascore.utils.misc import check_filter_kwargs
from dascore.utils.patch import (
    get_dim_axis_value,
    get_dim_sampling_rate,
    get_patch_window_size,
)

from dasjax import kernels

from ..common import update_patch
from ..patch_ops import PatchOp


@dataclass(frozen=True)
class GaussianFilterOp(PatchOp):
    """Compiled gaussian_filter operation."""

    sigma: tuple[float, ...]
    axes: tuple[int, ...]
    mode: str = "reflect"
    cval: float = 0.0
    truncate: float = 4.0
    requires_materialized_patch_for_prepare = True

    @classmethod
    def prepare(
        cls,
        patch: PatchType,
        samples: bool = False,
        mode: str = "reflect",
        cval: float = 0.0,
        truncate: float = 4.0,
        **kwargs,
    ) -> "GaussianFilterOp":
        _, prepared = prepare_gaussian_call(
            patch,
            (),
            {
                "samples": samples,
                "mode": mode,
                "cval": cval,
                "truncate": truncate,
                **kwargs,
            },
        )
        return cls(**prepared)

    def kernel(self, data):
        return {
            "data": kernels.gaussian_filter_kernel(
                data,
                sigma=self.sigma,
                axes=self.axes,
                mode=self.mode,
                cval=self.cval,
                truncate=self.truncate,
            )
        }


@dataclass(frozen=True)
class HampelFilterOp(PatchOp):
    """Compiled hampel_filter operation."""

    size: tuple[int, ...]
    threshold: float
    approximate: bool = True
    requires_materialized_patch_for_prepare = True

    @classmethod
    def prepare(
        cls,
        patch: PatchType,
        *,
        threshold: float = 10.0,
        samples: bool = False,
        approximate: bool = True,
        **kwargs,
    ) -> "HampelFilterOp":
        validate_hampel_filter_patch_input(patch, threshold=threshold)
        validate_hampel_filter_compiled_input(
            patch,
            (),
            {
                "threshold": threshold,
                "samples": samples,
                "approximate": approximate,
                **kwargs,
            },
        )
        _, prepared = prepare_hampel_call(
            patch,
            (),
            {
                "threshold": threshold,
                "samples": samples,
                "approximate": approximate,
                **kwargs,
            },
        )
        return cls(**prepared)

    def kernel(self, data):
        return {
            "data": kernels.hampel_filter_kernel(
                data,
                size=self.size,
                threshold=self.threshold,
                approximate=self.approximate,
            )
        }


@dataclass(frozen=True)
class PassFilterOp(PatchOp):
    """Compiled pass_filter operation."""

    sos: Any
    zi: Any
    padlen: int
    axis: int
    zerophase: bool = True
    requires_materialized_patch_for_prepare = True

    @classmethod
    def prepare(
        cls,
        patch: PatchType,
        corners: int = 4,
        zerophase: bool = True,
        **kwargs,
    ) -> "PassFilterOp":
        _, prepared = prepare_pass_filter_call(
            patch,
            (),
            {"corners": corners, "zerophase": zerophase, **kwargs},
        )
        return cls(**prepared)

    def kernel(self, data):
        return {
            "data": kernels.pass_filter_kernel(
                data,
                sos=self.sos,
                zi=self.zi,
                padlen=self.padlen,
                axis=self.axis,
                zerophase=self.zerophase,
            )
        }


def prepare_gaussian_call(
    patch: PatchType,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    _ = args
    call_kwargs = dict(kwargs)
    samples = call_kwargs.pop("samples", False)
    mode = call_kwargs.pop("mode", "reflect")
    cval = float(call_kwargs.pop("cval", 0.0))
    truncate = float(call_kwargs.pop("truncate", 4.0))
    dimfo = get_dim_axis_value(patch, kwargs=call_kwargs, allow_multiple=True)
    axes = []
    sigma = []
    for dim, axis, value in dimfo:
        coord = patch.get_coord(dim)
        sigma.append(float(coord.get_sample_count(value, samples=samples)))
        axes.append(axis)
    return (), {
        "sigma": tuple(sigma),
        "axes": tuple(axes),
        "mode": mode,
        "cval": cval,
        "truncate": truncate,
    }


def gaussian_filter_patch(
    patch: PatchType,
    samples: bool = False,
    mode: str = "reflect",
    cval: float = 0.0,
    truncate: float = 4.0,
    **kwargs,
) -> PatchType:
    _, prepared = prepare_gaussian_call(
        patch,
        (),
        {
            "samples": samples,
            "mode": mode,
            "cval": cval,
            "truncate": truncate,
            **kwargs,
        },
    )
    return update_patch(patch, kernels.gaussian_filter_kernel(patch.data, **prepared))


def gaussian_filter_leaves(
    data: Any,
    coord_leaves: tuple[Any, ...],
    *,
    dims: tuple[str, ...],
    sigma: tuple[float, ...],
    axes: tuple[int, ...],
    mode: str = "reflect",
    cval: float = 0.0,
    truncate: float = 4.0,
) -> tuple[Any, tuple[Any, ...]]:
    _ = dims
    return kernels.gaussian_filter_kernel(
        data, sigma=sigma, axes=axes, mode=mode, cval=cval, truncate=truncate
    ), coord_leaves


def validate_hampel_filter_patch_input(
    patch: PatchType,
    threshold: float = 10.0,
    **kwargs,
) -> None:
    _ = patch, kwargs
    if threshold <= 0 or not np.isfinite(threshold):
        raise ParameterError(
            "hampel_filter threshold must be finite and greater than zero"
        )


def validate_hampel_filter_compiled_input(
    patch: PatchType,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    validate_hampel_filter_patch_input(patch, **kwargs)
    if not kwargs.get("approximate", True):
        raise NotImplementedError(
            "Compiled hampel_filter currently requires approximate=True."
        )
    if not kernels.is_finite_array(patch.data):
        raise NotImplementedError(
            "Compiled hampel_filter currently requires finite input data."
        )


def fallback_gaussian_filter_reason(
    patch: PatchType,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> str | None:
    _ = patch, args, kwargs
    return "gaussian_filter uses a host callback in compiled pipelines."


def fallback_hampel_filter_reason(
    patch: PatchType,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> str | None:
    _ = args
    if not kwargs.get("approximate", True):
        return "hampel_filter(approximate=False) is not natively compiled."
    if not kernels.is_finite_array(patch.data):
        return "hampel_filter with non-finite input data is not natively compiled."
    return None


def guard_compiled_hampel_case(
    patch: PatchType,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[type[Exception], str] | None:
    _ = args
    if not kwargs.get("approximate", True):
        return (
            NotImplementedError,
            "Compiled hampel_filter currently requires approximate=True.",
        )
    if not kernels.is_finite_array(patch.data):
        return (
            NotImplementedError,
            "Compiled hampel_filter currently requires finite input data.",
        )
    return None


def prepare_hampel_call(
    patch: PatchType,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    _ = args
    call_kwargs = dict(kwargs)
    threshold = float(call_kwargs.pop("threshold", 10.0))
    samples = call_kwargs.pop("samples", False)
    approximate = bool(call_kwargs.pop("approximate", True))
    size = get_patch_window_size(
        patch, call_kwargs, samples, require_odd=True, warn_above=10, min_samples=3
    )
    return (), {
        "size": tuple(int(x) for x in size),
        "threshold": threshold,
        "approximate": approximate,
    }


def hampel_filter_patch(
    patch: PatchType,
    *,
    threshold: float = 10.0,
    samples: bool = False,
    approximate: bool = True,
    **kwargs,
) -> PatchType:
    validate_hampel_filter_patch_input(patch, threshold=threshold)
    _, prepared = prepare_hampel_call(
        patch,
        (),
        {
            "threshold": threshold,
            "samples": samples,
            "approximate": approximate,
            **kwargs,
        },
    )
    if approximate and kernels.is_finite_array(patch.data):
        return update_patch(patch, kernels.hampel_filter_kernel(patch.data, **prepared))
    return update_patch(
        patch, kernels.hampel_filter_callback_kernel(patch.data, **prepared)
    )


def hampel_filter_leaves(
    data: Any,
    coord_leaves: tuple[Any, ...],
    *,
    dims: tuple[str, ...],
    size: tuple[int, ...],
    threshold: float,
    approximate: bool = True,
) -> tuple[Any, tuple[Any, ...]]:
    _ = dims
    return kernels.hampel_filter_kernel(
        data, size=size, threshold=threshold, approximate=approximate
    ), coord_leaves


def prepare_pass_filter_call(
    patch: PatchType,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    _ = args
    call_kwargs = dict(kwargs)
    corners = int(call_kwargs.pop("corners", 4))
    zerophase = bool(call_kwargs.pop("zerophase", True))
    dim, (arg1, arg2) = check_filter_kwargs(call_kwargs)
    coord_units = patch.coords.coord_map[dim].units
    filt_min, filt_max = get_filter_units(arg1, arg2, to_unit=coord_units, dim=dim)

    def pass_filter() -> float:
        return get_dim_sampling_rate(patch, dim)

    sr = pass_filter()
    sos = kernels.design_pass_filter_sos(sr, filt_min, filt_max, corners)
    return (), {
        "sos": np.asarray(sos),
        "zi": kernels.pass_filter_initial_state(np.asarray(sos)),
        "padlen": kernels.pass_filter_default_padlen(np.asarray(sos)),
        "axis": patch.get_axis(dim),
        "zerophase": zerophase,
    }


def pass_filter_patch(
    patch: PatchType,
    corners: int = 4,
    zerophase: bool = True,
    **kwargs,
) -> PatchType:
    _, prepared = prepare_pass_filter_call(
        patch,
        (),
        {"corners": corners, "zerophase": zerophase, **kwargs},
    )
    return update_patch(patch, kernels.pass_filter_kernel(patch.data, **prepared))


def pass_filter_leaves(
    data: Any,
    coord_leaves: tuple[Any, ...],
    *,
    dims: tuple[str, ...],
    sos: Any,
    zi: Any,
    padlen: int,
    axis: int,
    zerophase: bool = True,
) -> tuple[Any, tuple[Any, ...]]:
    _ = dims
    return kernels.pass_filter_kernel(
        data, sos=sos, zi=zi, padlen=padlen, axis=axis, zerophase=zerophase
    ), coord_leaves
