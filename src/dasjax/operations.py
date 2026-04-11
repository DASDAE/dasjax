"""Operation specs for eager patch methods and compiled pipelines."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import dascore as dc
import numpy as np
from dascore.constants import PatchType
from dascore.exceptions import ParameterError
from dascore.transform.fourier import _get_stft_coords
from dascore.units import get_filter_units
from dascore.utils.misc import check_filter_kwargs
from dascore.utils.patch import (
    get_dim_axis_value,
    get_dim_sampling_rate,
    get_patch_window_size,
)
from scipy.signal import ShortTimeFFT, get_window

from . import kernels

LeafTransform = Callable[..., tuple[Any, tuple[Any, ...]]]
PatchValidator = Callable[..., None]
CompiledPatchValidator = Callable[[PatchType, tuple[Any, ...], dict[str, Any]], None]
CompiledFallbackReason = Callable[
    [PatchType, tuple[Any, ...], dict[str, Any]], str | None
]
CompiledCaseGuard = Callable[
    [PatchType, tuple[Any, ...], dict[str, Any]],
    tuple[type[Exception], str] | None,
]
CallResolver = Callable[[PatchType], tuple[tuple[Any, ...], dict[str, Any]]]
Baseline = Callable[[PatchType, tuple[Any, ...], dict[str, Any]], PatchType]
PatchPreparer = Callable[[PatchType], PatchType]
CallPreparer = Callable[
    [PatchType, tuple[Any, ...], dict[str, Any]],
    tuple[tuple[Any, ...], dict[str, Any]],
]


def _identity_prepare(patch: PatchType) -> PatchType:
    return patch


def _empty_call(_: PatchType) -> tuple[tuple[Any, ...], dict[str, Any]]:
    return (), {}


@dataclass(frozen=True)
class OperationCase:
    """One shared test case for eager and compiled parity."""

    case_id: str
    resolve_call: CallResolver
    baseline: Baseline
    prepare_patch: PatchPreparer = _identity_prepare
    compiled_baseline: Baseline | None = None
    compiled_error: type[Exception] | None = None
    compiled_error_message: str | None = None
    compiled_guard: CompiledCaseGuard | None = None


@dataclass(frozen=True)
class JaxPatchOperation:
    """Read-only operation spec for patch namespace and pipeline support."""

    name: str
    patch_impl: Callable[..., PatchType]
    leaf_transform: LeafTransform
    validate_patch: PatchValidator | None = None
    validate_compiled_patch: CompiledPatchValidator | None = None
    compiled_fallback_reason: CompiledFallbackReason | None = None
    prepare_call: CallPreparer | None = None
    shape_changing: bool = False
    test_cases: tuple[OperationCase, ...] = field(default_factory=tuple)


_OPERATIONS: dict[str, JaxPatchOperation] = {}


def _register_operation(operation: JaxPatchOperation) -> JaxPatchOperation:
    _OPERATIONS[operation.name] = operation
    return operation


def get_operation(name: str) -> JaxPatchOperation:
    """Return one registered operation."""
    try:
        return _OPERATIONS[name]
    except KeyError as exc:
        msg = f"Unknown jax patch operation {name!r}."
        raise AttributeError(msg) from exc


def iter_operations() -> tuple[JaxPatchOperation, ...]:
    """Return the registered operation specs in registration order."""
    return tuple(_OPERATIONS.values())


def list_operations() -> tuple[str, ...]:
    """Return the public operation names."""
    return tuple(_OPERATIONS)


def _get_axis(patch: PatchType, dim: str | None) -> int | None:
    if dim is None:
        return None
    try:
        return patch.dims.index(dim)
    except ValueError as exc:
        msg = f"Patch does not contain dimension {dim!r}."
        raise ValueError(msg) from exc


def _get_axis_from_dims(dims: tuple[str, ...], dim: str | None) -> int | None:
    if dim is None:
        return None
    try:
        return dims.index(dim)
    except ValueError as exc:
        msg = f"Patch does not contain dimension {dim!r}."
        raise ValueError(msg) from exc


def _update_patch(
    patch: PatchType,
    data: Any,
) -> PatchType:
    return patch.update(data=np.asarray(data))


def _prepare_operation_call(
    operation: JaxPatchOperation,
    patch: PatchType,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if operation.prepare_call is None:
        return args, kwargs
    return operation.prepare_call(patch, args, kwargs)


def _identity_patch(patch: PatchType) -> PatchType:
    return patch


def _identity_leaves(
    data: Any,
    coord_leaves: tuple[Any, ...],
    *,
    dims: tuple[str, ...],
) -> tuple[Any, tuple[Any, ...]]:
    return kernels.identity_kernel(data), coord_leaves


def _scale_patch(patch: PatchType, factor: float) -> PatchType:
    return _update_patch(patch, kernels.scale_kernel(patch.data, factor))


def _scale_leaves(
    data: Any,
    coord_leaves: tuple[Any, ...],
    factor: float,
    *,
    dims: tuple[str, ...],
) -> tuple[Any, tuple[Any, ...]]:
    return kernels.scale_kernel(data, factor), coord_leaves


def _add_patch(patch: PatchType, value: float) -> PatchType:
    return _update_patch(patch, kernels.add_kernel(patch.data, value))


def _add_leaves(
    data: Any,
    coord_leaves: tuple[Any, ...],
    value: float,
    *,
    dims: tuple[str, ...],
) -> tuple[Any, tuple[Any, ...]]:
    return kernels.add_kernel(data, value), coord_leaves


def _abs_patch(patch: PatchType) -> PatchType:
    return _update_patch(patch, kernels.abs_kernel(patch.data))


def _abs_leaves(
    data: Any,
    coord_leaves: tuple[Any, ...],
    *,
    dims: tuple[str, ...],
) -> tuple[Any, tuple[Any, ...]]:
    return kernels.abs_kernel(data), coord_leaves


def _clip_patch(patch: PatchType, min_value: float, max_value: float) -> PatchType:
    return _update_patch(patch, kernels.clip_kernel(patch.data, min_value, max_value))


def _clip_leaves(
    data: Any,
    coord_leaves: tuple[Any, ...],
    min_value: float,
    max_value: float,
    *,
    dims: tuple[str, ...],
) -> tuple[Any, tuple[Any, ...]]:
    return kernels.clip_kernel(data, min_value, max_value), coord_leaves


def _validate_detrend_patch_input(
    patch: PatchType,
    dim: str,
    type: str = "linear",
) -> None:
    _get_axis(patch, dim)
    detrend_type = kernels.validate_detrend_type(type)
    if detrend_type == "linear" and not kernels.is_finite_array(patch.data):
        raise ValueError("array must not contain infs or NaNs")


def _detrend_patch(
    patch: PatchType,
    dim: str,
    type: str = "linear",
) -> PatchType:
    _validate_detrend_patch_input(patch, dim=dim, type=type)
    axis = _get_axis(patch, dim)
    return _update_patch(
        patch, kernels.detrend_kernel(patch.data, axis=axis, type=type)
    )


def _detrend_leaves(
    data: Any,
    coord_leaves: tuple[Any, ...],
    *,
    dims: tuple[str, ...],
    dim: str,
    type: str = "linear",
) -> tuple[Any, tuple[Any, ...]]:
    axis = _get_axis_from_dims(dims, dim)
    return kernels.detrend_kernel(data, axis=axis, type=type), coord_leaves


def _normalize_patch(
    patch: PatchType,
    dim: str,
    norm: str = "l2",
) -> PatchType:
    axis = _get_axis(patch, dim)
    return _update_patch(
        patch, kernels.normalize_kernel(patch.data, axis=axis, norm=norm)
    )


def _normalize_leaves(
    data: Any,
    coord_leaves: tuple[Any, ...],
    *,
    dims: tuple[str, ...],
    dim: str,
    norm: str = "l2",
) -> tuple[Any, tuple[Any, ...]]:
    axis = _get_axis_from_dims(dims, dim)
    return kernels.normalize_kernel(data, axis=axis, norm=norm), coord_leaves


def _offset_patch(patch: PatchType, offset: float) -> PatchType:
    return patch.update(data=patch.data + offset)


def _baseline_update(
    patch: PatchType,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    fn: Callable[..., Any],
) -> PatchType:
    return patch.update(data=np.asarray(fn(patch.data, *args, **kwargs)))


def _baseline_patch_method(
    patch: PatchType,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    method_name: str,
) -> PatchType:
    return getattr(patch, method_name)(*args, **kwargs)


def _prepare_gaussian_call(
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


def _prepare_hampel_call(
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
        patch,
        call_kwargs,
        samples,
        require_odd=True,
        warn_above=10,
        min_samples=3,
    )
    return (), {
        "size": tuple(int(x) for x in size),
        "threshold": threshold,
        "approximate": approximate,
    }


def _prepare_pass_filter_call(
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


def _prepare_fbe_call(
    patch: PatchType,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    _ = args
    call_kwargs = dict(kwargs)
    samples = bool(call_kwargs.pop("samples", False))
    detrend = bool(call_kwargs.pop("detrend", False))
    taper_window = call_kwargs.pop("taper_window", "hann")
    overlap = call_kwargs.pop("overlap", 0)
    fmin = call_kwargs.pop("fmin", None)
    fmax = call_kwargs.pop("fmax", None)
    (dim, axis, val) = get_dim_axis_value(patch, kwargs=call_kwargs)[0]

    def stft():
        return patch.get_coord(dim, require_evenly_sampled=True)

    coord = stft()
    window_samples = coord.get_sample_count(val, samples=samples, enforce_lt_coord=True)
    step = dc.to_float(coord.step)
    sampling_rate = 1 / abs(step)
    if isinstance(taper_window, np.ndarray):
        window = taper_window
    else:
        window = get_window(taper_window, window_samples, fftbins=False)
    if overlap is not None:
        overlap = coord[:window_samples].get_sample_count(
            overlap,
            samples=samples,
            enforce_lt_coord=True,
        )
    else:
        overlap = 0
    hop = window_samples - overlap
    stft = ShortTimeFFT(
        win=window,
        hop=hop,
        fs=sampling_rate,
        fft_mode="onesided",
        mfft=window_samples,
    )
    frame_times = np.asarray(stft.t(len(coord)))
    frame_starts = (
        np.rint(frame_times * sampling_rate).astype(np.int64) - stft.m_num_mid
    )
    frequencies = np.asarray(stft.f, dtype=np.float64)
    mask = np.ones(len(frequencies), dtype=bool)
    if fmin is not None:
        mask &= frequencies >= fmin
    if fmax is not None:
        mask &= frequencies <= fmax
    selected_bins = np.flatnonzero(mask).astype(np.int64)
    return (), {
        "axis": axis,
        "window": np.asarray(window),
        "hop": hop,
        "frame_starts": frame_starts,
        "frame_times": frame_times,
        "selected_bins": selected_bins,
        "sample_step": step,
        "detrend": detrend,
        "dim": dim,
        "window_samples": window_samples,
        "sampling_rate": sampling_rate,
        "stft_obj": stft,
    }


def _gaussian_filter_patch(
    patch: PatchType,
    samples: bool = False,
    mode: str = "reflect",
    cval: float = 0.0,
    truncate: float = 4.0,
    **kwargs,
) -> PatchType:
    operation = get_operation("gaussian_filter")
    _, prepared = _prepare_operation_call(
        operation,
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
    return _update_patch(patch, kernels.gaussian_filter_kernel(patch.data, **prepared))


def _gaussian_filter_leaves(
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
    return (
        kernels.gaussian_filter_kernel(
            data,
            sigma=sigma,
            axes=axes,
            mode=mode,
            cval=cval,
            truncate=truncate,
        ),
        coord_leaves,
    )


def _validate_hampel_filter_patch_input(
    patch: PatchType,
    threshold: float = 10.0,
    **kwargs,
) -> None:
    _ = patch, kwargs
    if threshold <= 0 or not np.isfinite(threshold):
        raise ParameterError(
            "hampel_filter threshold must be finite and greater than zero"
        )


def _validate_hampel_filter_compiled_input(
    patch: PatchType,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    _validate_hampel_filter_patch_input(patch, **kwargs)
    if not kwargs.get("approximate", True):
        msg = "Compiled hampel_filter currently requires approximate=True."
        raise NotImplementedError(msg)
    if not kernels.is_finite_array(patch.data):
        msg = "Compiled hampel_filter currently requires finite input data."
        raise NotImplementedError(msg)


def _fallback_gaussian_filter_reason(
    patch: PatchType,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> str | None:
    _ = patch, args, kwargs
    return "gaussian_filter uses a host callback in compiled pipelines."


def _fallback_hampel_filter_reason(
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


def _guard_compiled_hampel_case(
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


def _hampel_filter_patch(
    patch: PatchType,
    *,
    threshold: float = 10.0,
    samples: bool = False,
    approximate: bool = True,
    **kwargs,
) -> PatchType:
    operation = get_operation("hampel_filter")
    _validate_hampel_filter_patch_input(patch, threshold=threshold)
    if approximate and kernels.is_finite_array(patch.data):
        _, prepared = _prepare_operation_call(
            operation,
            patch,
            (),
            {
                "threshold": threshold,
                "samples": samples,
                "approximate": approximate,
                **kwargs,
            },
        )
        return _update_patch(
            patch, kernels.hampel_filter_kernel(patch.data, **prepared)
        )
    _, prepared = _prepare_operation_call(
        operation,
        patch,
        (),
        {
            "threshold": threshold,
            "samples": samples,
            "approximate": approximate,
            **kwargs,
        },
    )
    return _update_patch(
        patch,
        kernels.hampel_filter_callback_kernel(patch.data, **prepared),
    )


def _hampel_filter_leaves(
    data: Any,
    coord_leaves: tuple[Any, ...],
    *,
    dims: tuple[str, ...],
    size: tuple[int, ...],
    threshold: float,
    approximate: bool = True,
) -> tuple[Any, tuple[Any, ...]]:
    _ = dims
    return (
        kernels.hampel_filter_kernel(
            data,
            size=size,
            threshold=threshold,
            approximate=approximate,
        ),
        coord_leaves,
    )


def _pass_filter_patch(
    patch: PatchType,
    corners: int = 4,
    zerophase: bool = True,
    **kwargs,
) -> PatchType:
    operation = get_operation("pass_filter")
    _, prepared = _prepare_operation_call(
        operation,
        patch,
        (),
        {"corners": corners, "zerophase": zerophase, **kwargs},
    )
    return _update_patch(patch, kernels.pass_filter_kernel(patch.data, **prepared))


def _pass_filter_leaves(
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
    return (
        kernels.pass_filter_kernel(
            data,
            sos=sos,
            zi=zi,
            padlen=padlen,
            axis=axis,
            zerophase=zerophase,
        ),
        coord_leaves,
    )


def _fbe_patch(
    patch: PatchType,
    overlap: Any = 0,
    samples: bool = False,
    detrend: bool = False,
    taper_window: str | np.ndarray | tuple = "hann",
    fmin: float | None = None,
    fmax: float | None = None,
    **kwargs,
) -> PatchType:
    operation = get_operation("fbe")
    _, prepared = _prepare_operation_call(
        operation,
        patch,
        (),
        {
            "overlap": overlap,
            "samples": samples,
            "detrend": detrend,
            "taper_window": taper_window,
            "fmin": fmin,
            "fmax": fmax,
            **kwargs,
        },
    )
    reduced = kernels.banded_stft_kernel(
        patch.data,
        axis=prepared["axis"],
        window=prepared["window"],
        frame_starts=prepared["frame_starts"],
        selected_bins=prepared["selected_bins"],
        sample_step=prepared["sample_step"],
        detrend=prepared["detrend"],
    )
    coord = patch.get_coord(prepared["dim"], require_evenly_sampled=True)
    stft_cm = _get_stft_coords(
        patch,
        prepared["dim"],
        prepared["axis"],
        coord,
        prepared["stft_obj"],
        prepared["window"],
    )
    ft_dim = stft_cm.dims[prepared["axis"]]
    coord_map = {
        name: value
        for name, value in stft_cm.get_coord_tuple_map().items()
        if name != ft_dim
    }
    dims = tuple(dim for dim in stft_cm.dims if dim != ft_dim)
    return dc.Patch(
        data=np.asarray(reduced), coords=coord_map, dims=dims, attrs=patch.attrs
    )


def _fbe_leaves(
    data: Any,
    coord_leaves: tuple[Any, ...],
    *,
    dims: tuple[str, ...],
    **kwargs,
) -> tuple[Any, tuple[Any, ...]]:
    _ = data, coord_leaves, dims, kwargs
    raise NotImplementedError(
        "fbe cannot run as a leaf transform: it changes the patch shape. "
        "In compiled pipelines, fbe executes via patch_impl (the eager path)."
    )


def _baseline_fbe_patch(
    patch: PatchType,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> PatchType:
    call_kwargs = dict(kwargs)
    fmin = call_kwargs.pop("fmin", None)
    fmax = call_kwargs.pop("fmax", None)
    out = patch.stft(**call_kwargs).abs()
    ft_dim = next(dim for dim in out.dims if dim.startswith("ft_"))
    if fmin is not None or fmax is not None:
        out = out.select(**{ft_dim: (fmin, fmax)})
    return out.sum(dim=ft_dim, dim_reduce="squeeze")


_register_operation(
    JaxPatchOperation(
        name="identity",
        patch_impl=dc.patch_function(history=None)(_identity_patch),
        leaf_transform=_identity_leaves,
        test_cases=(
            OperationCase(
                case_id="identity",
                resolve_call=_empty_call,
                baseline=lambda patch, args, kwargs: patch,
            ),
        ),
    )
)

_register_operation(
    JaxPatchOperation(
        name="scale",
        patch_impl=dc.patch_function(history="method_name")(_scale_patch),
        leaf_transform=_scale_leaves,
        test_cases=tuple(
            OperationCase(
                case_id=f"scale-{value}",
                resolve_call=lambda patch, value=value: ((value,), {}),
                baseline=lambda patch, args, kwargs: patch * args[0],
            )
            for value in (2.0, -0.75)
        ),
    )
)

_register_operation(
    JaxPatchOperation(
        name="add",
        patch_impl=dc.patch_function(history="method_name")(_add_patch),
        leaf_transform=_add_leaves,
        test_cases=tuple(
            OperationCase(
                case_id=f"add-{value}",
                resolve_call=lambda patch, value=value: ((value,), {}),
                baseline=lambda patch, args, kwargs: patch + args[0],
            )
            for value in (2.0, -0.75)
        ),
    )
)

_register_operation(
    JaxPatchOperation(
        name="abs",
        patch_impl=dc.patch_function(history="method_name")(_abs_patch),
        leaf_transform=_abs_leaves,
        test_cases=(
            OperationCase(
                case_id="abs",
                resolve_call=_empty_call,
                prepare_patch=lambda patch: _offset_patch(patch, -0.5),
                baseline=lambda patch, args, kwargs: patch.abs(),
            ),
        ),
    )
)

_register_operation(
    JaxPatchOperation(
        name="clip",
        patch_impl=dc.patch_function(history="method_name")(_clip_patch),
        leaf_transform=_clip_leaves,
        test_cases=tuple(
            OperationCase(
                case_id=f"clip-{low}-{high}",
                resolve_call=lambda patch, low=low, high=high: ((low, high), {}),
                baseline=lambda patch, args, kwargs: _baseline_update(
                    patch,
                    args,
                    kwargs,
                    lambda data, min_value, max_value: np.clip(
                        data, min_value, max_value
                    ),
                ),
            )
            for low, high in ((-0.25, 0.25), (-1.0, 1.5))
        ),
    )
)

_register_operation(
    JaxPatchOperation(
        name="detrend",
        patch_impl=dc.patch_function(history="method_name")(_detrend_patch),
        leaf_transform=_detrend_leaves,
        validate_patch=_validate_detrend_patch_input,
        test_cases=tuple(
            OperationCase(
                case_id=f"detrend-{kind}",
                resolve_call=lambda patch, kind=kind: (
                    (),
                    {"dim": patch.dims[-1], "type": kind},
                ),
                baseline=lambda patch, args, kwargs: _baseline_patch_method(
                    patch, args, kwargs, "detrend"
                ),
            )
            for kind in ("constant", "linear")
        ),
    )
)

_register_operation(
    JaxPatchOperation(
        name="normalize",
        patch_impl=dc.patch_function(history="method_name")(_normalize_patch),
        leaf_transform=_normalize_leaves,
        test_cases=tuple(
            OperationCase(
                case_id=f"normalize-{norm}",
                resolve_call=lambda patch, norm=norm: (
                    (),
                    {"dim": patch.dims[-1], "norm": norm},
                ),
                baseline=lambda patch, args, kwargs: _baseline_patch_method(
                    patch, args, kwargs, "normalize"
                ),
            )
            for norm in ("l1", "l2", "max", "bit")
        ),
    )
)

_register_operation(
    JaxPatchOperation(
        name="gaussian_filter",
        patch_impl=dc.patch_function(history="method_name")(_gaussian_filter_patch),
        leaf_transform=_gaussian_filter_leaves,
        compiled_fallback_reason=_fallback_gaussian_filter_reason,
        prepare_call=_prepare_gaussian_call,
        test_cases=(
            OperationCase(
                case_id="gaussian-filter-default",
                resolve_call=lambda patch: (
                    (),
                    {
                        patch.dims[-1]: 3,
                        "samples": True,
                    },
                ),
                baseline=lambda patch, args, kwargs: _baseline_patch_method(
                    patch, args, kwargs, "gaussian_filter"
                ),
            ),
            OperationCase(
                case_id="gaussian-filter-constant",
                resolve_call=lambda patch: (
                    (),
                    {
                        patch.dims[0]: 3,
                        "samples": True,
                        "mode": "constant",
                        "cval": 0.25,
                    },
                ),
                baseline=lambda patch, args, kwargs: _baseline_patch_method(
                    patch, args, kwargs, "gaussian_filter"
                ),
            ),
        ),
    )
)

_register_operation(
    JaxPatchOperation(
        name="hampel_filter",
        patch_impl=dc.patch_function(history="method_name")(_hampel_filter_patch),
        leaf_transform=_hampel_filter_leaves,
        validate_patch=_validate_hampel_filter_patch_input,
        validate_compiled_patch=_validate_hampel_filter_compiled_input,
        compiled_fallback_reason=_fallback_hampel_filter_reason,
        prepare_call=_prepare_hampel_call,
        test_cases=(
            OperationCase(
                case_id="hampel-filter-approximate",
                resolve_call=lambda patch: (
                    (),
                    {
                        patch.dims[-1]: 3,
                        "samples": True,
                        "threshold": 3.5,
                        "approximate": True,
                    },
                ),
                baseline=lambda patch, args, kwargs: _baseline_patch_method(
                    patch, args, kwargs, "hampel_filter"
                ),
                compiled_guard=_guard_compiled_hampel_case,
            ),
            OperationCase(
                case_id="hampel-filter-exact",
                resolve_call=lambda patch: (
                    (),
                    {
                        patch.dims[-1]: 3,
                        "samples": True,
                        "threshold": 3.5,
                        "approximate": False,
                    },
                ),
                baseline=lambda patch, args, kwargs: _baseline_patch_method(
                    patch, args, kwargs, "hampel_filter"
                ),
                compiled_guard=_guard_compiled_hampel_case,
            ),
        ),
    )
)

_register_operation(
    JaxPatchOperation(
        name="pass_filter",
        patch_impl=dc.patch_function(history="method_name")(_pass_filter_patch),
        leaf_transform=_pass_filter_leaves,
        prepare_call=_prepare_pass_filter_call,
        test_cases=(
            OperationCase(
                case_id="pass-filter-bandpass",
                resolve_call=lambda patch: (
                    (),
                    {
                        patch.dims[-1]: (1.0, 10.0),
                        "corners": 4,
                        "zerophase": True,
                    },
                ),
                baseline=lambda patch, args, kwargs: _baseline_patch_method(
                    patch, args, kwargs, "pass_filter"
                ),
            ),
            OperationCase(
                case_id="pass-filter-lowpass",
                resolve_call=lambda patch: (
                    (),
                    {
                        patch.dims[-1]: (None, 12.0),
                        "corners": 2,
                        "zerophase": False,
                    },
                ),
                baseline=lambda patch, args, kwargs: _baseline_patch_method(
                    patch, args, kwargs, "pass_filter"
                ),
            ),
        ),
    )
)

_register_operation(
    JaxPatchOperation(
        name="fbe",
        patch_impl=dc.patch_function(history="method_name")(_fbe_patch),
        leaf_transform=_fbe_leaves,
        prepare_call=_prepare_fbe_call,
        shape_changing=True,
        test_cases=(
            OperationCase(
                case_id="fbe-default",
                resolve_call=lambda patch: (
                    (),
                    {
                        patch.dims[-1]: 64,
                        "samples": True,
                        "overlap": 32,
                        "fmin": 2.0,
                        "fmax": 10.0,
                    },
                ),
                baseline=_baseline_fbe_patch,
            ),
        ),
    )
)
