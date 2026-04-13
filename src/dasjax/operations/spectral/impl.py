"""Executable implementations for spectral operations."""

from __future__ import annotations

from typing import Any

import dascore as dc
import numpy as np
from dascore.constants import PatchType
from dascore.exceptions import ParameterError
from dascore.proc.basic import pad as dc_pad
from dascore.proc.whiten import whiten as dc_whiten
from dascore.transform.fourier import dft as dc_dft
from dascore.transform.fourier import idft as dc_idft
from dascore.transform.hilbert import envelope as dc_envelope
from dascore.transform.hilbert import hilbert as dc_hilbert
from dascore.units import invert_quantity
from dascore.utils.patch import get_dim_axis_value
from dascore.utils.time import is_datetime64, is_timedelta64
from dascore.utils.transformatter import FourierTransformatter
from scipy.signal import ShortTimeFFT, get_window

from dasjax import kernels

from ..common import get_axis_from_dims


def get_dim_freq_range_from_kwargs_local(
    patch: PatchType,
    kwargs: dict[str, Any],
) -> tuple[str, Any]:
    dim_set = set(patch.dims)
    if not kwargs:
        expected = {"time", "ft_time"} & dim_set
        if not expected:
            raise ParameterError(
                "No dim name provided in kwargs and patch has no time dimension."
            )
        return "time", None
    if len(kwargs) == 1:
        dim, freq_range = next(iter(kwargs.items()))
        fft_dim = FourierTransformatter().rename_dims(dim)[0]
        if dim not in dim_set and fft_dim not in dim_set:
            raise ParameterError(
                f"passed dim of {dim} to whiten but it is not in patch dimensions."
            )
        return dim, freq_range
    raise ParameterError(
        "Whiten kwargs must specify a single patch dimension. "
        f"You passed {kwargs}."
    )


def check_smooth_local(fft_coord, smooth_size, water_level):
    if water_level is not None:
        if not isinstance(water_level, float) or water_level < 0 or water_level > 1:
            raise ParameterError("water_level must be a float between 0 and 1.")
    if smooth_size <= 0:
        raise ParameterError("Frequency smoothing size must be positive")
    if smooth_size >= fft_coord.max():
        raise ParameterError("Frequency smoothing size is larger than Nyquist")


def check_freq_range_local(fft_coord, freq_range):
    frange = np.asarray(freq_range)
    diffs = frange[1:] - frange[:-1]
    min_size = fft_coord.step * 2
    if np.any(diffs < min_size):
        raise ParameterError("Frequency range is too narrow")


def get_stft_coords_local(patch, dim, axis, coord, stft, window):
    ft = FourierTransformatter()
    time = stft.t(len(coord))
    if is_datetime64(coord.dtype) or is_timedelta64(coord.dtype):
        time = dc.to_timedelta64(time)
    new_dims = list(ft.rename_dims(patch.dims, index=axis, forward=True))
    new_dims.append(dim)
    coord_map = patch.coords.disassociate_coord(dim).get_coord_tuple_map()
    new_units = invert_quantity(coord.units)
    coord_map.update(
        {
            dim: dc.get_coord(values=time + coord.min(), units=coord.units),
            new_dims[axis]: dc.get_coord(values=stft.f, units=new_units),
            "_stft_window": (None, window),
            "_stft_old_coord": (None, patch.get_coord(dim)),
        }
    )
    return dc.get_coord_manager(coords=coord_map, dims=tuple(new_dims))


def pad_patch(
    patch: PatchType,
    mode: str = "constant",
    constant_values: Any = 0,
    expand_coords: bool = True,
    samples: bool = False,
    **kwargs,
) -> PatchType:
    return dc_pad.func(
        patch,
        mode=mode,
        constant_values=constant_values,
        expand_coords=expand_coords,
        samples=samples,
        **kwargs,
    )


def dft_patch(
    patch: PatchType,
    dim: str | None | tuple[str, ...],
    *,
    real: str | bool | None = None,
    pad: bool = True,
) -> PatchType:
    return dc_dft.func(patch, dim=dim, real=real, pad=pad)


def idft_patch(
    patch: PatchType,
    dim: str | None | tuple[str, ...] = None,
) -> PatchType:
    return dc_idft.func(patch, dim=dim)


def hilbert_patch(patch: PatchType, dim: str) -> PatchType:
    return dc_hilbert.func(patch, dim=dim)


def hilbert_leaves(
    data: Any,
    coord_leaves: tuple[Any, ...],
    *,
    dims: tuple[str, ...],
    dim: str,
) -> tuple[Any, tuple[Any, ...]]:
    axis = get_axis_from_dims(dims, dim)
    return kernels.hilbert_kernel(data, axis=axis), coord_leaves


def envelope_patch(patch: PatchType, dim: str) -> PatchType:
    return dc_envelope.func(patch, dim=dim)


def envelope_leaves(
    data: Any,
    coord_leaves: tuple[Any, ...],
    *,
    dims: tuple[str, ...],
    dim: str,
) -> tuple[Any, tuple[Any, ...]]:
    axis = get_axis_from_dims(dims, dim)
    return kernels.envelope_kernel(data, axis=axis), coord_leaves


def whiten_patch(
    patch: PatchType,
    smooth_size: float | None = None,
    water_level: float | None = None,
    **kwargs,
) -> PatchType:
    return dc_whiten.func(patch, smooth_size=smooth_size, water_level=water_level, **kwargs)


def prepare_fbe_call(
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
    window = taper_window if isinstance(taper_window, np.ndarray) else get_window(taper_window, window_samples, fftbins=False)
    if overlap is not None:
        overlap = coord[:window_samples].get_sample_count(overlap, samples=samples, enforce_lt_coord=True)
    else:
        overlap = 0
    hop = window_samples - overlap
    stft = ShortTimeFFT(win=window, hop=hop, fs=sampling_rate, fft_mode="onesided", mfft=window_samples)
    frame_times = np.asarray(stft.t(len(coord)))
    frame_starts = np.rint(frame_times * sampling_rate).astype(np.int64) - stft.m_num_mid
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


def fbe_patch(
    patch: PatchType,
    overlap: Any = 0,
    samples: bool = False,
    detrend: bool = False,
    taper_window: str | np.ndarray | tuple = "hann",
    fmin: float | None = None,
    fmax: float | None = None,
    **kwargs,
) -> PatchType:
    _, prepared = prepare_fbe_call(
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
    stft_cm = get_stft_coords_local(
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
    return dc.Patch(data=np.asarray(reduced), coords=coord_map, dims=dims, attrs=patch.attrs)


def baseline_fbe_patch(
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
