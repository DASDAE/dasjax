"""Array-level JAX kernels shared by eager and compiled patch operations."""

from __future__ import annotations

import functools
from typing import Any

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy import ndimage
from dascore.exceptions import FilterValueError
from scipy.ndimage import median_filter as scipy_median_filter
from scipy.signal import iirfilter, sosfilt_zi, zpk2sos


def identity_kernel(data: Any) -> Any:
    """Return the input unchanged."""
    return data


def scale_kernel(data: Any, factor: float) -> Any:
    """Scale data by a constant factor."""
    return jnp.asarray(data) * factor


def add_kernel(data: Any, value: float) -> Any:
    """Add a constant value to the data."""
    return jnp.asarray(data) + value


def abs_kernel(data: Any) -> Any:
    """Take the absolute value of the data."""
    return jnp.abs(jnp.asarray(data))


def clip_kernel(data: Any, min_value: float, max_value: float) -> Any:
    """Clip data to the provided range."""
    return jnp.clip(jnp.asarray(data), min_value, max_value)


def validate_detrend_type(type: str) -> str:
    """Normalize supported detrend type aliases."""
    alias_map = {
        "linear": "linear",
        "l": "linear",
        "constant": "constant",
        "c": "constant",
    }
    try:
        return alias_map[type]
    except KeyError as exc:
        msg = "Trend type must be 'linear' or 'constant'."
        raise ValueError(msg) from exc


def detrend_kernel(data: Any, axis: int, type: str) -> Any:
    """Detrend an array along one axis using JAX-compatible math."""
    data = jnp.asarray(data)
    detrend_type = validate_detrend_type(type)
    if detrend_type == "constant":
        return data - jnp.mean(data, axis=axis, keepdims=True)

    moved = jnp.moveaxis(data, axis, 0)
    npts = moved.shape[0]
    reshaped = moved.reshape(npts, -1)
    dtype = reshaped.dtype
    ramp = jnp.arange(1, npts + 1, dtype=dtype) / npts
    ramp_centered = ramp - jnp.mean(ramp)
    mean = jnp.mean(reshaped, axis=0, keepdims=True)
    numerator = jnp.sum(ramp_centered[:, None] * (reshaped - mean), axis=0)
    denominator = jnp.sum(ramp_centered * ramp_centered)
    slope = jnp.where(denominator != 0, numerator / denominator, 0)
    trend = mean + ramp_centered[:, None] * slope[None, :]
    detrended = reshaped - trend
    restored = detrended.reshape(moved.shape)
    return jnp.moveaxis(restored, 0, axis)


def normalize_kernel(data: Any, axis: int, norm: str = "l2") -> Any:
    """Normalize an array along one axis using DASCore semantics."""
    data = jnp.asarray(data)
    if norm in {"l1", "l2"}:
        order = int(norm[-1])
        norm_values = jnp.linalg.norm(data, axis=axis, ord=order)
        expanded_norm = jnp.expand_dims(norm_values, axis=axis)
        return jnp.where(expanded_norm != 0, data / expanded_norm, 0)
    if norm == "max":
        norm_values = jnp.max(data, axis=axis)
        expanded_norm = jnp.expand_dims(norm_values, axis=axis)
        return jnp.where(expanded_norm != 0, data / expanded_norm, 0)
    if norm == "bit":
        abs_data = jnp.abs(data)
        return jnp.where(abs_data != 0, data / abs_data, 0)
    msg = f"Unsupported normalization mode {norm!r}."
    raise ValueError(msg)


def is_finite_array(data: Any) -> bool:
    """Return True when every value in the array is finite."""
    return bool(np.isfinite(np.asarray(data)).all())


def gaussian_filter_kernel(
    data: Any,
    sigma: tuple[float, ...],
    axes: tuple[int, ...],
    mode: str = "reflect",
    cval: float = 0.0,
    truncate: float = 4.0,
) -> Any:
    """Apply a Gaussian filter with SciPy-compatible semantics."""
    arr = jnp.asarray(data)
    shape_dtype = jax.ShapeDtypeStruct(arr.shape, arr.dtype)
    return jax.pure_callback(
        lambda x: ndimage.gaussian_filter(
            np.array(x, copy=True),
            sigma=sigma,
            mode=mode,
            cval=cval,
            truncate=truncate,
            axes=axes,
        ),
        shape_dtype,
        arr,
        vmap_method="sequential",
    )


def _sliding_windows_last_axis(data: jnp.ndarray, window: int) -> jnp.ndarray:
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = jnp.pad(data, ((0, 0), (pad_left, pad_right)), mode="edge")
    length = data.shape[-1]

    def _slice_at(idx: jnp.ndarray) -> jnp.ndarray:
        return lax.dynamic_slice_in_dim(padded, idx, window, axis=1)

    windows = jax.vmap(_slice_at)(jnp.arange(length))
    return jnp.swapaxes(windows, 0, 1)


def _median_filter_axis(data: jnp.ndarray, axis: int, window: int) -> jnp.ndarray:
    moved = jnp.moveaxis(data, axis, -1)
    flat = moved.reshape(-1, moved.shape[-1])
    windows = _sliding_windows_last_axis(flat, window)
    med = jnp.sort(windows, axis=-1)[..., window // 2]
    restored = med.reshape(moved.shape)
    return jnp.moveaxis(restored, -1, axis)


def _median_filter_exact_callback(
    data: np.ndarray, size: tuple[int, ...]
) -> np.ndarray:
    return scipy_median_filter(data, size=size, mode="reflect")


def _median_filter_exact(data: jnp.ndarray, size: tuple[int, ...]) -> jnp.ndarray:
    shape_dtype = jax.ShapeDtypeStruct(data.shape, data.dtype)
    return jax.pure_callback(
        lambda x: _median_filter_exact_callback(x, size),
        shape_dtype,
        data,
        vmap_method="sequential",
    )


def hampel_filter_callback_kernel(
    data: Any,
    size: tuple[int, ...],
    threshold: float,
    approximate: bool = True,
) -> Any:
    """Apply Hampel filtering via SciPy-compatible host callback."""
    source = jnp.asarray(data)
    shape_dtype = jax.ShapeDtypeStruct(source.shape, source.dtype)

    def _callback(x: np.ndarray) -> np.ndarray:
        original = np.asarray(x)
        is_int = original.dtype.kind in {"i", "u"}
        work = original.copy() if not is_int else original.astype(np.float32)
        if approximate:
            med = np.empty_like(work)
            np.copyto(med, work)
            for axis, window in enumerate(size):
                if window > 1:
                    filtered = scipy_median_filter(
                        med,
                        size=tuple(1 if i != axis else window for i in range(med.ndim)),
                        mode="reflect",
                    )
                    np.copyto(med, filtered)
            abs_diff = np.abs(work - med)
            mad = np.empty_like(abs_diff)
            np.copyto(mad, abs_diff)
            for axis, window in enumerate(size):
                if window > 1:
                    filtered = scipy_median_filter(
                        mad,
                        size=tuple(1 if i != axis else window for i in range(mad.ndim)),
                        mode="reflect",
                    )
                    np.copyto(mad, filtered)
        else:
            med = scipy_median_filter(work, size=size, mode="reflect")
            abs_diff = np.abs(work - med)
            mad = scipy_median_filter(abs_diff, size=size, mode="reflect")
        mad_safe = np.where(mad == 0.0, np.finfo(work.dtype).eps, mad)
        out = np.where(abs_diff / mad_safe > threshold, med, work)
        if is_int:
            out = np.rint(out)
        return out.astype(original.dtype, copy=False)

    return jax.pure_callback(_callback, shape_dtype, source, vmap_method="sequential")


def hampel_filter_kernel(
    data: Any,
    size: tuple[int, ...],
    threshold: float,
    approximate: bool = True,
) -> Any:
    """Apply a Hampel filter with a native JAX approximate path."""
    source = jnp.asarray(data)
    is_int = jnp.issubdtype(source.dtype, jnp.integer)
    compute_dtype = jnp.float32 if is_int else source.dtype
    work = source.astype(compute_dtype)
    if approximate:
        med = work
        for axis, window in enumerate(size):
            if window > 1:
                med = _median_filter_axis(med, axis, window)
        abs_diff = jnp.abs(work - med)
        mad = abs_diff
        for axis, window in enumerate(size):
            if window > 1:
                mad = _median_filter_axis(mad, axis, window)
    else:
        med = _median_filter_exact(work, size)
        abs_diff = jnp.abs(work - med)
        mad = _median_filter_exact(abs_diff, size)
    mad_safe = jnp.where(mad == 0.0, jnp.finfo(work.dtype).eps, mad)
    out = jnp.where(abs_diff / mad_safe > threshold, med, work)
    if is_int:
        out = jnp.rint(out).astype(source.dtype)
    return out


def design_pass_filter_sos(
    sr: float,
    filt_min: float | None,
    filt_max: float | None,
    corners: int,
) -> np.ndarray:
    """Design Butterworth SOS sections matching DASCore's pass_filter."""
    nyquist = 0.5 * sr
    low = None if pd.isnull(filt_min) else filt_min / nyquist
    high = None if pd.isnull(filt_max) else filt_max / nyquist
    if low is not None and ((0 > low) or (low > 1)):
        raise FilterValueError(
            f"possible filter bounds are [0, {nyquist}] you passed {filt_min}"
        )
    if high is not None and ((0 > high) or (high > 1)):
        raise FilterValueError(
            f"possible filter bounds are [0, {nyquist}] you passed {filt_max}"
        )
    if high is not None and low is not None and high <= low:
        raise FilterValueError(
            "Low filter param must be less than high filter param, "
            f"you passed:filt_min = {filt_min}, filt_max = {filt_max}"
        )
    if (low is not None) and (high is not None):
        z, p, k = iirfilter(
            corners, [low, high], btype="band", ftype="butter", output="zpk"
        )
    elif low is not None:
        z, p, k = iirfilter(
            corners, low, btype="highpass", ftype="butter", output="zpk"
        )
    elif high is not None:
        z, p, k = iirfilter(
            corners, high, btype="lowpass", ftype="butter", output="zpk"
        )
    else:
        msg = "At least one of filt_min or filt_max must be provided."
        raise FilterValueError(msg)
    return zpk2sos(z, p, k)


def pass_filter_initial_state(sos: np.ndarray) -> np.ndarray:
    """Return steady-state SOS initial conditions for filtfilt/sosfilt."""
    return np.asarray(sosfilt_zi(sos))


def pass_filter_default_padlen(sos: np.ndarray) -> int:
    """Match SciPy's default SOS filtfilt padding heuristic."""
    n_sections = len(sos)
    ntaps = 2 * n_sections + 1
    ntaps -= min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum())
    return 3 * ntaps


def _normalize_sos(sos: jnp.ndarray) -> jnp.ndarray:
    a0 = sos[:, 3:4]
    numer = sos[:, :3] / a0
    denom = sos[:, 4:] / a0
    return jnp.concatenate([numer, jnp.ones_like(a0), denom], axis=1)


def _sosfilt_rows(
    flat: jnp.ndarray,
    sos: jnp.ndarray,
    zi_rows: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    n_rows = flat.shape[0]
    if zi_rows is None:
        states = jnp.zeros((sos.shape[0], n_rows, 2), dtype=flat.dtype)
    else:
        states = jnp.moveaxis(zi_rows, 0, 1)

    def _sample_step(carry: jnp.ndarray, x_t: jnp.ndarray):
        y = x_t
        new_states = []
        # Unroll the short SOS chain so XLA can generate a simpler loop body.
        for idx in range(sos.shape[0]):
            coeff = sos[idx]
            state = carry[idx]
            b0, b1, b2, _a0, a1, a2 = coeff
            z1 = state[:, 0]
            z2 = state[:, 1]
            y_next = b0 * y + z1
            new_z1 = b1 * y - a1 * y_next + z2
            new_z2 = b2 * y - a2 * y_next
            y = y_next
            new_states.append(jnp.stack([new_z1, new_z2], axis=1))
        return jnp.stack(new_states, axis=0), y

    final_states, ys = lax.scan(_sample_step, states, flat.T)
    return ys.T, jnp.moveaxis(final_states, 0, 1)


def _odd_ext_last_axis(flat: jnp.ndarray, edge: int) -> jnp.ndarray:
    if edge == 0:
        return flat
    left = 2 * flat[:, :1] - flat[:, 1 : edge + 1][:, ::-1]
    right = 2 * flat[:, -1:] - flat[:, -edge - 1 : -1][:, ::-1]
    return jnp.concatenate([left, flat, right], axis=1)


def pass_filter_kernel(
    data: Any,
    sos: Any,
    axis: int,
    zerophase: bool = True,
    zi: Any | None = None,
    padlen: int | None = None,
) -> Any:
    """Apply a Butterworth pass filter with native JAX SOS filtering."""
    arr = jnp.asarray(data)
    moved = jnp.moveaxis(arr, axis, -1)
    flat = moved.reshape(-1, moved.shape[-1])
    sos_arr = _normalize_sos(jnp.asarray(sos, dtype=arr.dtype))

    if zi is None:
        zi_arr = jnp.asarray(
            pass_filter_initial_state(np.asarray(sos)), dtype=arr.dtype
        )
    else:
        zi_arr = jnp.asarray(zi, dtype=arr.dtype)

    if not zerophase:
        out_flat, _ = _sosfilt_rows(flat, sos_arr)
        out = out_flat.reshape(moved.shape)
        return jnp.moveaxis(out, -1, axis)

    edge = (
        pass_filter_default_padlen(np.asarray(sos)) if padlen is None else int(padlen)
    )
    if edge > 0 and flat.shape[1] <= edge:
        msg = (
            f"The length of the input vector x must be greater than padlen, "
            f"which is {edge}."
        )
        raise ValueError(msg)
    ext = _odd_ext_last_axis(flat, edge)
    zi_rows = jnp.broadcast_to(zi_arr[None, :, :], (ext.shape[0],) + zi_arr.shape)
    x0 = ext[:, :1]
    y_flat, _ = _sosfilt_rows(ext, sos_arr, zi_rows=zi_rows * x0[:, :, None])
    y0 = y_flat[:, -1:]
    y_rev = jnp.flip(y_flat, axis=1)
    y2_flat, _ = _sosfilt_rows(y_rev, sos_arr, zi_rows=zi_rows * y0[:, :, None])
    out_flat = jnp.flip(y2_flat, axis=1)
    if edge > 0:
        out_flat = out_flat[:, edge:-edge]
    out = out_flat.reshape(moved.shape)
    return jnp.moveaxis(out, -1, axis)


def _extract_zero_padded_frames(
    flat: jnp.ndarray,
    frame_starts: jnp.ndarray,
    window_samples: int,
) -> jnp.ndarray:
    sample_offsets = jnp.arange(window_samples, dtype=frame_starts.dtype)
    indices = frame_starts[:, None] + sample_offsets[None, :]
    valid = (indices >= 0) & (indices < flat.shape[1])
    clipped = jnp.clip(indices, 0, max(flat.shape[1] - 1, 0))
    gathered = jnp.take(flat, clipped, axis=1)
    return jnp.where(valid[None, :, :], gathered, 0)


def _linear_detrend_frames(frames: jnp.ndarray) -> jnp.ndarray:
    window_samples = frames.shape[-1]
    if window_samples <= 1:
        return frames
    x = jnp.arange(window_samples, dtype=frames.dtype)
    x_mean = 0.5 * (window_samples - 1)
    denom = window_samples * (window_samples * window_samples - 1) / 12.0
    sum_y = jnp.sum(frames, axis=-1, keepdims=True)
    sum_xy = jnp.sum(frames * x, axis=-1, keepdims=True)
    slope = (sum_xy - x_mean * sum_y) / denom
    intercept = sum_y / window_samples - slope * x_mean
    return frames - (slope * x + intercept)


@functools.partial(jax.jit, static_argnames=("axis", "detrend"))
def banded_stft_kernel(
    data: Any,
    *,
    axis: int,
    window: Any,
    frame_starts: Any,
    selected_bins: Any,
    sample_step: float,
    detrend: bool = False,
) -> Any:
    """Compute banded STFT magnitude sums over selected frequency bins."""
    arr = jnp.asarray(data)
    nan_mask = ~jnp.isfinite(arr)
    arr = jnp.where(~nan_mask, arr, jnp.zeros_like(arr))
    moved = jnp.moveaxis(arr, axis, -1)
    flat = moved.reshape(-1, moved.shape[-1])
    # Track which output frames overlapped any non-finite input sample so they
    # can be zeroed, matching DASCore's nansum semantics.
    moved_nan = jnp.moveaxis(nan_mask, axis, -1)
    flat_nan = moved_nan.reshape(-1, moved_nan.shape[-1])
    window_arr = jnp.asarray(window, dtype=arr.dtype)
    frame_starts_arr = jnp.asarray(frame_starts)
    selected_bins_arr = jnp.asarray(selected_bins)
    nan_in_frames = _extract_zero_padded_frames(
        flat_nan.astype(jnp.float32), frame_starts_arr, window_arr.shape[0]
    )
    frame_has_nan = jnp.any(nan_in_frames > 0, axis=-1)
    frames = _extract_zero_padded_frames(flat, frame_starts_arr, window_arr.shape[0])
    if detrend:
        frames = _linear_detrend_frames(frames)
    tapered = frames * window_arr
    spectrum = jnp.fft.rfft(tapered, axis=-1)
    reduced = jnp.sum(jnp.abs(spectrum[..., selected_bins_arr]), axis=-1) * abs(sample_step)
    reduced = jnp.where(frame_has_nan, 0.0, reduced)
    out = reduced.reshape(moved.shape[:-1] + (frame_starts_arr.shape[0],))
    return jnp.moveaxis(out, -1, axis)
