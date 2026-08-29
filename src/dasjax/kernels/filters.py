"""Filter kernels."""

from __future__ import annotations

from typing import Any

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from dascore.exceptions import FilterValueError
from scipy.signal import (
    iirfilter,
    iirnotch,
    lfilter_zi,
    savgol_coeffs,
    sosfilt_zi,
    zpk2sos,
)


def is_finite_array(data: Any) -> bool:
    """Return True when every value in the array is finite."""
    # This intentionally uses NumPy because it is called from Python validation paths.
    return bool(np.isfinite(np.asarray(data)).all())


def gaussian_filter_kernel(
    data: Any,
    sigma: tuple[float, ...],
    axes: tuple[int, ...],
    mode: str = "reflect",
    cval: float = 0.0,
    truncate: float = 4.0,
) -> Any:
    """Apply a separable Gaussian filter with JAX-native operations."""
    arr = jnp.asarray(data)
    if not sigma or not axes:
        return arr
    out = arr
    for axis, sigma_value in zip(axes, sigma, strict=True):
        out = _gaussian_filter_axis(
            out,
            sigma=float(sigma_value),
            axis=int(axis),
            mode=mode,
            cval=cval,
            truncate=truncate,
        )
    return out


def _gaussian_filter_axis(
    data: jnp.ndarray,
    *,
    sigma: float,
    axis: int,
    mode: str,
    cval: float,
    truncate: float,
) -> jnp.ndarray:
    """Apply one static 1D Gaussian pass along an axis."""
    if sigma <= 0:
        return data
    radius = int(truncate * sigma + 0.5)
    if radius == 0:
        return data
    x = jnp.arange(-radius, radius + 1, dtype=data.real.dtype)
    weights = jnp.exp(-0.5 / (sigma * sigma) * x * x)
    weights = weights / jnp.sum(weights)
    moved = jnp.moveaxis(data, axis, -1)
    flat = moved.reshape(-1, moved.shape[-1])
    padded = _pad_gaussian_axis(flat, radius=radius, mode=mode, cval=cval)

    def _convolve(row: jnp.ndarray) -> jnp.ndarray:
        return jnp.convolve(row, weights, mode="valid")

    filtered = jax.vmap(_convolve)(padded)
    return jnp.moveaxis(filtered.reshape(moved.shape), -1, axis)


def _pad_gaussian_axis(
    data: jnp.ndarray,
    *,
    radius: int,
    mode: str,
    cval: float,
) -> jnp.ndarray:
    """Pad flattened rows with SciPy ndimage-compatible mode names."""
    pad_width = ((0, 0), (radius, radius))
    if mode == "reflect":
        return jnp.pad(data, pad_width, mode="symmetric")
    if mode == "nearest":
        return jnp.pad(data, pad_width, mode="edge")
    if mode == "constant":
        return jnp.pad(data, pad_width, mode="constant", constant_values=cval)
    if mode == "wrap":
        return jnp.pad(data, pad_width, mode="wrap")
    if mode == "mirror":
        return jnp.pad(data, pad_width, mode="reflect")
    msg = f"Unsupported gaussian_filter mode {mode!r}."
    raise ValueError(msg)


def _normalize_filter_mode(mode: str) -> str:
    """Normalize SciPy ndimage-style mode aliases."""
    aliases = {
        "grid-constant": "constant",
        "grid-mirror": "reflect",
        "grid-wrap": "wrap",
    }
    return aliases.get(mode, mode)


def _pad_axis(
    data: jnp.ndarray,
    *,
    axis: int,
    before: int,
    after: int,
    mode: str,
    cval: float = 0.0,
) -> jnp.ndarray:
    """Pad one axis with SciPy ndimage-compatible mode names."""
    pad_width = [(0, 0)] * data.ndim
    pad_width[axis] = (before, after)
    mode = _normalize_filter_mode(mode)
    if mode == "reflect":
        return jnp.pad(data, tuple(pad_width), mode="symmetric")
    if mode == "nearest":
        return jnp.pad(data, tuple(pad_width), mode="edge")
    if mode == "constant":
        return jnp.pad(data, tuple(pad_width), mode="constant", constant_values=cval)
    if mode == "wrap":
        return jnp.pad(data, tuple(pad_width), mode="wrap")
    if mode == "mirror":
        return jnp.pad(data, tuple(pad_width), mode="reflect")
    msg = f"Unsupported filter mode {mode!r}."
    raise ValueError(msg)


def _sliding_windows_last_axis(data: jnp.ndarray, window: int) -> jnp.ndarray:
    # Mirror the historical median-filter edge handling with explicit edge padding.
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    pad_width = [(0, 0)] * data.ndim
    pad_width[-1] = (pad_left, pad_right)
    padded = jnp.pad(data, tuple(pad_width), mode="edge")
    length = data.shape[-1]

    def _slice_at(idx: jnp.ndarray) -> jnp.ndarray:
        # Slice one fixed-width window out of the padded last axis.
        return lax.dynamic_slice_in_dim(padded, idx, window, axis=data.ndim - 1)

    # Vectorize over output positions to produce one window per input sample.
    windows = jax.vmap(_slice_at)(jnp.arange(length))
    # Swap axes so callers get `(rows, length, window)` windows.
    return jnp.swapaxes(windows, 0, 1)


def _sliding_windows_nd(
    data: jnp.ndarray,
    size: tuple[int, ...],
    out_shape: tuple[int, ...],
) -> jnp.ndarray:
    """Return flattened N-D windows for every unpadded output element."""
    out_size = int(np.prod(out_shape))
    starts = jnp.stack(jnp.unravel_index(jnp.arange(out_size), out_shape), axis=1)

    def _slice_at(start: jnp.ndarray) -> jnp.ndarray:
        return lax.dynamic_slice(data, tuple(start), size).reshape(-1)

    return jax.vmap(_slice_at)(starts).reshape(*out_shape, int(np.prod(size)))


def _window_sum_kernel(data: jnp.ndarray, size: tuple[int, ...]) -> jnp.ndarray:
    """Sum zero-padded local windows in SciPy signal-correlate style."""
    before = tuple(window // 2 for window in size)
    after = tuple(window - 1 - pad for window, pad in zip(size, before, strict=True))
    return lax.reduce_window(
        data,
        jnp.asarray(0, dtype=data.dtype),
        lax.add,
        window_dimensions=tuple(int(x) for x in size),
        window_strides=(1,) * data.ndim,
        padding=tuple(zip(before, after, strict=True)),
    )


def median_filter_kernel(
    data: Any,
    size: tuple[int, ...],
    mode: str = "reflect",
    cval: float = 0.0,
) -> Any:
    """Apply an exact N-D median filter with JAX-native windows."""
    arr = jnp.asarray(data)
    if all(window == 1 for window in size):
        return arr
    before = tuple(window // 2 for window in size)
    after = tuple(window - 1 - pad for window, pad in zip(size, before, strict=True))
    padded = arr
    for axis, (left, right) in enumerate(zip(before, after, strict=True)):
        if left or right:
            padded = _pad_axis(
                padded,
                axis=axis,
                before=left,
                after=right,
                mode=mode,
                cval=cval,
            )
    windows = _sliding_windows_nd(padded, tuple(int(x) for x in size), tuple(arr.shape))
    return jnp.sort(windows, axis=-1)[..., windows.shape[-1] // 2]


def wiener_filter_kernel(
    data: Any,
    size: tuple[int, ...],
    noise: float | None = None,
) -> Any:
    """Apply an N-D Wiener filter matching SciPy's local-statistic formula."""
    arr = jnp.asarray(data)
    window_size = float(np.prod(size))
    local_mean = _window_sum_kernel(arr, size) / window_size
    local_var = (
        _window_sum_kernel(arr * arr, size) / window_size - local_mean * local_mean
    )
    noise_value = (
        jnp.mean(local_var) if noise is None else jnp.asarray(noise, dtype=arr.dtype)
    )
    res = (arr - local_mean) * (1 - noise_value / local_var) + local_mean
    return jnp.where(local_var < noise_value, local_mean, res)


def _median_filter_axis(data: jnp.ndarray, axis: int, window: int) -> jnp.ndarray:
    # Reduce the target axis to the last position so the window helper can be reused.
    moved = jnp.moveaxis(data, axis, -1)
    flat = moved.reshape(-1, moved.shape[-1])
    windows = _sliding_windows_last_axis(flat, window)
    # Sorting the small window and taking the center reproduces a median.
    med = jnp.sort(windows, axis=-1)[..., window // 2]
    restored = med.reshape(moved.shape)
    return jnp.moveaxis(restored, -1, axis)


def hampel_filter_kernel(
    data: Any,
    size: tuple[int, ...],
    threshold: float,
    approximate: bool = True,
) -> Any:
    """Apply a Hampel filter with a native JAX approximate path."""
    if not approximate:
        msg = "Exact Hampel median filtering is not implemented in pure JAX."
        raise NotImplementedError(msg)
    source = jnp.asarray(data)
    is_int = jnp.issubdtype(source.dtype, jnp.integer)
    # Promote integers before robust MAD math, then cast back on exit.
    compute_dtype = jnp.float32 if is_int else source.dtype
    work = source.astype(compute_dtype)
    # Separable medians keep the kernel JAX-native and reasonably cheap.
    med = work
    for axis, window in enumerate(size):
        if window > 1:
            med = _median_filter_axis(med, axis, window)
    abs_diff = jnp.abs(work - med)
    mad = abs_diff
    # Apply the same separable median strategy to absolute deviations.
    for axis, window in enumerate(size):
        if window > 1:
            mad = _median_filter_axis(mad, axis, window)
    # Protect constant windows from generating infinities.
    mad_safe = jnp.where(mad == 0.0, jnp.finfo(work.dtype).eps, mad)
    out = jnp.where(abs_diff / mad_safe > threshold, med, work)
    if is_int:
        # Cast back to the original integer dtype after robust filtering.
        out = jnp.rint(out).astype(source.dtype)
    return out


def sobel_filter_kernel(
    data: Any,
    axis: int,
    mode: str = "reflect",
    cval: float = 0.0,
) -> Any:
    """Apply a SciPy-compatible Sobel filter along one axis."""
    arr = jnp.asarray(data)
    out = arr
    for dim_axis in range(arr.ndim):
        weights = (
            jnp.asarray([-1.0, 0.0, 1.0], dtype=arr.dtype)
            if dim_axis == axis
            else jnp.asarray([1.0, 2.0, 1.0], dtype=arr.dtype)
        )
        out = _correlate1d_kernel(out, weights, axis=dim_axis, mode=mode, cval=cval)
    return out


def _correlate1d_kernel(
    data: jnp.ndarray,
    weights: jnp.ndarray,
    *,
    axis: int,
    mode: str,
    cval: float,
) -> jnp.ndarray:
    """Apply 1-D correlation over one axis."""
    radius = weights.shape[0] // 2
    padded = _pad_axis(
        data,
        axis=axis,
        before=radius,
        after=radius,
        mode=mode,
        cval=cval,
    )
    moved = jnp.moveaxis(padded, axis, -1)
    flat = moved.reshape(-1, moved.shape[-1])

    def _correlate(row: jnp.ndarray) -> jnp.ndarray:
        return jnp.convolve(row, weights[::-1], mode="valid")

    filtered = jax.vmap(_correlate)(flat)
    out_shape = (*moved.shape[:-1], data.shape[axis])
    return jnp.moveaxis(filtered.reshape(out_shape), -1, axis)


def _savgol_filter_axis(
    data: jnp.ndarray,
    *,
    axis: int,
    coeffs: Any,
    left_coeffs: Any,
    right_coeffs: Any,
    mode: str,
    cval: float,
) -> jnp.ndarray:
    """Apply one Savitzky-Golay filter pass along an axis."""
    coeff_arr = jnp.asarray(coeffs, dtype=data.dtype)
    window = int(coeff_arr.shape[0])
    half = window // 2
    moved = jnp.moveaxis(data, axis, -1)
    flat = moved.reshape(-1, moved.shape[-1])
    if flat.shape[1] < window:
        msg = "savgol_filter window_length must be less than or equal to axis length."
        raise ValueError(msg)

    def _valid_convolve(row: jnp.ndarray) -> jnp.ndarray:
        return jnp.convolve(row, coeff_arr, mode="valid")

    middle = jax.vmap(_valid_convolve)(flat)
    if mode == "interp":
        left = jnp.einsum(
            "hw,rw->rh", jnp.asarray(left_coeffs, dtype=data.dtype), flat[:, :window]
        )
        right = jnp.einsum(
            "hw,rw->rh", jnp.asarray(right_coeffs, dtype=data.dtype), flat[:, -window:]
        )
        out_flat = jnp.concatenate([left, middle, right], axis=1)
    else:
        padded = _pad_axis(
            flat,
            axis=1,
            before=half,
            after=half,
            mode=mode,
            cval=cval,
        )
        out_flat = jax.vmap(_valid_convolve)(padded)
    return jnp.moveaxis(out_flat.reshape(moved.shape), -1, axis)


def savgol_filter_kernel(
    data: Any,
    size: tuple[int, ...],
    axes: tuple[int, ...],
    coeffs: tuple[Any, ...],
    left_coeffs: tuple[Any, ...],
    right_coeffs: tuple[Any, ...],
    mode: str = "interp",
    cval: float = 0.0,
) -> Any:
    """Apply Savitzky-Golay filters sequentially along requested axes."""
    out = jnp.asarray(data)
    for axis, coeff, left, right in zip(
        axes, coeffs, left_coeffs, right_coeffs, strict=True
    ):
        _ = size
        out = _savgol_filter_axis(
            out,
            axis=axis,
            coeffs=coeff,
            left_coeffs=left,
            right_coeffs=right,
            mode=mode,
            cval=cval,
        )
    return out


def savgol_coefficients(
    window_length: int, polyorder: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return center and interp-edge Savitzky-Golay coefficients."""
    coeff = np.asarray(
        savgol_coeffs(window_length, polyorder, use="conv"), dtype=np.float64
    )
    half = window_length // 2
    left = np.asarray(
        [
            savgol_coeffs(window_length, polyorder, pos=idx, use="dot")
            for idx in range(half)
        ],
        dtype=np.float64,
    )
    right = np.asarray(
        [
            savgol_coeffs(window_length, polyorder, pos=idx, use="dot")
            for idx in range(window_length - half, window_length)
        ],
        dtype=np.float64,
    )
    return coeff, left, right


def design_notch_filter(
    sr: float,
    w0: float,
    q: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Design a second-order notch filter and return filtfilt parameters."""
    nyquist = 0.5 * sr
    if w0 > nyquist:
        msg = f"possible filter values are in [0, {nyquist}] you passed {w0}"
        raise FilterValueError(msg)
    b, a = iirnotch(w0, Q=q, fs=sr)
    zi = lfilter_zi(b, a)
    padlen = 3 * max(len(a), len(b))
    return (
        np.asarray(b, dtype=np.float64),
        np.asarray(a, dtype=np.float64),
        np.asarray(zi, dtype=np.float64),
        padlen,
    )


def _iirfilt_rows(
    flat: jnp.ndarray,
    b: jnp.ndarray,
    a: jnp.ndarray,
    zi_rows: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply one direct-form II IIR filter to flattened rows."""
    b = b / a[0]
    a = a / a[0]
    if zi_rows is None:
        states = jnp.zeros((flat.shape[0], b.shape[0] - 1), dtype=flat.dtype)
    else:
        states = zi_rows

    def _sample_step(carry: jnp.ndarray, x_t: jnp.ndarray):
        y = b[0] * x_t + carry[:, 0]
        new_states = []
        for idx in range(1, b.shape[0] - 1):
            new_states.append(b[idx] * x_t - a[idx] * y + carry[:, idx])
        new_states.append(b[-1] * x_t - a[-1] * y)
        return jnp.stack(new_states, axis=1), y

    final_states, ys = lax.scan(_sample_step, states, flat.T)
    return ys.T, final_states


def notch_filter_kernel(
    data: Any,
    b: Any,
    a: Any,
    zi: Any,
    padlen: int,
    axis: int,
) -> Any:
    """Apply a zero-phase second-order notch filter."""
    arr = jnp.asarray(data)
    moved = jnp.moveaxis(arr, axis, -1)
    flat = moved.reshape(-1, moved.shape[-1])
    edge = int(padlen)
    if edge > 0 and flat.shape[1] <= edge:
        msg = (
            f"The length of the input vector x must be greater than padlen, "
            f"which is {edge}."
        )
        raise ValueError(msg)
    ext = _odd_ext_last_axis(flat, edge)
    b_arr = jnp.asarray(b, dtype=arr.dtype)
    a_arr = jnp.asarray(a, dtype=arr.dtype)
    zi_arr = jnp.asarray(zi, dtype=arr.dtype)
    zi_rows = jnp.broadcast_to(zi_arr[None, :], (ext.shape[0], zi_arr.shape[0]))
    x0 = ext[:, :1]
    y_flat, _ = _iirfilt_rows(ext, b_arr, a_arr, zi_rows=zi_rows * x0)
    y0 = y_flat[:, -1:]
    y_rev = jnp.flip(y_flat, axis=1)
    y2_flat, _ = _iirfilt_rows(y_rev, b_arr, a_arr, zi_rows=zi_rows * y0)
    out_flat = jnp.flip(y2_flat, axis=1)
    if edge > 0:
        out_flat = out_flat[:, edge:-edge]
    out = out_flat.reshape(moved.shape)
    return jnp.moveaxis(out, -1, axis)


def design_pass_filter_sos(
    sr: float,
    filt_min: float | None,
    filt_max: float | None,
    corners: int,
) -> np.ndarray:
    """Design Butterworth SOS sections matching DASCore's pass_filter."""
    nyquist = 0.5 * sr
    # DASCore uses nullable bounds; NaN is treated the same as missing here.
    low = None if pd.isnull(filt_min) else filt_min / nyquist
    high = None if pd.isnull(filt_max) else filt_max / nyquist
    # Validate normalized cutoffs before calling SciPy's filter designer.
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
    # Pick the Butterworth topology from the available cutoffs.
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
    # This is SciPy's filtfilt heuristic rewritten for the SOS representation.
    ntaps = 2 * n_sections + 1
    ntaps -= min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum())
    return 3 * ntaps


def _normalize_sos(sos: jnp.ndarray) -> jnp.ndarray:
    # Normalize each section so the recursive denominator starts with a0 == 1.
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
        # State is `(sections, rows, two-delay-registers)` for each trace.
        states = jnp.zeros((sos.shape[0], n_rows, 2), dtype=flat.dtype)
    else:
        states = jnp.moveaxis(zi_rows, 0, 1)

    def _sample_step(carry: jnp.ndarray, x_t: jnp.ndarray):
        # Scan one sample at a time while explicitly unrolling the short SOS chain.
        y = x_t
        new_states = []
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

    # Scan over time samples and return both filtered output and final state.
    final_states, ys = lax.scan(_sample_step, states, flat.T)
    return ys.T, jnp.moveaxis(final_states, 0, 1)


def _odd_ext_last_axis(flat: jnp.ndarray, edge: int) -> jnp.ndarray:
    # Odd reflection matches SciPy's filtfilt padding behavior at the boundaries.
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
    # Move the filter axis last so each row is a trace we can scan over.
    arr = jnp.asarray(data)
    moved = jnp.moveaxis(arr, axis, -1)
    flat = moved.reshape(-1, moved.shape[-1])
    # SOS coefficients are normalized once up front for stable recursion.
    sos_arr = _normalize_sos(jnp.asarray(sos, dtype=arr.dtype))

    if zi is None:
        # Traced/JIT use must provide `zi`; only pure NumPy callers can infer it here.
        if not isinstance(sos, np.ndarray):
            raise ValueError(
                "pass_filter_kernel requires zi when used with traced/JAX SOS arrays."
            )
        zi_arr = jnp.asarray(
            pass_filter_initial_state(np.asarray(sos)), dtype=arr.dtype
        )
    else:
        zi_arr = jnp.asarray(zi, dtype=arr.dtype)

    if not zerophase:
        # Forward-only filtering is a single SOS scan over each trace.
        out_flat, _ = _sosfilt_rows(flat, sos_arr)
        out = out_flat.reshape(moved.shape)
        return jnp.moveaxis(out, -1, axis)

    # Zero-phase filtering uses forward/backward filtering with odd edge extension.
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
    # Broadcast the section state to every flattened trace.
    zi_rows = jnp.broadcast_to(zi_arr[None, :, :], (ext.shape[0],) + zi_arr.shape)
    x0 = ext[:, :1]
    y_flat, _ = _sosfilt_rows(ext, sos_arr, zi_rows=zi_rows * x0[:, :, None])
    y0 = y_flat[:, -1:]
    # Run the backward pass on reversed data, then flip back into forward order.
    y_rev = jnp.flip(y_flat, axis=1)
    y2_flat, _ = _sosfilt_rows(y_rev, sos_arr, zi_rows=zi_rows * y0[:, :, None])
    out_flat = jnp.flip(y2_flat, axis=1)
    if edge > 0:
        # Remove the synthetic edge samples after the bidirectional pass.
        out_flat = out_flat[:, edge:-edge]
    out = out_flat.reshape(moved.shape)
    return jnp.moveaxis(out, -1, axis)
