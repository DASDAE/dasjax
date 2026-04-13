"""Filter kernels and callback-backed helpers."""

from __future__ import annotations

from typing import Any

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from dascore.exceptions import FilterValueError
from scipy import ndimage
from scipy.ndimage import median_filter as scipy_median_filter
from scipy.signal import iirfilter, sosfilt_zi, zpk2sos


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
    """Apply a Gaussian filter with SciPy-compatible semantics."""
    # Gaussian filtering still relies on SciPy's exact implementation semantics.
    arr = jnp.asarray(data)
    # The callback result shape/dtype is identical to the source array.
    shape_dtype = jax.ShapeDtypeStruct(arr.shape, arr.dtype)
    return jax.pure_callback(
        lambda x: ndimage.gaussian_filter(
            # Force a private NumPy copy so SciPy can mutate internally if needed.
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


def _median_filter_axis(data: jnp.ndarray, axis: int, window: int) -> jnp.ndarray:
    # Reduce the target axis to the last position so the window helper can be reused.
    moved = jnp.moveaxis(data, axis, -1)
    flat = moved.reshape(-1, moved.shape[-1])
    windows = _sliding_windows_last_axis(flat, window)
    # Sorting the small window and taking the center reproduces a median.
    med = jnp.sort(windows, axis=-1)[..., window // 2]
    restored = med.reshape(moved.shape)
    return jnp.moveaxis(restored, -1, axis)


def _median_filter_exact_callback(
    data: np.ndarray, size: tuple[int, ...]
) -> np.ndarray:
    return scipy_median_filter(data, size=size, mode="reflect")


def _median_filter_exact(data: jnp.ndarray, size: tuple[int, ...]) -> jnp.ndarray:
    # The exact multidimensional SciPy median is still callback-backed.
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
    # The callback preserves both shape and dtype of the incoming array.
    shape_dtype = jax.ShapeDtypeStruct(source.shape, source.dtype)

    def _callback(x: np.ndarray) -> np.ndarray:
        original = np.asarray(x)
        is_int = original.dtype.kind in {"i", "u"}
        # Integer inputs are promoted for robust MAD math, then cast back on exit.
        work = original.copy() if not is_int else original.astype(np.float32)
        if approximate:
            # Approximate mode applies separable 1D median filters per axis.
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
            # Reuse the same separable strategy for the MAD estimate.
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
            # Exact mode delegates the full multidimensional median to SciPy.
            med = scipy_median_filter(work, size=size, mode="reflect")
            abs_diff = np.abs(work - med)
            mad = scipy_median_filter(abs_diff, size=size, mode="reflect")
        # Replace zeros to avoid divide-by-zero when a local window is constant.
        mad_safe = np.where(mad == 0.0, np.finfo(work.dtype).eps, mad)
        out = np.where(abs_diff / mad_safe > threshold, med, work)
        if is_int:
            # Integer inputs round back to the nearest representable sample value.
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
    # Mirror the callback path by promoting integers before the MAD computation.
    compute_dtype = jnp.float32 if is_int else source.dtype
    work = source.astype(compute_dtype)
    if approximate:
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
    else:
        # The exact path falls back to the SciPy callback helper above.
        med = _median_filter_exact(work, size)
        abs_diff = jnp.abs(work - med)
        mad = _median_filter_exact(abs_diff, size)
    # Protect constant windows from generating infinities.
    mad_safe = jnp.where(mad == 0.0, jnp.finfo(work.dtype).eps, mad)
    out = jnp.where(abs_diff / mad_safe > threshold, med, work)
    if is_int:
        # Cast back to the original integer dtype after robust filtering.
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
