"""Spectral and FFT-backed kernels."""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .basic import apply_1d_weight_kernel


@functools.lru_cache(maxsize=32)
def _analytic_multiplier(length: int, dtype_token: str) -> np.ndarray:
    """Return the FFT-domain multiplier for an analytic signal."""
    dtype = np.dtype(dtype_token)
    # The Hilbert analytic-signal mask depends only on sequence length parity.
    if length % 2 == 0:
        head = np.array([1], dtype=dtype)
        middle = np.full((length // 2 - 1,), 2, dtype=dtype)
        nyquist = np.array([1], dtype=dtype)
        tail = np.zeros((length // 2 - 1,), dtype=dtype)
        return np.concatenate([head, middle, nyquist, tail])
    head = np.array([1], dtype=dtype)
    middle = np.full(((length - 1) // 2,), 2, dtype=dtype)
    tail = np.zeros(((length - 1) // 2,), dtype=dtype)
    return np.concatenate([head, middle, tail])


def hilbert_kernel(data: Any, axis: int) -> Any:
    """Compute the analytic signal along one axis."""
    # Work on the last axis so the FFT-domain mask is 1D and easy to apply.
    arr = jnp.asarray(data)
    moved = jnp.moveaxis(arr, axis, -1)
    fft = jnp.fft.fft(moved, axis=-1)
    # The multiplier is cached by `(length, dtype)` to avoid rebuilding constants.
    mult = jnp.asarray(_analytic_multiplier(moved.shape[-1], fft.dtype.str))
    out = jnp.fft.ifft(fft * mult, axis=-1)
    return jnp.moveaxis(out, -1, axis)


def envelope_kernel(data: Any, axis: int) -> Any:
    """Compute the amplitude envelope along one axis."""
    # Envelope is the magnitude of the analytic signal.
    return jnp.abs(hilbert_kernel(data, axis=axis))


def dft_kernel(
    data: Any,
    axes: tuple[int, ...],
    dxs: tuple[float, ...],
    real_axis: int | None = None,
) -> Any:
    """Apply a scaled DFT over one or more axes."""
    arr = jnp.asarray(data)
    # Multiply by sample spacing so forward and inverse transforms stay consistent.
    scale_factor = np.prod(dxs)
    if real_axis is None:
        fft_data = jnp.fft.fftn(arr, axes=axes) * scale_factor
        # Shift zero frequency to the center on fully complex transforms.
        return jnp.fft.fftshift(fft_data, axes=axes)
    # Real transforms keep the final axis in rfft layout while shifting the others.
    fft_axes = tuple(ax for ax in axes if ax != real_axis) + (real_axis,)
    fft_data = jnp.fft.rfftn(arr, axes=fft_axes) * scale_factor
    shift_axes = tuple(ax for ax in fft_axes if ax != real_axis)
    return jnp.fft.fftshift(fft_data, axes=shift_axes) if shift_axes else fft_data


def idft_kernel(
    data: Any,
    axes: tuple[int, ...],
    new_steps: tuple[float, ...],
    sizes: tuple[int, ...] | None = None,
    real: bool = False,
) -> Any:
    """Apply a scaled inverse DFT over one or more axes."""
    arr = jnp.asarray(data)
    # Undo the forward scaling using the frequency-domain step sizes.
    scale_factor = np.prod(new_steps)
    if real:
        real_axis = axes[-1]
        # Only the non-rfft axes were shifted in the forward real transform.
        shift_axes = tuple(ax for ax in axes if ax != real_axis)
        shifted = jnp.fft.ifftshift(arr / scale_factor, axes=shift_axes)
        return jnp.fft.irfftn(shifted, s=sizes, axes=axes)
    shifted = jnp.fft.ifftshift(arr / scale_factor, axes=axes)
    return jnp.fft.ifftn(shifted, s=sizes, axes=axes)


def _uniform_wrap_kernel(data: jnp.ndarray, window_len: int) -> jnp.ndarray:
    """Apply a wrapped moving average along the last axis."""
    if window_len <= 1:
        return data
    # Whitening smooths the amplitude spectrum with circular edge handling.
    pad_left = window_len // 2
    pad_right = window_len - 1 - pad_left
    padded = jnp.concatenate(
        [data[..., -pad_left:], data, data[..., :pad_right]], axis=-1
    )
    kernel = jnp.ones((window_len,), dtype=data.dtype) / window_len

    def _convolve(row: jnp.ndarray) -> jnp.ndarray:
        # Each flattened row is convolved independently with the same boxcar.
        return jnp.convolve(row, kernel, mode="valid")

    flat = padded.reshape(-1, padded.shape[-1])
    filtered = jax.vmap(_convolve)(flat)
    return filtered.reshape(data.shape)


def whiten_kernel(
    data: Any,
    axis: int,
    window_len: int | None = None,
    water_level: float | None = None,
    freq_weight: Any | None = None,
) -> Any:
    """Whiten data along one axis and return to the original domain."""
    # Whitening is implemented by normalizing FFT amplitudes while keeping phase.
    arr = jnp.asarray(data)
    moved = jnp.moveaxis(arr, axis, -1)
    is_real = not jnp.issubdtype(moved.dtype, jnp.complexfloating)
    if is_real:
        # Real inputs use the half-spectrum rFFT path.
        fft_data = jnp.fft.rfft(moved, axis=-1)
    else:
        fft_data = jnp.fft.fft(moved, axis=-1)
    amp = jnp.abs(fft_data)
    if window_len is None:
        # The unsmoothed case is straight phase-only whitening.
        norm_amp = jnp.where(amp != 0, amp, 1)
    else:
        # Smooth the spectrum envelope before division to avoid sharp amplification.
        env = _uniform_wrap_kernel(amp, window_len)
        if water_level is not None:
            # Water-level clipping prevents deep spectral notches from exploding.
            min_level = water_level * jnp.max(env, axis=-1, keepdims=True)
            env = jnp.maximum(env, min_level)
        norm_amp = jnp.where(env != 0, env, 1)
    phase_only = fft_data / norm_amp
    if freq_weight is not None:
        # Optional frequency weighting is applied in the spectrum before inversion.
        phase_only = apply_1d_weight_kernel(
            phase_only, axis=phase_only.ndim - 1, weight=freq_weight
        )
    if is_real:
        out = jnp.fft.irfft(phase_only, n=moved.shape[-1], axis=-1)
    else:
        out = jnp.fft.ifft(phase_only, axis=-1)
    return jnp.moveaxis(out, -1, axis)


def _extract_zero_padded_frames(
    flat: jnp.ndarray,
    frame_starts: jnp.ndarray,
    window_samples: int,
) -> jnp.ndarray:
    # Frame extraction is written in index form so negative and overflow samples
    # can be represented and zero-filled without Python loops.
    sample_offsets = jnp.arange(window_samples, dtype=frame_starts.dtype)
    indices = frame_starts[:, None] + sample_offsets[None, :]
    valid = (indices >= 0) & (indices < flat.shape[1])
    clipped = jnp.clip(indices, 0, max(flat.shape[1] - 1, 0))
    gathered = jnp.take(flat, clipped, axis=1)
    return jnp.where(valid[None, :, :], gathered, 0)


def _linear_detrend_frames(frames: jnp.ndarray) -> jnp.ndarray:
    # This is a batched least-squares line fit over the frame axis.
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
    # Non-finite inputs are zeroed before the FFT, but tracked separately below.
    nan_mask = ~jnp.isfinite(arr)
    arr = jnp.where(~nan_mask, arr, jnp.zeros_like(arr))
    # Flatten non-transform axes so each row is one trace through the STFT.
    moved = jnp.moveaxis(arr, axis, -1)
    flat = moved.reshape(-1, moved.shape[-1])
    moved_nan = jnp.moveaxis(nan_mask, axis, -1)
    flat_nan = moved_nan.reshape(-1, moved_nan.shape[-1])
    # Static STFT parameters arrive precomputed from the operation layer.
    window_arr = jnp.asarray(window, dtype=arr.dtype)
    frame_starts_arr = jnp.asarray(frame_starts)
    selected_bins_arr = jnp.asarray(selected_bins)
    nan_in_frames = _extract_zero_padded_frames(
        flat_nan.astype(jnp.float32), frame_starts_arr, window_arr.shape[0]
    )
    # Any frame that touched a NaN is zeroed to match DASCore's reduction semantics.
    frame_has_nan = jnp.any(nan_in_frames > 0, axis=-1)
    frames = _extract_zero_padded_frames(flat, frame_starts_arr, window_arr.shape[0])
    if detrend:
        # Optional linear detrending happens in the time domain before tapering.
        frames = _linear_detrend_frames(frames)
    tapered = frames * window_arr
    spectrum = jnp.fft.rfft(tapered, axis=-1)
    # Reduce the selected band to a single scalar per frame.
    reduced = jnp.sum(jnp.abs(spectrum[..., selected_bins_arr]), axis=-1) * abs(
        sample_step
    )
    reduced = jnp.where(frame_has_nan, 0.0, reduced)
    # Rebuild the original outer shape with the frame axis replacing samples.
    out = reduced.reshape(moved.shape[:-1] + (frame_starts_arr.shape[0],))
    return jnp.moveaxis(out, -1, axis)
