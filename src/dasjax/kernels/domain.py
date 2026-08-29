"""Domain-specific transform kernels."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from .spectral import dft_kernel, hilbert_kernel, idft_kernel


def apply_mask_kernel(data: Any, mask: Any) -> Any:
    """Apply a static broadcastable mask to data."""
    return jnp.asarray(data) * jnp.asarray(mask, dtype=jnp.asarray(data).dtype)


def slope_filter_kernel(
    data: Any,
    *,
    axes: tuple[int, int],
    dxs: tuple[float, float],
    mask: Any,
    steps: tuple[float, float],
    sizes: tuple[int, int],
    pad_width: tuple[tuple[int, int], ...] = (),
) -> Any:
    """Apply a 2-D slope/FK filter and return to the input domain."""
    arr = jnp.asarray(data)
    orig_shape = arr.shape
    if pad_width:
        arr = jnp.pad(arr, pad_width)
    spectrum = dft_kernel(arr, axes=axes, dxs=dxs, real_axis=None)
    filtered = spectrum * jnp.asarray(mask, dtype=spectrum.dtype)
    out = idft_kernel(filtered, axes=axes, new_steps=steps, sizes=sizes, real=False)
    if pad_width:
        slices = tuple(slice(0, size) for size in orig_shape)
        out = out[slices]
    return jnp.real(out)


def phase_weighted_stack_kernel(
    data: Any,
    *,
    stack_axis: int,
    transform_axis: int,
    power: float,
    squeeze: bool,
) -> Any:
    """Apply phase-weighted stacking over one axis."""
    arr = jnp.asarray(data)
    analytic = hilbert_kernel(arr, axis=transform_axis)
    eps = jnp.finfo(analytic.real.dtype).eps
    amp = jnp.maximum(jnp.abs(analytic), eps)
    unit_phasors = analytic / amp
    mean_phasor = jnp.mean(unit_phasors, axis=stack_axis, keepdims=True)
    weights = jnp.abs(mean_phasor) ** power
    out = jnp.mean(arr, axis=stack_axis, keepdims=True) * weights
    return jnp.squeeze(out, axis=stack_axis) if squeeze else out


def velocity_to_strain_rate_edgeless_kernel(
    data: Any,
    *,
    axis: int,
    step_multiple: int,
    gauge_length: float,
) -> Any:
    """Estimate edgeless strain rate with central differences."""
    arr = jnp.asarray(data)
    moved = jnp.moveaxis(arr, axis, 0)
    out = (moved[step_multiple:] - moved[:-step_multiple]) / gauge_length
    return jnp.moveaxis(out, 0, axis)


def radians_to_strain_kernel(data: Any, factor: float) -> Any:
    """Scale radians data into strain units."""
    return jnp.asarray(data) * factor


def _interp_along_time(row: jnp.ndarray, sample: jnp.ndarray) -> jnp.ndarray:
    """Linearly interpolate one trace at fractional sample positions."""
    nt = row.shape[0]
    idx = jnp.floor(sample).astype(jnp.int32)
    valid = (idx >= 0) & (idx + 1 < nt)
    idx_clip = jnp.clip(idx, 0, max(nt - 2, 0))
    frac = sample - idx
    vals = (1.0 - frac) * row[idx_clip] + frac * row[idx_clip + 1]
    return jnp.where(valid, vals, 0)


def tau_p_kernel(
    data: Any,
    *,
    distances: Any,
    dt: float,
    p_values: Any,
) -> Any:
    """Compute a linear tau-p transform using JAX interpolation."""
    arr = jnp.asarray(data)
    distances = jnp.asarray(distances, dtype=arr.real.dtype)
    distances = distances - distances[0]
    p_vals = jnp.asarray(p_values, dtype=arr.real.dtype)
    nt = arr.shape[1]
    taus = jnp.arange(nt, dtype=arr.real.dtype)

    def _positive(p):
        samples = taus[None, :] + (p * distances[:, None] / dt)
        vals = jax.vmap(_interp_along_time)(arr, samples)
        return jnp.sum(vals, axis=0)

    def _negative(p):
        rev_arr = arr[::-1]
        rev_dist = distances[-1] - distances[::-1]
        samples = taus[None, :] + (p * rev_dist[:, None] / dt)
        vals = jax.vmap(_interp_along_time)(rev_arr, samples)
        return jnp.sum(vals, axis=0)

    pos = jax.vmap(_positive)(p_vals)
    neg = jax.vmap(_negative)(p_vals)[::-1]
    return jnp.concatenate([neg, pos], axis=0)


def dispersion_phase_shift_kernel(
    data: Any,
    *,
    distances: Any,
    velocities: Any,
    nf: int,
    first_live_f: int,
    last_live_f: int,
    fs: float,
) -> Any:
    """Compute a phase-shift dispersion image."""
    arr = jnp.asarray(data)
    dist = jnp.asarray(distances, dtype=arr.real.dtype)
    velocities = jnp.asarray(velocities, dtype=arr.real.dtype)
    freqs = jnp.arange(nf, dtype=arr.real.dtype) * fs / (nf - 1)
    omega = 2 * jnp.pi * freqs
    fft_d = jnp.fft.fft(arr, n=nf, axis=1)
    amp = jnp.abs(fft_d)
    fft_d = jnp.where(amp != 0, fft_d / amp, 0)
    omega_live = omega[first_live_f:last_live_f]
    fft_live = fft_d[:, first_live_f:last_live_f]
    preamb = 1j * dist[:, None] * omega_live[None, :]

    def _velocity_row(velocity):
        return jnp.abs(jnp.sum(jnp.exp(preamb / velocity) * fft_live, axis=0))

    return jax.vmap(_velocity_row)(velocities) / arr.shape[0]
