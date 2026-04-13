"""Signal-processing kernels that preserve patch shape."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np


def validate_detrend_type(type: str) -> str:
    """Normalize supported detrend type aliases."""
    # Keep alias handling in one place so eager and compiled validation agree.
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
    # All math is written against JAX arrays so the same code can be jitted.
    data = jnp.asarray(data)
    detrend_type = validate_detrend_type(type)
    # Constant detrending is just mean subtraction along the selected axis.
    if detrend_type == "constant":
        return data - jnp.mean(data, axis=axis, keepdims=True)

    # Move the target axis to the front so the regression math is 1D in layout.
    moved = jnp.moveaxis(data, axis, 0)
    npts = moved.shape[0]
    # Collapse trailing dimensions so the least-squares fit runs per trace column.
    reshaped = moved.reshape(npts, -1)
    dtype = reshaped.dtype
    # Use a normalized ramp to match the historical DASCore linear detrend shape.
    ramp = jnp.arange(1, npts + 1, dtype=dtype) / npts
    ramp_centered = ramp - jnp.mean(ramp)
    mean = jnp.mean(reshaped, axis=0, keepdims=True)
    # The centered ramp keeps the intercept and slope computations numerically tidy.
    numerator = jnp.sum(ramp_centered[:, None] * (reshaped - mean), axis=0)
    denominator = jnp.sum(ramp_centered * ramp_centered)
    slope = jnp.where(denominator != 0, numerator / denominator, 0)
    trend = mean + ramp_centered[:, None] * slope[None, :]
    detrended = reshaped - trend
    # Restore the original axis order before returning to the caller.
    restored = detrended.reshape(moved.shape)
    return jnp.moveaxis(restored, 0, axis)


def normalize_kernel(data: Any, axis: int, norm: str = "l2") -> Any:
    """Normalize an array along one axis using DASCore semantics."""
    data = jnp.asarray(data)
    # L1/L2 both reduce to a norm plus a broadcasted divide.
    if norm in {"l1", "l2"}:
        order = int(norm[-1])
        norm_values = jnp.linalg.norm(data, axis=axis, ord=order)
        expanded_norm = jnp.expand_dims(norm_values, axis=axis)
        # Zero-norm slices map to zeros instead of NaNs.
        return jnp.where(expanded_norm != 0, data / expanded_norm, 0)
    if norm == "max":
        # Max normalization matches DASCore's sign-preserving divide-by-peak behavior.
        norm_values = jnp.max(data, axis=axis)
        expanded_norm = jnp.expand_dims(norm_values, axis=axis)
        return jnp.where(expanded_norm != 0, data / expanded_norm, 0)
    if norm == "bit":
        # Bit normalization keeps only the sign/phase of each sample.
        abs_data = jnp.abs(data)
        return jnp.where(abs_data != 0, data / abs_data, 0)
    msg = f"Unsupported normalization mode {norm!r}."
    raise ValueError(msg)


def differentiate_kernel(
    data: Any,
    axis: int,
    dx_or_spacing: float | np.ndarray,
    order: int = 2,
    step: int = 1,
) -> Any:
    """Differentiate an array along one axis."""
    arr = jnp.asarray(data)
    # Accept either a scalar spacing or a full coordinate vector.
    spacing = jnp.asarray(dx_or_spacing) if np.ndim(dx_or_spacing) else dx_or_spacing
    # The current compiled contract only covers DASCore's default path.
    if step > 1:
        msg = "Compiled differentiate currently requires step=1."
        raise NotImplementedError(msg)
    if order != 2:
        msg = "Compiled differentiate currently requires order=2."
        raise NotImplementedError(msg)
    # `jnp.gradient` handles both scalar and per-sample spacing inputs.
    return jnp.gradient(arr, spacing, axis=axis, edge_order=2)


def integrate_kernel(
    data: Any,
    axis: int,
    dx_or_spacing: float | np.ndarray,
    definite: bool = False,
) -> Any:
    """Integrate an array along one axis using the trapezoidal rule."""
    arr = jnp.asarray(data)
    # Moving the integration axis last makes pairwise trapezoid math simpler.
    moved = jnp.moveaxis(arr, axis, -1)
    if np.ndim(dx_or_spacing):
        # Uneven spacing is represented as a coordinate vector of sample positions.
        coord = jnp.asarray(dx_or_spacing)
        dx = coord[1:] - coord[:-1]
    else:
        # Evenly sampled traces can use a scalar spacing directly.
        dx = jnp.asarray(dx_or_spacing, dtype=moved.real.dtype)
    # Trapezoids are the average of adjacent samples times the local spacing.
    avs = 0.5 * (moved[..., 1:] + moved[..., :-1]) * dx
    if definite:
        # Definite integration collapses the target axis to one sample.
        out = jnp.sum(avs, axis=-1, keepdims=True)
    else:
        # Cumulative integration keeps the input length by prepending an origin sample.
        zero = jnp.zeros((*moved.shape[:-1], 1), dtype=moved.dtype)
        out = jnp.concatenate([zero, jnp.cumsum(avs, axis=-1)], axis=-1)
    # Restore the original axis order after working in last-axis layout.
    return jnp.moveaxis(out, -1, axis)
