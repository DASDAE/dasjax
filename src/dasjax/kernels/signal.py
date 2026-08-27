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

    if norm in {"l1", "l2"}:
        # The float exponent promotes ints so the powers cannot overflow a
        # narrow dtype, exactly as DASCore's own kernel does.
        order = float(norm[-1])
        total = jnp.nansum(jnp.abs(data) ** order, axis=axis)
        divisor = jnp.expand_dims(total ** (1 / order), axis=axis)
    elif norm == "max":
        # The peak absolute value, not the signed maximum: the two differ for
        # any slice whose most negative sample outweighs its most positive one.
        divisor = jnp.expand_dims(jnp.nanmax(jnp.abs(data), axis=axis), axis=axis)
    elif norm == "bit":
        # Bit normalization keeps only the sign/phase of each sample.
        divisor = jnp.abs(data)
    else:
        msg = f"Unsupported normalization mode {norm!r}."
        raise ValueError(msg)

    # A zero divisor means there is nothing but zeros and nulls to scale, so
    # divide those by one: the zeros stay zero and the nulls stay null.
    one = jnp.asarray(1, dtype=divisor.dtype)
    out = data / jnp.where(divisor == 0, one, divisor)
    # JAX and numpy disagree about the dtype of an integer division: int32
    # over int32 is float32 here and float64 there. Ask numpy which it would
    # have produced rather than guessing at the promotion rules.
    return out.astype((np.ones((), data.dtype) / np.ones((), divisor.dtype)).dtype)


def differentiate_kernel(
    data: Any,
    axis: int,
    dx_or_spacing: float | np.ndarray,
    order: int = 2,
    step: int = 1,
) -> Any:
    """Differentiate an array along one axis."""
    arr = jnp.asarray(data)
    if order != 2:
        msg = "Compiled differentiate currently requires order=2."
        raise NotImplementedError(msg)
    moved = jnp.moveaxis(arr, axis, 0)
    if step > 1:
        out = jnp.zeros_like(moved)
        for step_index in range(step):
            sub = moved[step_index::step]
            spacing = (
                jnp.asarray(dx_or_spacing)[step_index::step]
                if np.ndim(dx_or_spacing)
                else dx_or_spacing * step
            )
            out = out.at[step_index::step].set(_differentiate_axis0(sub, spacing))
        return jnp.moveaxis(out, 0, axis)
    spacing = jnp.asarray(dx_or_spacing) if np.ndim(dx_or_spacing) else dx_or_spacing
    return jnp.moveaxis(_differentiate_axis0(moved, spacing), 0, axis)


def _differentiate_axis0(data: Any, spacing: float | Any) -> Any:
    """Differentiate along axis 0 using NumPy edge_order=2 formulas."""
    arr = jnp.asarray(data)
    if arr.shape[0] < 3:
        msg = (
            "Shape of array too small to calculate a numerical gradient, "
            "at least (edge_order + 1) elements are required."
        )
        raise ValueError(msg)
    out = jnp.empty_like(arr)
    if np.ndim(spacing) == 0:
        dx = jnp.asarray(spacing, dtype=arr.real.dtype)
        out = out.at[1:-1].set((arr[2:] - arr[:-2]) / (2.0 * dx))
        out = out.at[0].set((-1.5 * arr[0] + 2.0 * arr[1] - 0.5 * arr[2]) / dx)
        out = out.at[-1].set((0.5 * arr[-3] - 2.0 * arr[-2] + 1.5 * arr[-1]) / dx)
        return out

    coord_values = jnp.asarray(spacing, dtype=arr.real.dtype)
    coord_diffs = (
        coord_values[1:] - coord_values[:-1]
        if coord_values.shape[0] == arr.shape[0]
        else coord_values
    )
    dx1 = coord_diffs[:-1]
    dx2 = coord_diffs[1:]
    a = -(dx2) / (dx1 * (dx1 + dx2))
    b = (dx2 - dx1) / (dx1 * dx2)
    c = dx1 / (dx2 * (dx1 + dx2))
    reshape = (-1,) + (1,) * (arr.ndim - 1)
    out = out.at[1:-1].set(
        a.reshape(reshape) * arr[:-2]
        + b.reshape(reshape) * arr[1:-1]
        + c.reshape(reshape) * arr[2:]
    )

    first_dx1 = coord_diffs[0]
    first_dx2 = coord_diffs[1]
    out = out.at[0].set(
        (-(2.0 * first_dx1 + first_dx2) / (first_dx1 * (first_dx1 + first_dx2)))
        * arr[0]
        + ((first_dx1 + first_dx2) / (first_dx1 * first_dx2)) * arr[1]
        - (first_dx1 / (first_dx2 * (first_dx1 + first_dx2))) * arr[2]
    )

    last_dx1 = coord_diffs[-2]
    last_dx2 = coord_diffs[-1]
    out = out.at[-1].set(
        (last_dx2 / (last_dx1 * (last_dx1 + last_dx2))) * arr[-3]
        - ((last_dx2 + last_dx1) / (last_dx1 * last_dx2)) * arr[-2]
        + ((2.0 * last_dx2 + last_dx1) / (last_dx2 * (last_dx1 + last_dx2))) * arr[-1]
    )
    return out


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
