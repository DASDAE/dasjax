"""Basic array kernels that do not require callbacks or patch metadata."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp


def identity_kernel(data: Any) -> Any:
    """Return the input unchanged."""
    return data


def scale_kernel(data: Any, factor: float) -> Any:
    """Scale data by a constant factor."""
    # Normalize to a JAX array so eager and compiled callers share the same path.
    return jnp.asarray(data) * factor


def add_kernel(data: Any, value: float) -> Any:
    """Add a constant value to the data."""
    # Scalar addition is elementwise once the input is in JAX form.
    return jnp.asarray(data) + value


def abs_kernel(data: Any) -> Any:
    """Take the absolute value of the data."""
    # `jnp.abs` preserves complex magnitude semantics automatically.
    return jnp.abs(jnp.asarray(data))


def clip_kernel(data: Any, min_value: float, max_value: float) -> Any:
    """Clip data to the provided range."""
    # Clipping is fully elementwise, so there is no axis handling here.
    return jnp.clip(jnp.asarray(data), min_value, max_value)


def real_kernel(data: Any) -> Any:
    """Return the real part of the array."""
    # Complex inputs drop their imaginary part; real inputs pass through.
    return jnp.real(jnp.asarray(data))


def imag_kernel(data: Any) -> Any:
    """Return the imaginary part of the array."""
    # Real inputs become zeros with the same broadcastable shape.
    return jnp.imag(jnp.asarray(data))


def angle_kernel(data: Any) -> Any:
    """Return the phase angle of the array."""
    # `jnp.angle` matches NumPy-style phase extraction for complex numbers.
    return jnp.angle(jnp.asarray(data))


def conj_kernel(data: Any) -> Any:
    """Return the complex conjugate of the array."""
    # Real dtypes are unchanged; complex dtypes flip the imaginary sign.
    return jnp.conj(jnp.asarray(data))


def flip_kernel(data: Any, axes: tuple[int, ...]) -> Any:
    """Flip an array along one or more axes."""
    # Axis resolution already happened in operation preparation.
    return jnp.flip(jnp.asarray(data), axis=axes)


def roll_kernel(data: Any, shift: int, axis: int) -> Any:
    """Roll an array along one axis."""
    # The kernel only handles sample movement; coord updates stay at patch level.
    return jnp.roll(jnp.asarray(data), shift, axis=axis)


def standardize_kernel(data: Any, axis: int) -> Any:
    """Standardize an array along one axis."""
    # Work with a JAX array so reduction and broadcasting stay on device.
    arr = jnp.asarray(data)
    # Keepdims preserves a shape that can broadcast back across the input axis.
    mean = jnp.mean(arr, axis=axis, keepdims=True)
    std = jnp.std(arr, axis=axis, keepdims=True)
    # Zero-std handling, when needed, is coordinated in the operation layer.
    return (arr - mean) / std


def apply_1d_weight_kernel(data: Any, axis: int, weight: Any) -> Any:
    """Multiply an array by a 1D weight broadcast along one axis."""
    # Convert once so the incoming array and weight share dtype/device semantics.
    arr = jnp.asarray(data)
    weight_arr = jnp.asarray(weight, dtype=arr.dtype)
    # Build an explicit broadcast shape with the target axis expanded in place.
    shape = [1] * arr.ndim
    shape[axis] = weight_arr.shape[0]
    # Reshape rather than tile so XLA can broadcast without materializing copies.
    return arr * jnp.reshape(weight_arr, shape)
