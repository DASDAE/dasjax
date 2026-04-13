"""Executable implementations for basic operations."""

from __future__ import annotations

from typing import Any

import dascore as dc
from dascore.constants import PatchType
from dascore.proc.basic import flip as dc_flip
from dascore.proc.basic import roll as dc_roll
from dascore.proc.basic import standardize as dc_standardize

from dasjax import kernels

from ..common import get_axis_from_dims, update_patch


def identity_patch(patch: PatchType) -> PatchType:
    return patch


def identity_leaves(
    data: Any,
    coord_leaves: tuple[Any, ...],
    *,
    dims: tuple[str, ...],
) -> tuple[Any, tuple[Any, ...]]:
    _ = dims
    return kernels.identity_kernel(data), coord_leaves


def scale_patch(patch: PatchType, factor: float) -> PatchType:
    return update_patch(patch, kernels.scale_kernel(patch.data, factor))


def scale_leaves(
    data: Any,
    coord_leaves: tuple[Any, ...],
    factor: float,
    *,
    dims: tuple[str, ...],
) -> tuple[Any, tuple[Any, ...]]:
    _ = dims
    return kernels.scale_kernel(data, factor), coord_leaves


def add_patch(patch: PatchType, value: float) -> PatchType:
    return update_patch(patch, kernels.add_kernel(patch.data, value))


def add_leaves(
    data: Any,
    coord_leaves: tuple[Any, ...],
    value: float,
    *,
    dims: tuple[str, ...],
) -> tuple[Any, tuple[Any, ...]]:
    _ = dims
    return kernels.add_kernel(data, value), coord_leaves


def abs_patch(patch: PatchType) -> PatchType:
    return update_patch(patch, kernels.abs_kernel(patch.data))


def abs_leaves(
    data: Any,
    coord_leaves: tuple[Any, ...],
    *,
    dims: tuple[str, ...],
) -> tuple[Any, tuple[Any, ...]]:
    _ = dims
    return kernels.abs_kernel(data), coord_leaves


def clip_patch(patch: PatchType, min_value: float, max_value: float) -> PatchType:
    return update_patch(patch, kernels.clip_kernel(patch.data, min_value, max_value))


def clip_leaves(
    data: Any,
    coord_leaves: tuple[Any, ...],
    min_value: float,
    max_value: float,
    *,
    dims: tuple[str, ...],
) -> tuple[Any, tuple[Any, ...]]:
    _ = dims
    return kernels.clip_kernel(data, min_value, max_value), coord_leaves


def real_patch(patch: PatchType) -> PatchType:
    return update_patch(patch, kernels.real_kernel(patch.data))


def real_leaves(
    data: Any,
    coord_leaves: tuple[Any, ...],
    *,
    dims: tuple[str, ...],
) -> tuple[Any, tuple[Any, ...]]:
    _ = dims
    return kernels.real_kernel(data), coord_leaves


def imag_patch(patch: PatchType) -> PatchType:
    return update_patch(patch, kernels.imag_kernel(patch.data))


def imag_leaves(
    data: Any,
    coord_leaves: tuple[Any, ...],
    *,
    dims: tuple[str, ...],
) -> tuple[Any, tuple[Any, ...]]:
    _ = dims
    return kernels.imag_kernel(data), coord_leaves


def angle_patch(patch: PatchType) -> PatchType:
    return update_patch(patch, kernels.angle_kernel(patch.data))


def angle_leaves(
    data: Any,
    coord_leaves: tuple[Any, ...],
    *,
    dims: tuple[str, ...],
) -> tuple[Any, tuple[Any, ...]]:
    _ = dims
    return kernels.angle_kernel(data), coord_leaves


def conj_patch(patch: PatchType) -> PatchType:
    return update_patch(patch, kernels.conj_kernel(patch.data))


def conj_leaves(
    data: Any,
    coord_leaves: tuple[Any, ...],
    *,
    dims: tuple[str, ...],
) -> tuple[Any, tuple[Any, ...]]:
    _ = dims
    return kernels.conj_kernel(data), coord_leaves


def flip_patch(
    patch: PatchType,
    *dims: str,
    flip_coords: bool = True,
) -> PatchType:
    return dc_flip.func(patch, *dims, flip_coords=flip_coords)


def prepare_roll_call(
    patch: PatchType,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    _ = args
    call_kwargs = dict(kwargs)
    samples = bool(call_kwargs.pop("samples", False))
    update_coord = bool(call_kwargs.pop("update_coord", False))
    dim = next(key for key in call_kwargs if key in patch.dims)
    axis = patch.get_axis(dim)
    coord = patch.get_coord(dim)
    shift = int(coord.get_sample_count(call_kwargs[dim], samples=samples))
    return (), {"axis": axis, "shift": shift, "update_coord": update_coord}


def validate_roll_compiled_input(
    patch: PatchType,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    _ = patch, args
    if kwargs.get("update_coord", False):
        raise NotImplementedError("Compiled roll currently requires update_coord=False.")


def guard_compiled_roll_case(
    patch: PatchType,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[type[Exception], str] | None:
    _ = patch, args
    if kwargs.get("update_coord", False):
        return (NotImplementedError, "Compiled roll currently requires update_coord=False.")
    return None


def roll_patch(
    patch: PatchType,
    samples: bool = False,
    update_coord: bool = False,
    **kwargs,
) -> PatchType:
    return dc_roll.func(patch, samples=samples, update_coord=update_coord, **kwargs)


def roll_leaves(
    data: Any,
    coord_leaves: tuple[Any, ...],
    *,
    dims: tuple[str, ...],
    axis: int,
    shift: int,
    update_coord: bool = False,
) -> tuple[Any, tuple[Any, ...]]:
    _ = dims, update_coord
    return kernels.roll_kernel(data, shift=shift, axis=axis), coord_leaves


def standardize_patch(patch: PatchType, dim: str) -> PatchType:
    return dc_standardize.func(patch, dim=dim)


def standardize_leaves(
    data: Any,
    coord_leaves: tuple[Any, ...],
    *,
    dims: tuple[str, ...],
    dim: str,
) -> tuple[Any, tuple[Any, ...]]:
    axis = get_axis_from_dims(dims, dim)
    return kernels.standardize_kernel(data, axis=axis), coord_leaves
