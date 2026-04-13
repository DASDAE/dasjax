"""Shared helpers used by operation-family modules."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import dascore as dc
import numpy as np
from dascore.constants import PatchType
from dascore.exceptions import ParameterError

from .types import OperationSpec


def get_axis(patch: PatchType, dim: str | None) -> int | None:
    if dim is None:
        return None
    try:
        return patch.dims.index(dim)
    except ValueError as exc:
        msg = f"Patch does not contain dimension {dim!r}."
        raise ValueError(msg) from exc


def get_axis_from_dims(dims: tuple[str, ...], dim: str | None) -> int | None:
    if dim is None:
        return None
    try:
        return dims.index(dim)
    except ValueError as exc:
        msg = f"Patch does not contain dimension {dim!r}."
        raise ValueError(msg) from exc


def update_patch(patch: PatchType, data: Any) -> PatchType:
    return patch.update(data=np.asarray(data))


def prepare_operation_call(
    operation: OperationSpec,
    patch: PatchType,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if operation.prepare_call is None:
        return args, kwargs
    return operation.prepare_call(patch, args, kwargs)


def offset_patch(patch: PatchType, offset: float) -> PatchType:
    return patch.update(data=patch.data + offset)


def baseline_update(
    patch: PatchType,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    fn: Callable[..., Any],
) -> PatchType:
    return patch.update(data=np.asarray(fn(patch.data, *args, **kwargs)))


def baseline_patch_method(
    patch: PatchType,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    method_name: str,
) -> PatchType:
    return getattr(patch, method_name)(*args, **kwargs)


def get_evenly_sampled_dim(patch: PatchType) -> str:
    for dim in reversed(patch.dims):
        if patch.get_coord(dim).step is not None:
            return dim
    msg = f"Patch does not contain an evenly sampled dimension: {patch.dims}"
    raise ParameterError(msg)


def get_preferred_transform_dim(patch: PatchType) -> str:
    for dim in reversed(patch.dims):
        if patch.get_coord(dim).step is not None:
            return dim
    return patch.dims[-1]


def resolve_even_dim_call(
    patch: PatchType,
    **kwargs: Any,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    out = dict(kwargs)
    out["dim"] = get_preferred_transform_dim(patch)
    return (), out


def prepare_complex_patch(patch: PatchType) -> PatchType:
    data = np.asarray(patch.data, dtype=np.complex128)
    return patch.update(data=data + 1j * data)


def prepare_fourier_patch(patch: PatchType) -> PatchType:
    from dascore.transform.fourier import dft as dc_dft

    dim = get_preferred_transform_dim(patch)
    try:
        return dc_dft.func(patch, dim=dim, real=True, pad=False)
    except Exception:
        return patch


def prepare_even_patch_for_dim(patch: PatchType) -> PatchType:
    dim = get_preferred_transform_dim(patch)
    coord = patch.get_coord(dim)
    if coord.step is not None:
        return patch
    new_coord = dc.get_coord(start=0, stop=len(coord), step=1, units=coord.units)
    return patch.update(coords=patch.coords.update(**{dim: new_coord}))


def validate_registry(operations: tuple[OperationSpec, ...]) -> None:
    seen: set[str] = set()
    for operation in operations:
        if operation.name in seen:
            msg = f"Duplicate jax patch operation {operation.name!r}."
            raise ValueError(msg)
        seen.add(operation.name)
        if (
            operation.execution_policy.value == "leaf"
            and operation.leaf_transform is None
        ):
            msg = f"Leaf operation {operation.name!r} must define leaf_transform."
            raise ValueError(msg)
        if (
            operation.execution_policy.value == "patch"
            and operation.leaf_transform is not None
        ):
            msg = f"Patch operation {operation.name!r} must not define leaf_transform."
            raise ValueError(msg)
