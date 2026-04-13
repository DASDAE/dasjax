"""Executable implementations for basic operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from dascore.constants import PatchType
from dascore.proc.basic import flip as dc_flip
from dascore.proc.basic import roll as dc_roll
from dascore.proc.basic import standardize as dc_standardize

from dasjax import kernels

from ..common import get_axis_from_dims, update_patch
from ..patch_ops import EagerPatchOp, PatchOp


@dataclass(frozen=True)
class IdentityOp(PatchOp):
    """Compiled identity operation."""

    def kernel(self, data):
        return {"data": kernels.identity_kernel(data)}


@dataclass(frozen=True)
class ScaleOp(PatchOp):
    """Compiled scale operation with signature-inferred state binding."""

    factor: float

    def kernel(self, data):
        return {"data": kernels.scale_kernel(data, self.factor)}


@dataclass(frozen=True)
class AddOp(PatchOp):
    """Compiled add operation."""

    value: float

    def kernel(self, data):
        return {"data": kernels.add_kernel(data, self.value)}


@dataclass(frozen=True)
class AbsOp(PatchOp):
    """Compiled abs operation."""

    def kernel(self, data):
        return {"data": kernels.abs_kernel(data)}


@dataclass(frozen=True)
class ClipOp(PatchOp):
    """Compiled clip operation."""

    min_value: float
    max_value: float

    def kernel(self, data):
        return {"data": kernels.clip_kernel(data, self.min_value, self.max_value)}


@dataclass(frozen=True)
class RealOp(PatchOp):
    """Compiled real operation."""

    def kernel(self, data):
        return {"data": kernels.real_kernel(data)}


@dataclass(frozen=True)
class ImagOp(PatchOp):
    """Compiled imag operation."""

    def kernel(self, data):
        return {"data": kernels.imag_kernel(data)}


@dataclass(frozen=True)
class AngleOp(PatchOp):
    """Compiled angle operation."""

    def kernel(self, data):
        return {"data": kernels.angle_kernel(data)}


@dataclass(frozen=True)
class ConjOp(PatchOp):
    """Compiled conj operation."""

    def kernel(self, data):
        return {"data": kernels.conj_kernel(data)}


@dataclass(frozen=True)
class FlipOp(PatchOp):
    """Compiled flip operation over patch state."""

    dims: tuple[str, ...]
    data_axes: tuple[int, ...]
    coord_axes: dict[str, tuple[int, ...]]
    flip_coords: bool = True
    mutates_spec = False

    @classmethod
    def prepare(
        cls,
        patch: PatchType,
        *dims: str,
        flip_coords: bool = True,
    ) -> "FlipOp":
        resolved_dims = tuple(dims) if dims else tuple(patch.dims)
        data_axes = tuple(patch.get_axis(dim) for dim in resolved_dims)
        coord_axes: dict[str, tuple[int, ...]] = {}
        if flip_coords:
            for name, coord_dims in patch.coords.dim_map.items():
                axes = tuple(
                    idx
                    for idx, coord_dim in enumerate(coord_dims)
                    if coord_dim in resolved_dims
                )
                if axes:
                    coord_axes[name] = axes
        return cls(
            dims=resolved_dims,
            data_axes=data_axes,
            coord_axes=coord_axes,
            flip_coords=flip_coords,
        )

    def kernel(self, state):
        coords = state.coords
        out = {"data": jnp.flip(state.data, axis=self.data_axes)}
        if self.flip_coords:
            new_coords = {}
            for name, axes in self.coord_axes.items():
                new_coords[name] = jnp.flip(coords[name], axis=axes)
            if new_coords:
                out["coords"] = new_coords
        return out


@dataclass(frozen=True)
class RollOp(PatchOp):
    """Compiled roll operation using selected-axis inference."""

    shift: int
    selected_dim: str
    requires_materialized_patch_for_prepare = False

    @classmethod
    def prepare(
        cls,
        patch: PatchType,
        samples: bool = False,
        update_coord: bool = False,
        **kwargs,
    ) -> "RollOp":
        if update_coord:
            raise NotImplementedError(
                "Compiled roll currently requires update_coord=False."
            )
        selected_dim, _, coord = cls._resolve_selected_context(patch, (), kwargs)
        shift = int(coord.get_sample_count(kwargs[selected_dim], samples=samples))
        op = cls(shift=shift, selected_dim=selected_dim)
        object.__setattr__(op, "_prepared_attrs", {})
        object.__setattr__(
            op,
            "_prepared_selected",
            cls._prepare_selected(patch, (), {"dim": selected_dim}, op),
        )
        cls._validate_kernel_bindings(patch, op)
        return op

    def kernel(self, data, selected_axis):
        return {"data": kernels.roll_kernel(data, shift=self.shift, axis=selected_axis)}


@dataclass(frozen=True)
class StandardizeOp(EagerPatchOp):
    """Unified eager-backed standardize operation."""

    patch_impl_fn = staticmethod(dc_standardize.func)


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
        raise NotImplementedError(
            "Compiled roll currently requires update_coord=False."
        )


def guard_compiled_roll_case(
    patch: PatchType,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[type[Exception], str] | None:
    _ = patch, args
    if kwargs.get("update_coord", False):
        return (
            NotImplementedError,
            "Compiled roll currently requires update_coord=False.",
        )
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
