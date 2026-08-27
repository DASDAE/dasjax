"""Unified PatchOp runtime for compiled patch-state execution."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import dascore as dc
import jax
import numpy as np

from ..pytree import _decode_leaf, _encode_leaf


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PatchState:
    """JAX-friendly dynamic patch state."""

    data: Any
    coords: dict[str, Any]

    def tree_flatten(self):
        return ((self.data, self.coords), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        _ = aux_data
        data, coords = children
        return cls(data=data, coords=coords)

    def with_updates(
        self,
        *,
        data: Any | None = None,
        coords: dict[str, Any] | None = None,
    ) -> "PatchState":
        return PatchState(
            data=self.data if data is None else data,
            coords=self.coords if coords is None else coords,
        )


@dataclass(frozen=True)
class CoordSpec:
    """Static reconstruction metadata for one coordinate."""

    name: str
    dims: tuple[str, ...]
    dtype: str
    coord: Any


@dataclass(frozen=True)
class PatchSpec:
    """Static reconstruction metadata for one patch state."""

    dims: tuple[str, ...]
    attrs: Any
    coord_specs: tuple[CoordSpec, ...]

    def apply_meta(self, meta: "MetaDelta | None") -> "PatchSpec":
        """Return a new spec with a metadata delta applied."""
        if meta is None:
            return self
        return PatchSpec(
            dims=self.dims if meta.dims is None else meta.dims,
            attrs=self.attrs if meta.attrs is None else meta.attrs,
            coord_specs=self.coord_specs
            if meta.coord_specs is None
            else meta.coord_specs,
        )


@dataclass(frozen=True)
class MetaDelta:
    """Static metadata changes produced by a PatchOp."""

    dims: tuple[str, ...] | None = None
    attrs: Any | None = None
    coord_specs: tuple[CoordSpec, ...] | None = None


@dataclass(frozen=True)
class OpResult:
    """Normalized output of one compiled patch op."""

    data: Any | None = None
    coords: dict[str, Any] | None = None
    meta: MetaDelta | None = None


def patch_to_state_spec(patch: dc.Patch) -> tuple[PatchState, PatchSpec]:
    """Convert a DASCore patch into dynamic state plus static spec."""
    coord_specs = []
    coord_values: dict[str, Any] = {}
    for name, coord in patch.coords.coord_map.items():
        encoded_values, dtype_token = _encode_leaf(coord.values)
        coord_values[name] = encoded_values
        coord_specs.append(
            CoordSpec(
                name=name,
                dims=tuple(patch.coords.dim_map[name]),
                dtype=dtype_token,
                coord=coord,
            )
        )
    return PatchState(
        data=patch.data,
        coords=coord_values,
    ), PatchSpec(
        dims=tuple(patch.dims),
        attrs=patch.attrs,
        coord_specs=tuple(coord_specs),
    )


def patch_from_state_spec(
    state: PatchState,
    spec: PatchSpec,
    *,
    coerce_numpy: bool = True,
) -> dc.Patch:
    """Rebuild a DASCore patch from dynamic state and static spec."""
    coords: dict[str, Any] = {}
    for coord_spec in spec.coord_specs:
        values = _decode_leaf(state.coords[coord_spec.name], coord_spec.dtype)
        coord = coord_spec.coord.update_data(data=values)
        dims = tuple(coord_spec.dims)
        coords[coord_spec.name] = (
            coord if len(dims) == 1 and dims[0] == coord_spec.name else (dims, coord)
        )
    return dc.Patch(
        data=np.asarray(state.data) if coerce_numpy else state.data,
        coords=coords,
        dims=spec.dims,
        attrs=spec.attrs,
    )


class PatchOp:
    """Base class for unified compiled patch operations."""

    _subclasses: list[type["PatchOp"]] = []
    _register_subclass = True
    mutates_spec = False
    requires_materialized_patch_after = False
    requires_materialized_patch_for_prepare = False
    compile_category = "kernel_fusible"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls is PatchOp:
            return
        if cls.__dict__.get("_register_subclass", True):
            PatchOp._subclasses.append(cls)

    @classmethod
    def iter_subclasses(cls) -> tuple[type["PatchOp"], ...]:
        """Return registered PatchOp subclasses."""
        return tuple(cls._subclasses)

    @classmethod
    def compile_category_table(cls) -> dict[str, tuple[str, ...]]:
        """Group registered PatchOp subclasses by compile category."""
        categories: dict[str, list[str]] = {}
        for subclass in cls._subclasses:
            categories.setdefault(subclass.compile_category, []).append(
                subclass.__name__
            )
        return {key: tuple(sorted(value)) for key, value in sorted(categories.items())}

    @classmethod
    def prepare(cls, patch: dc.Patch, *args: Any, **kwargs: Any) -> "PatchOp":
        """Default prepare: instantiate and bind inferred patch attrs."""
        op = cls(*args, **kwargs)
        object.__setattr__(op, "_prepared_attrs", cls._prepare_attrs(patch, op))
        object.__setattr__(
            op,
            "_prepared_selected",
            cls._prepare_selected(patch, args, kwargs, op),
        )
        cls._validate_kernel_bindings(patch, op)
        return op

    @staticmethod
    def _kernel_parameters(op: "PatchOp") -> tuple[str, ...]:
        params = inspect.signature(op.kernel).parameters
        return tuple(name for name in params if name != "self")

    @classmethod
    def _prepare_attrs(cls, patch: dc.Patch, op: "PatchOp") -> dict[str, Any]:
        prepared: dict[str, Any] = {}
        for name in cls._kernel_parameters(op):
            if name.startswith("attr_"):
                attr_name = name[5:]
                prepared[attr_name] = patch.attrs[attr_name]
        return prepared

    @staticmethod
    def _requires_selected(op: "PatchOp") -> bool:
        return any(
            name.startswith("selected_") for name in PatchOp._kernel_parameters(op)
        )

    @staticmethod
    def _encode_bound_value(value: Any) -> Any:
        encoded, _ = _encode_leaf(np.asarray(value))
        if np.asarray(encoded).shape == ():
            return np.asarray(encoded).item()
        return encoded

    @classmethod
    def _resolve_selected_context(
        cls,
        patch: dc.Patch,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[str, int, Any]:
        _ = args
        selected_dim = kwargs.get("dim")
        if selected_dim is not None:
            if selected_dim not in patch.dims:
                msg = f"Selected dim {selected_dim!r} is not in patch dims {patch.dims!r}."
                raise ValueError(msg)
            return (
                selected_dim,
                patch.get_axis(selected_dim),
                patch.get_coord(selected_dim),
            )
        dim_keys = tuple(key for key in kwargs if key in patch.dims)
        if not dim_keys:
            msg = (
                "Kernel requested selected_* binding but no selected dim was provided."
            )
            raise ValueError(msg)
        if len(dim_keys) > 1:
            msg = (
                "Kernel requested selected_* binding but the call matches multiple dims: "
                f"{dim_keys!r}."
            )
            raise ValueError(msg)
        selected_dim = dim_keys[0]
        return selected_dim, patch.get_axis(selected_dim), patch.get_coord(selected_dim)

    @classmethod
    def _prepare_selected(
        cls,
        patch: dc.Patch,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        op: "PatchOp",
    ) -> dict[str, Any]:
        if not cls._requires_selected(op):
            return {}
        selected_dim, selected_axis, selected_coord = cls._resolve_selected_context(
            patch, args, kwargs
        )
        prepared = {
            "selected_dim": selected_dim,
            "selected_axis": selected_axis,
            "selected_size": len(selected_coord),
            "selected_coord": selected_dim,
            "selected_start": cls._encode_bound_value(selected_coord.min()),
            "selected_stop": cls._encode_bound_value(selected_coord.max()),
        }
        if selected_coord.step is not None:
            prepared["selected_step"] = cls._encode_bound_value(selected_coord.step)
        return prepared

    @classmethod
    def _validate_kernel_bindings(cls, patch: dc.Patch, op: "PatchOp") -> None:
        for name in cls._kernel_parameters(op):
            if name.startswith("coord_"):
                coord_name = name[6:]
                if coord_name not in patch.coords.coord_map:
                    msg = f"Kernel requested unknown coordinate {coord_name!r}."
                    raise ValueError(msg)
            elif name.startswith("selected_"):
                prepared = getattr(op, "_prepared_selected", {})
                if name == "selected_step" and "selected_step" not in prepared:
                    msg = (
                        "Kernel requested selected_step but the selected coordinate "
                        "is not evenly sampled."
                    )
                    raise ValueError(msg)
                if name not in {
                    "selected_axis",
                    "selected_coord",
                    "selected_step",
                    "selected_size",
                    "selected_start",
                    "selected_stop",
                }:
                    msg = f"Unsupported selected binding {name!r}."
                    raise TypeError(msg)

    def _normalize_result(self, raw: Any) -> OpResult:
        if isinstance(raw, OpResult):
            return raw
        if isinstance(raw, dict):
            unknown = set(raw) - {"data", "coords", "meta"}
            if unknown:
                msg = f"Unsupported OpResult keys: {sorted(unknown)!r}"
                raise TypeError(msg)
            return OpResult(
                data=raw.get("data"),
                coords=raw.get("coords"),
                meta=raw.get("meta"),
            )
        msg = f"Kernel {type(self).__name__}.kernel must return OpResult or dict."
        raise TypeError(msg)

    def _bind_kernel_inputs(self, state: PatchState) -> dict[str, Any]:
        bound: dict[str, Any] = {}
        prepared_attrs = getattr(self, "_prepared_attrs", {})
        prepared_selected = getattr(self, "_prepared_selected", {})
        for name in self._kernel_parameters(self):
            if name == "state":
                bound[name] = state
            elif name == "data":
                bound[name] = state.data
            elif name == "coords":
                bound[name] = state.coords
            elif name.startswith("coord_"):
                bound[name] = state.coords[name[6:]]
            elif name.startswith("attr_"):
                bound[name] = prepared_attrs[name[5:]]
            elif name == "selected_coord":
                bound[name] = state.coords[prepared_selected["selected_coord"]]
            elif name.startswith("selected_"):
                bound[name] = prepared_selected[name]
            else:
                msg = f"Unsupported kernel parameter {name!r}."
                raise TypeError(msg)
        return bound

    def apply(self, state: PatchState) -> PatchState:
        """Apply one op to patch state and return updated state."""
        result = self._normalize_result(self.kernel(**self._bind_kernel_inputs(state)))
        next_data = state.data if result.data is None else result.data
        next_coords = state.coords
        if result.coords:
            next_coords = dict(state.coords)
            next_coords.update(result.coords)
        return state.with_updates(data=next_data, coords=next_coords)

    def reconstruct(
        self,
        previous_patch: dc.Patch,
        spec: PatchSpec,
        state: PatchState,
    ) -> dc.Patch:
        """Rebuild a DASCore patch from state after one compiled segment."""
        _ = previous_patch
        return patch_from_state_spec(state, spec)

    def meta_delta(self, spec: PatchSpec) -> MetaDelta | None:
        """Return static spec changes implied by this prepared op."""
        _ = spec
        return None

    def compile_key(self) -> tuple[Any, ...]:
        """Return a stable cache-key component for this prepared op."""
        entries = []
        for key, value in vars(self).items():
            entries.append((key, _freeze_value(value)))
        return (type(self).__name__, tuple(sorted(entries)))


@dataclass(frozen=True)
class EagerPatchOp(PatchOp):
    """PatchOp wrapper that materializes state and delegates to patch_impl."""

    _register_subclass = False
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    mutates_spec = True
    requires_materialized_patch_after = True
    requires_materialized_patch_for_prepare = True
    compile_category = "eager_boundary"
    patch_impl_fn: Callable[..., dc.Patch] | None = None

    @classmethod
    def prepare(cls, patch: dc.Patch, *args: Any, **kwargs: Any) -> "EagerPatchOp":
        _ = patch
        return cls(args=args, kwargs=dict(kwargs))

    def kernel(self, state):
        return {}

    def reconstruct(
        self,
        previous_patch: dc.Patch,
        spec: PatchSpec,
        state: PatchState,
    ) -> dc.Patch:
        current_patch = patch_from_state_spec(state, spec)
        patch_impl = type(self).patch_impl_fn
        assert patch_impl is not None
        _ = previous_patch
        return patch_impl(current_patch, *self.args, **self.kwargs)


def _freeze_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return (
            "ndarray",
            str(value.dtype),
            value.shape,
            tuple(np.asarray(value).reshape(-1).tolist()),
        )
    if isinstance(value, dict):
        return tuple((key, _freeze_value(val)) for key, val in sorted(value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_value(item) for item in value)
    try:
        hash(value)
    except TypeError as exc:
        msg = f"Unhashable prepared value {value!r} cannot be compiled."
        raise TypeError(msg) from exc
    return value
