"""Core authoring API for JAX-backed DASCore patch operations."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import Any, ClassVar, Self

import dascore as dc
import jax
import numpy as np

_DTYPE_TO_CODE = {
    np.dtype("bool").str: 1,
    np.dtype("int32").str: 2,
    np.dtype("int64").str: 3,
    np.dtype("float32").str: 4,
    np.dtype("float64").str: 5,
    np.dtype("complex64").str: 6,
    np.dtype("complex128").str: 7,
    np.dtype("datetime64[ns]").str: 8,
    np.dtype("timedelta64[ns]").str: 9,
}
_CODE_TO_DTYPE = {value: np.dtype(key) for key, value in _DTYPE_TO_CODE.items()}
TIME_DTYPES = (np.dtype("datetime64[ns]"), np.dtype("timedelta64[ns]"))
SUPPORTED_DTYPES = tuple(
    np.dtype(key) for key in _DTYPE_TO_CODE if np.dtype(key) not in TIME_DTYPES
)
_PatchTreeChildren = tuple[Any, tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]]


def _camel_to_snake(name: str) -> str:
    """Convert class names into default pipeline method names."""
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def _matches_dtype(dtype: np.dtype, candidates: tuple[np.dtype, ...]) -> bool:
    return any(np.dtype(dtype) == candidate for candidate in candidates)


def _encode_leaf(array: Any) -> tuple[Any, np.ndarray]:
    arr = np.asarray(array)
    if np.issubdtype(arr.dtype, np.datetime64):
        arr = arr.astype("datetime64[ns]")
    elif np.issubdtype(arr.dtype, np.timedelta64):
        arr = arr.astype("timedelta64[ns]")
    dtype_code = _DTYPE_TO_CODE.get(arr.dtype.str)
    if dtype_code is None:
        msg = f"Unsupported coordinate dtype for JAX patch conversion: {arr.dtype}"
        raise TypeError(msg)
    if _matches_dtype(arr.dtype, SUPPORTED_DTYPES):
        return arr, np.asarray(dtype_code, dtype=np.int32)
    if _matches_dtype(arr.dtype, TIME_DTYPES):
        return arr.view(np.int64), np.asarray(dtype_code, dtype=np.int32)


def _decode_leaf(array: Any, dtype_code: Any) -> np.ndarray:
    out = np.asarray(array)
    dtype = _CODE_TO_DTYPE[int(np.asarray(dtype_code).item())]
    if _matches_dtype(dtype, TIME_DTYPES):
        return out.astype(np.int64, copy=False).view(dtype)
    return out.astype(dtype, copy=False)


def _coord_values_match(left: Any, right: Any) -> bool:
    """Return True when decoded coord values match existing coord values."""
    left_arr = np.asarray(left)
    right_arr = np.asarray(right)
    if left_arr.shape != right_arr.shape:
        return False
    if np.issubdtype(left_arr.dtype, np.datetime64):
        left_i = left_arr.astype("datetime64[ns]", copy=False).view(np.int64)
        right_i = right_arr.astype("datetime64[ns]", copy=False).view(np.int64)
        return bool(np.array_equal(left_i, right_i))
    if np.issubdtype(left_arr.dtype, np.timedelta64):
        left_i = left_arr.astype("timedelta64[ns]", copy=False).view(np.int64)
        right_i = right_arr.astype("timedelta64[ns]", copy=False).view(np.int64)
        return bool(np.array_equal(left_i, right_i))
    return bool(np.array_equal(left_arr, right_arr))


@dataclass(frozen=True)
class PatchBoundary:
    """DASCore-native metadata owned by pipeline boundaries."""

    attrs: Any
    coords: Any
    coord_names: tuple[str, ...]

    @property
    def dims(self) -> tuple[str, ...]:
        """Return ordered data dimension names."""
        return tuple(self.coords.dims)

    def axis(self, dim: str) -> int:
        """Return the data axis for a named dimension."""
        return self.dims.index(dim)

    def coord(self, name: str) -> Any:
        """Return a DASCore coordinate by name."""
        return self.coords.coord_map[name]

    def coord_dims(self, name: str) -> tuple[str, ...]:
        """Return the dimensions associated with a coordinate."""
        return tuple(self.coords.dim_map[name])

    def coord_index(self, name: str) -> int:
        """Return the ordered dynamic coordinate slot for a coordinate name."""
        return self.coord_names.index(name)

    def to_patch(
        self,
        patch_tree: "PatchPyTree",
        *,
        coerce_numpy: bool = True,
    ) -> dc.Patch:
        """Rebuild a DASCore patch by swapping dynamic values into metadata."""
        coord_values = {
            name: _decode_leaf(value, dtype_code)
            for name, value, dtype_code in zip(
                self.coord_names,
                patch_tree.coord_values,
                patch_tree.coord_dtype_codes,
                strict=True,
            )
        }
        coords: dict[str, Any] = {}
        for name, coord in self.coords.coord_map.items():
            out_coord = coord
            if name in coord_values:
                if _coord_values_match(coord_values[name], coord.values):
                    out_coord = coord
                else:
                    out_coord = coord.update_data(data=coord_values[name])
                if out_coord is not coord and coord.units is not None:
                    out_coord = out_coord.set_units(coord.units)
            dims = tuple(self.coords.dim_map[name])
            coords[name] = (
                out_coord if len(dims) == 1 and dims[0] == name else (dims, out_coord)
            )
        return dc.Patch(
            data=np.asarray(patch_tree.data) if coerce_numpy else patch_tree.data,
            coords=coords,
            dims=patch_tree.dims,
            attrs=self.attrs,
        )

    def new(
        self,
        *,
        attrs: Any | None = None,
        coords: Any | None = None,
        coord_names: tuple[str, ...] | None = None,
    ) -> "PatchBoundary":
        """Return a copy with updated DASCore metadata."""
        return replace(
            self,
            attrs=self.attrs if attrs is None else attrs,
            coords=self.coords if coords is None else coords,
            coord_names=self.coord_names if coord_names is None else coord_names,
        )

    def with_metadata(
        self,
        *,
        attrs: Any | None = None,
        coords: Any | None = None,
    ) -> "PatchBoundary":
        """Return a copy with updated DASCore metadata."""
        return self.new(
            attrs=attrs,
            coords=coords,
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PatchPyTree:
    """JAX-friendly dynamic patch state with only dims as static metadata."""

    data: Any
    coord_values: tuple[Any, ...]
    coord_dtype_codes: tuple[Any, ...]
    dims: tuple[str, ...]
    attrs: tuple[Any, ...] = ()

    @classmethod
    def from_patch(cls, patch: dc.Patch) -> tuple[Self, PatchBoundary]:
        """Convert a DASCore patch into a tree plus boundary metadata."""
        coord_names = tuple(patch.coords.coord_map)
        coord_values = []
        coord_dtype_codes = []
        for name, coord in patch.coords.coord_map.items():
            values, dtype_code = _encode_leaf(coord.values)
            coord_values.append(values)
            coord_dtype_codes.append(dtype_code)
        return (
            cls(
                data=patch.data,
                coord_values=tuple(coord_values),
                coord_dtype_codes=tuple(coord_dtype_codes),
                dims=tuple(patch.dims),
            ),
            PatchBoundary(
                attrs=patch.attrs,
                coords=patch.coords,
                coord_names=coord_names,
            ),
        )

    def to_patch(self, *, coerce_numpy: bool = True) -> dc.Patch:
        """Reject reconstruction without boundary metadata."""
        _ = coerce_numpy
        msg = "PatchPyTree.to_patch() requires PatchBoundary.to_patch(patch_tree)."
        raise RuntimeError(msg)

    def tree_flatten(
        self,
    ) -> tuple[_PatchTreeChildren, dict[str, Any]]:
        """Split dynamic JAX leaves from static dims."""
        return (
            (self.data, self.coord_values, self.coord_dtype_codes, self.attrs),
            {"dims": self.dims},
        )

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: dict[str, Any],
        children: _PatchTreeChildren,
    ) -> Self:
        data, coord_values, coord_dtype_codes, attrs = children
        return cls(
            data=data,
            coord_values=coord_values,
            coord_dtype_codes=coord_dtype_codes,
            dims=aux_data["dims"],
            attrs=attrs,
        )

    def coord(self, index: int) -> Any:
        """Return one ordered coordinate-value slot."""
        return self.coord_values[index]

    def replace_coord(self, index: int, value: Any) -> tuple[Any, ...]:
        """Return ordered coordinate-value slots with one value replaced."""
        coords = list(self.coord_values)
        coords[index] = value
        return tuple(coords)

    def new(
        self,
        data: Any | None = None,
        coords: tuple[Any, ...] | None = None,
        *,
        dims: tuple[str, ...] | None = None,
        coord_dtype_codes: tuple[Any, ...] | None = None,
    ) -> Self:
        """Return a copy with updated JAX leaves, matching Patch.new style.

        ``coords`` are the ordered coordinate-value slots carried by this tree.
        Coordinate names and DASCore coordinate objects live in ``PatchBoundary``.
        """
        return replace(
            self,
            data=self.data if data is None else data,
            coord_values=self.coord_values if coords is None else coords,
            coord_dtype_codes=(
                self.coord_dtype_codes
                if coord_dtype_codes is None
                else coord_dtype_codes
            ),
            dims=self.dims if dims is None else dims,
        )


class PatchOperation:
    """Base class for clean, class-authored patch operations."""

    method_name: ClassVar[str | None] = None
    register: ClassVar[bool] = True

    _registry: ClassVar[dict[str, type["PatchOperation"]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not cls.__dict__.get("register", True):
            return
        name = cls.operation_name()
        if name in PatchOperation._registry:
            msg = f"Duplicate patch operation method name {name!r}."
            raise ValueError(msg)
        PatchOperation._registry[name] = cls

    @classmethod
    def operation_name(cls) -> str:
        """Return the public pipeline method name for this operation."""
        name = cls.method_name
        if name is not None:
            return name
        class_name = cls.__name__
        if class_name.endswith("Operation"):
            class_name = class_name.removesuffix("Operation")
        return _camel_to_snake(class_name)

    def bind(self, boundary: PatchBoundary) -> Self:
        """Resolve DASCore metadata into static kernel parameters."""
        _ = boundary
        return self

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        """Apply the JAX-compatible dynamic transform."""
        return patch_tree

    def update_boundary(self, boundary: PatchBoundary) -> PatchBoundary:
        """Plan DASCore metadata changes without reading computed array values."""
        return boundary

    def update_boundary_from_result(
        self,
        patch_tree: PatchPyTree,
        boundary: PatchBoundary,
    ) -> PatchBoundary:
        """Update metadata from computed results, forcing a segment boundary."""
        _ = patch_tree
        return boundary

    @classmethod
    def overrides(cls, method_name: str) -> bool:
        """Return True if a subclass overrides one base hook."""
        return getattr(cls, method_name) is not getattr(PatchOperation, method_name)


def get_patch_operation(name: str) -> type[PatchOperation]:
    """Return a class-authored patch operation by public method name."""
    try:
        return PatchOperation._registry[name]
    except KeyError as exc:
        raise AttributeError(f"Unknown jax patch operation {name!r}.") from exc


def iter_patch_operations() -> Iterator[type[PatchOperation]]:
    """Iterate over class-authored patch operations in registration order."""
    return iter(PatchOperation._registry.values())


def list_patch_operations() -> tuple[str, ...]:
    """List class-authored operation names in registration order."""
    return tuple(PatchOperation._registry)
