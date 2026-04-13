"""JAX pytree registration for DASCore patches."""

from __future__ import annotations

from typing import Any

import dascore as dc
import jax
import numpy as np

SUPPORTED_DTYPES = (np.bool_, np.integer, np.floating, np.complexfloating)
TIME_DTYPES = (np.datetime64, np.timedelta64)


def _matches_dtype(dtype: np.dtype, candidates: tuple[type[Any], ...]) -> bool:
    """Return True if dtype is a subdtype of any candidate."""
    return any(np.issubdtype(dtype, candidate) for candidate in candidates)


def _encode_leaf(array: Any) -> tuple[Any, str]:
    """Return a JAX-friendly leaf and a dtype token for reconstruction."""
    arr = np.asarray(array)
    dtype_token = arr.dtype.str
    if _matches_dtype(arr.dtype, SUPPORTED_DTYPES):
        return arr, dtype_token
    if _matches_dtype(arr.dtype, TIME_DTYPES):
        return arr.view(np.int64), dtype_token
    msg = f"Unsupported coordinate dtype for JAX pytree registration: {arr.dtype}"
    raise TypeError(msg)


def _decode_leaf(array: Any, dtype_token: str) -> np.ndarray:
    """Restore an encoded leaf to its original dtype."""
    out = np.asarray(array)
    dtype = np.dtype(dtype_token)
    if _matches_dtype(dtype, TIME_DTYPES):
        return out.astype(np.int64, copy=False).view(dtype)
    return out.astype(dtype, copy=False)


def patch_to_leaves(patch: dc.Patch) -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
    """Split a patch into dynamic leaves and static reconstruction metadata."""
    coord_names = tuple(patch.coords.coord_map)
    coord_leaves = []
    coord_meta = []
    for name in coord_names:
        coord = patch.coords.coord_map[name]
        encoded_values, dtype_token = _encode_leaf(coord.values)
        coord_leaves.append(encoded_values)
        coord_meta.append(
            {
                "name": name,
                "dims": tuple(patch.coords.dim_map[name]),
                "dtype": dtype_token,
                "coord": coord,
            }
        )
    aux_data = {
        "dims": tuple(patch.dims),
        "attrs": patch.attrs,
        "coord_meta": tuple(coord_meta),
    }
    return patch.data, tuple(coord_leaves), aux_data


def patch_from_leaves(
    data: Any,
    coord_leaves: tuple[Any, ...],
    aux_data: dict[str, Any],
    *,
    coerce_numpy: bool = True,
) -> dc.Patch:
    """Rebuild a patch from pytree leaves and static metadata."""
    coords: dict[str, Any] = {}
    for meta, leaf in zip(aux_data["coord_meta"], coord_leaves, strict=True):
        values = _decode_leaf(leaf, meta["dtype"])
        coord = meta["coord"].update_data(data=values)
        dims = tuple(meta["dims"])
        coords[meta["name"]] = (
            coord if len(dims) == 1 and dims[0] == meta["name"] else (dims, coord)
        )
    return dc.Patch(
        data=np.asarray(data) if coerce_numpy else data,
        coords=coords,
        dims=aux_data["dims"],
        attrs=aux_data["attrs"],
    )


def _flatten_patch(patch: dc.Patch) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """JAX pytree flatten hook."""
    data, coord_leaves, aux_data = patch_to_leaves(patch)
    return (data, *coord_leaves), aux_data


def _unflatten_patch(aux_data: dict[str, Any], children: tuple[Any, ...]) -> dc.Patch:
    """JAX pytree unflatten hook."""
    data, *coord_leaves = children
    return patch_from_leaves(data, tuple(coord_leaves), aux_data)


jax.tree_util.register_pytree_node(dc.Patch, _flatten_patch, _unflatten_patch)
