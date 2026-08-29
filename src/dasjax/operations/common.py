"""Shared helpers for patch operation implementations."""

from __future__ import annotations

from dataclasses import fields
from typing import Any

import dascore as dc
import numpy as np
from dascore.units import get_quantity

from dasjax.core import PatchBoundary, PatchPyTree


def replace(obj: Any, **changes: Any) -> Any:
    """Copy frozen operation instances without calling custom constructors."""
    out = object.__new__(type(obj))
    for field in fields(obj):
        object.__setattr__(
            out, field.name, changes.get(field.name, getattr(obj, field.name))
        )
    return out


def dummy_patch(boundary: PatchBoundary, dtype=np.float64) -> dc.Patch:
    """Create a zero-data patch with metadata from a pipeline boundary."""
    return dc.Patch(
        data=np.zeros(boundary.coords.shape, dtype=dtype),
        coords=boundary.coords,
        dims=boundary.dims,
        attrs=boundary.attrs,
    )


def tree_boundary_from_patch(patch: dc.Patch) -> tuple[PatchPyTree, PatchBoundary]:
    """Convert a DASCore patch into pipeline tree and boundary objects."""
    return PatchPyTree.from_patch(patch)


def get_data_units_from_dims(
    boundary: PatchBoundary,
    dims: tuple[str, ...],
    operator,
) -> Any:
    """Apply a unit operator between data units and dimension units."""
    if (data_units := get_quantity(boundary.attrs.data_units)) is None:
        return None
    dim_units = None
    for dim_name in dims:
        dim_unit = get_quantity(boundary.coord(dim_name).units)
        if dim_unit is None:
            continue
        dim_units = dim_unit if dim_units is None else dim_unit * dim_units
    return operator(data_units, dim_units) if dim_units is not None else data_units
