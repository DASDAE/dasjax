"""Tau-p transform operation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

import dascore as dc
import numpy as np
from dascore.exceptions import ParameterError
from dascore.units import convert_units

from dasjax.core import PatchBoundary, PatchOperation, PatchPyTree

from .. import kernels
from .common import dummy_patch, replace, tree_boundary_from_patch


@dataclass(frozen=True)
class TauP(PatchOperation):
    """Compute a linear tau-p transform."""

    method_name = "tau_p"

    velocities: Any
    distances: Any = None
    dt: float = 1.0
    p_values: Any = None
    out_boundary: PatchBoundary | None = None
    out_coords: tuple[Any, ...] | None = None
    out_dtype_codes: tuple[Any, ...] | None = None
    out_dims: tuple[str, ...] | None = None

    def bind(self, boundary: PatchBoundary) -> Self:
        velocities = np.asarray(convert_units(self.velocities, to_units="m/s"))
        if np.any(velocities <= 0):
            raise ParameterError("Input velocities must be positive.")
        if not np.all(np.diff(velocities) > 0):
            raise ParameterError("Input velocities must be monotonically increasing.")
        patch = dummy_patch(boundary).convert_units(distance="m", time="s").transpose(
            "distance", "time"
        )
        out_patch = patch.tau_p(velocities)
        out_tree, out_boundary = tree_boundary_from_patch(out_patch)
        dist = patch.get_coord("distance")
        distances = (
            np.arange(len(dist), dtype=np.float64) * float(dc.to_float(dist.step))
            if dist.evenly_sampled
            else np.asarray(dc.to_float(dist.values), dtype=np.float64)
        )
        return replace(
            self,
            velocities=velocities,
            distances=distances,
            dt=float(dc.to_float(patch.get_coord("time").step)),
            p_values=1.0 / velocities,
            out_boundary=out_boundary,
            out_coords=out_tree.coord_values,
            out_dtype_codes=out_tree.coord_dtype_codes,
            out_dims=out_tree.dims,
        )

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        data = kernels.tau_p_kernel(
            patch_tree.data,
            distances=self.distances,
            dt=self.dt,
            p_values=self.p_values,
        )
        return patch_tree.new(
            data=data,
            coords=self.out_coords,
            coord_dtype_codes=self.out_dtype_codes,
            dims=self.out_dims,
        )

    def update_boundary(self, boundary: PatchBoundary) -> PatchBoundary:
        _ = boundary
        assert self.out_boundary is not None
        return self.out_boundary
