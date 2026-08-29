"""Dispersion transform operation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

import numpy as np
from dascore.exceptions import ParameterError

from dasjax.core import PatchBoundary, PatchOperation, PatchPyTree

from .. import kernels
from .common import dummy_patch, replace, tree_boundary_from_patch


@dataclass(frozen=True)
class DispersionPhaseShift(PatchOperation):
    """Compute a phase-shift dispersion image."""

    phase_velocities: Any
    approx_resolution: float | None = None
    approx_freq: tuple[float, float] | None = None
    distances: Any = None
    velocities: Any = None
    nf: int = 0
    first_live_f: int = 0
    last_live_f: int = 0
    fs: float = 1.0
    out_boundary: PatchBoundary | None = None
    out_coords: tuple[Any, ...] | None = None
    out_dtype_codes: tuple[Any, ...] | None = None
    out_dims: tuple[str, ...] | None = None

    def bind(self, boundary: PatchBoundary) -> Self:
        velocities = np.asarray(self.phase_velocities, dtype=np.float64)
        if not np.all(np.diff(velocities) > 0):
            raise ParameterError(
                "Velocities for dispersion must be monotonically increasing"
            )
        if np.amin(velocities) <= 0:
            raise ParameterError("Velocities must be positive.")
        if self.approx_resolution is not None and self.approx_resolution <= 0:
            raise ParameterError("Frequency resolution has to be positive")
        patch = (
            dummy_patch(boundary)
            .convert_units(distance="m")
            .transpose("distance", "time")
        )
        time = patch.coords.get_array("time")
        dt = (time[1] - time[0]) / np.timedelta64(1, "s")
        fs = 1 / dt
        if not self.approx_freq:
            approx_min_freq = 0
            approx_max_freq = 0.5 / dt
        else:
            approx_min_freq, approx_max_freq = self.approx_freq
            if approx_min_freq <= 0 or approx_max_freq <= 0:
                raise ParameterError(
                    "Minimal and maximal frequencies have to be positive"
                )
            if approx_min_freq >= approx_max_freq:
                raise ParameterError(
                    "Maximal frequency needs to be larger than minimal frequency"
                )
            if approx_min_freq >= 0.5 / dt or approx_max_freq >= 0.5 / dt:
                raise ParameterError("Frequency range cannot exceed Nyquist")
        nt = len(time)
        if self.approx_resolution is not None:
            nf = int(nt * (fs / nt) / self.approx_resolution)
        else:
            nf = nt
        freq = np.arange(nf) * fs / (nf - 1)
        omega = 2 * np.pi * freq
        first_live_f = int(np.argmax(omega >= 2 * np.pi * approx_min_freq))
        last_live_f = int(np.argmax(omega >= 2 * np.pi * approx_max_freq))
        if last_live_f - first_live_f < 1:
            raise ParameterError(
                "Combination of frequency resolution and range is not an array"
            )
        out_patch = patch.dispersion_phase_shift(
            velocities,
            approx_resolution=self.approx_resolution,
            approx_freq=self.approx_freq,
        )
        out_tree, out_boundary = tree_boundary_from_patch(out_patch)
        return replace(
            self,
            velocities=velocities,
            distances=np.asarray(patch.coords.get_array("distance"), dtype=np.float64),
            nf=nf,
            first_live_f=first_live_f,
            last_live_f=last_live_f,
            fs=float(fs),
            out_boundary=out_boundary,
            out_coords=out_tree.coord_values,
            out_dtype_codes=out_tree.coord_dtype_codes,
            out_dims=out_tree.dims,
        )

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        data = kernels.dispersion_phase_shift_kernel(
            patch_tree.data,
            distances=self.distances,
            velocities=self.velocities,
            nf=self.nf,
            first_live_f=self.first_live_f,
            last_live_f=self.last_live_f,
            fs=self.fs,
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
