"""Strain transform operations."""

from __future__ import annotations

from dataclasses import dataclass
from operator import truediv
from typing import Any, Self

import dascore as dc
import numpy as np
from dascore.exceptions import ParameterError
from dascore.units import convert_units, get_factor_and_unit, get_unit

from dasjax.core import PatchBoundary, PatchOperation, PatchPyTree

from .. import kernels
from .common import (
    dummy_patch,
    get_data_units_from_dims,
    replace,
    tree_boundary_from_patch,
)


@dataclass(frozen=True)
class VelocityToStrainRate(PatchOperation):
    """Convert velocity DAS data to strain rate using central differences."""

    step_multiple: int = 2
    gauge_multiple: int | None = None
    order: int = 2
    axis: int | None = None
    spacing: float = 1.0
    attrs: Any = None

    def bind(self, boundary: PatchBoundary) -> Self:
        step_multiple = (
            self.gauge_multiple * 2
            if self.gauge_multiple is not None
            else self.step_multiple
        )
        if step_multiple <= 0:
            raise ParameterError("step_multiple must be positive.")
        if step_multiple % 2 != 0:
            msg = (
                "Step_multiple must be even. Use velocity_to_strain_rate_edgeless "
                "if odd step multiples are required."
            )
            raise ParameterError(msg)
        coord = dummy_patch(boundary).get_coord("distance", require_evenly_sampled=True)
        attrs = boundary.attrs.update(
            data_type="strain_rate",
            gauge_length=coord.step * step_multiple,
            data_units=get_data_units_from_dims(boundary, ("distance",), truediv),
        )
        return replace(
            self,
            step_multiple=step_multiple,
            axis=boundary.axis("distance"),
            spacing=float(dc.to_float(coord.step)),
            attrs=attrs,
        )

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        assert self.axis is not None
        data = kernels.differentiate_kernel(
            patch_tree.data,
            axis=self.axis,
            dx_or_spacing=self.spacing,
            order=self.order,
            step=self.step_multiple // 2,
        )
        return patch_tree.new(data=data)

    def update_boundary(self, boundary: PatchBoundary) -> PatchBoundary:
        return boundary.new(attrs=self.attrs)


@dataclass(frozen=True)
class VelocityToStrainRateEdgeless(PatchOperation):
    """Estimate strain rate with central differences and dropped edges."""

    step_multiple: int = 1
    axis: int | None = None
    gauge_length: float = 1.0
    out_boundary: PatchBoundary | None = None
    out_coords: tuple[Any, ...] | None = None
    out_dtype_codes: tuple[Any, ...] | None = None
    out_dims: tuple[str, ...] | None = None

    def bind(self, boundary: PatchBoundary) -> Self:
        if self.step_multiple <= 0:
            raise ParameterError("step_multiple must be positive.")
        out_patch = dummy_patch(boundary).velocity_to_strain_rate_edgeless(
            step_multiple=self.step_multiple
        )
        out_tree, out_boundary = tree_boundary_from_patch(out_patch)
        gauge_length = (
            float(dc.to_float(boundary.coord("distance").step)) * self.step_multiple
        )
        return replace(
            self,
            axis=boundary.axis("distance"),
            gauge_length=gauge_length,
            out_boundary=out_boundary,
            out_coords=out_tree.coord_values,
            out_dtype_codes=out_tree.coord_dtype_codes,
            out_dims=out_tree.dims,
        )

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        assert self.axis is not None
        data = kernels.velocity_to_strain_rate_edgeless_kernel(
            patch_tree.data,
            axis=self.axis,
            step_multiple=self.step_multiple,
            gauge_length=self.gauge_length,
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


@dataclass(frozen=True)
class RadiansToStrain(PatchOperation):
    """Convert radians data units to strain."""

    gauge_length: Any = None
    wave_length: float = 1550.0 * 10 ** (-9)
    stress_constant: float = 0.79
    refractive_index: float = 1.445
    factor: float = 1.0
    attrs: Any = None

    def bind(self, boundary: PatchBoundary) -> Self:
        gauge_source = (
            self.gauge_length
            if self.gauge_length is not None
            else getattr(boundary.attrs, "gauge_length", None)
        )
        gauge = convert_units(gauge_source, "m")
        if gauge is None or gauge <= 0:
            msg = (
                "Gauge length must be non-zero positive and provided "
                "or defined in patch attrs."
            )
            raise ParameterError(msg)
        quant = dc.get_quantity(boundary.attrs.data_units)
        if str(dc.get_unit("radians")) not in str(quant):
            return replace(self, factor=1.0, attrs=boundary.attrs)
        const = self.wave_length / (
            4 * np.pi * self.refractive_index * gauge * self.stress_constant
        )
        data_units = boundary.attrs.get("data_units", None)
        data_factor, data_unit = get_factor_and_unit(data_units, simplify=True)
        new_units = get_unit(data_unit) * get_unit("strain/radians")
        attrs = boundary.attrs.update(data_units=new_units)
        return replace(self, factor=float(const * data_factor), attrs=attrs)

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        return patch_tree.new(
            data=kernels.radians_to_strain_kernel(patch_tree.data, self.factor)
        )

    def update_boundary(self, boundary: PatchBoundary) -> PatchBoundary:
        return boundary.new(attrs=self.attrs)
