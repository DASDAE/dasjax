"""Fourier transform patch operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Self

import dascore as dc
import jax.numpy as jnp
import numpy as np
from dascore.proc.basic import pad as dc_pad
from dascore.transform.fourier import (
    _get_idft_dims_steps_axis,
    dft as dc_dft,
    idft as dc_idft,
)
from dascore.utils.misc import iterate
from dascore.utils.transformatter import FourierTransformatter
from scipy.fft import next_fast_len

from dasjax.core import PatchBoundary, PatchOperation, PatchPyTree

from .. import kernels
from .common import dummy_patch, replace, tree_boundary_from_patch


@dataclass(frozen=True)
class Dft(PatchOperation):
    """Transform patch data into the frequency domain."""

    method_name: ClassVar[str | None] = "dft"
    dim: str | None | tuple[str, ...]
    real: str | bool | None = None
    pad: bool = True
    axes: tuple[int, ...] = ()
    dxs: tuple[float, ...] = ()
    real_axis: int | None = None
    pad_width: tuple[tuple[int, int], ...] = ()
    out_boundary: PatchBoundary | None = None
    out_coords: tuple[Any, ...] | None = None
    out_dtype_codes: tuple[Any, ...] | None = None
    out_dims: tuple[str, ...] | None = None

    def bind(self, boundary: PatchBoundary) -> Self:
        patch = dummy_patch(boundary)
        dims = list(iterate(self.dim if self.dim is not None else boundary.dims))
        real = dims[-1] if self.real is True else self.real
        if isinstance(real, str) and real in dims:
            dims.append(dims.pop(dims.index(real)))
        pad_width = [(0, 0)] * len(boundary.dims)
        work_patch = patch
        if self.pad:
            for dim in dims:
                axis = work_patch.get_axis(dim)
                target = next_fast_len(len(work_patch.get_coord(dim)))
                pad_width[axis] = (0, target - len(work_patch.get_coord(dim)))
            work_patch = dc_pad.func(work_patch, **{dim: "fft" for dim in dims})
        axes = tuple(work_patch.get_axis(dim) for dim in dims)
        dxs = tuple(float(dc.to_float(work_patch.get_coord(dim).step)) for dim in dims)
        out = dc_dft.func(patch, dim=self.dim, real=self.real, pad=self.pad)
        out_tree, out_boundary = tree_boundary_from_patch(out)
        return replace(
            self,
            axes=axes,
            dxs=dxs,
            real_axis=work_patch.get_axis(real)
            if isinstance(real, str) and real in dims
            else None,
            pad_width=tuple(pad_width),
            out_boundary=out_boundary,
            out_coords=out_tree.coord_values,
            out_dtype_codes=out_tree.coord_dtype_codes,
            out_dims=out_tree.dims,
        )

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        data = patch_tree.data
        if any(before or after for before, after in self.pad_width):
            data = jnp.pad(data, self.pad_width)
        data = kernels.dft_kernel(
            data, axes=self.axes, dxs=self.dxs, real_axis=self.real_axis
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
class Idft(PatchOperation):
    """Transform patch data from the frequency domain."""

    method_name: ClassVar[str | None] = "idft"
    dim: str | None | tuple[str, ...] = None
    axes: tuple[int, ...] = ()
    steps: tuple[float, ...] = ()
    sizes: tuple[int, ...] | None = None
    real: bool = False
    out_boundary: PatchBoundary | None = None
    out_coords: tuple[Any, ...] | None = None
    out_dtype_codes: tuple[Any, ...] | None = None
    out_dims: tuple[str, ...] | None = None

    def bind(self, boundary: PatchBoundary) -> Self:
        patch = dummy_patch(boundary, dtype=np.complex128)
        dims, _steps, axes, real = _get_idft_dims_steps_axis(patch, self.dim)
        out = dc_idft.func(patch, dim=self.dim)
        out_tree, out_boundary = tree_boundary_from_patch(out)
        sizes = tuple(out.shape[axis] for axis in axes)
        new_dims = FourierTransformatter().rename_dims(dims, forward=False)
        out_steps = tuple(
            float(dc.to_float(out.get_coord(dim).step)) for dim in new_dims
        )
        return replace(
            self,
            axes=tuple(axes),
            steps=out_steps,
            sizes=sizes,
            real=real,
            out_boundary=out_boundary,
            out_coords=out_tree.coord_values,
            out_dtype_codes=out_tree.coord_dtype_codes,
            out_dims=out_tree.dims,
        )

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        data = kernels.idft_kernel(
            patch_tree.data,
            axes=self.axes,
            new_steps=self.steps,
            sizes=self.sizes,
            real=self.real,
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
