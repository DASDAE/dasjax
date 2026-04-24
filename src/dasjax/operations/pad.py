"""Pad patch operation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

import jax.numpy as jnp
from dascore.proc.basic import pad as dc_pad
from dascore.utils.patch import get_dim_axis_value
from scipy.fft import next_fast_len

from dasjax.core import PatchBoundary, PatchOperation, PatchPyTree

from .common import dummy_patch, replace, tree_boundary_from_patch


@dataclass(frozen=True)
class Pad(PatchOperation):
    """Pad patch data and optionally expand coordinates."""

    mode: str = "constant"
    constant_values: Any = 0
    expand_coords: bool = True
    samples: bool = False
    kwargs: dict[str, Any] | None = None
    data_pad_width: tuple[tuple[int, int], ...] = ()
    out_boundary: PatchBoundary | None = None
    out_coords: tuple[Any, ...] | None = None
    out_dtype_codes: tuple[Any, ...] | None = None
    out_dims: tuple[str, ...] | None = None

    def __init__(
        self,
        mode="constant",
        constant_values=0,
        expand_coords=True,
        samples=False,
        **kwargs,
    ):
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "constant_values", constant_values)
        object.__setattr__(self, "expand_coords", expand_coords)
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "kwargs", dict(kwargs))
        object.__setattr__(self, "data_pad_width", ())
        object.__setattr__(self, "out_boundary", None)
        object.__setattr__(self, "out_coords", None)
        object.__setattr__(self, "out_dtype_codes", None)
        object.__setattr__(self, "out_dims", None)

    def bind(self, boundary: PatchBoundary) -> Self:
        patch = dummy_patch(boundary)
        pad_width = [(0, 0)] * len(patch.shape)
        dimfo = get_dim_axis_value(patch, kwargs=self.kwargs or {}, allow_multiple=True)
        for dim, axis, value in dimfo:
            coord = patch.get_coord(dim, require_evenly_sampled=False)
            if value in {"fft", "correlate"}:
                target_length = len(coord) if value == "fft" else 2 * len(coord) - 1
                pad_tuple = (0, next_fast_len(target_length) - len(coord))
            else:
                if not isinstance(value, (tuple, list)):
                    value = (value, value)
                pad_tuple = (
                    tuple(int(coord.get_sample_count(x)) for x in value)
                    if not self.samples
                    else tuple(int(x) for x in value)
                )
            pad_width[axis] = tuple(pad_tuple)
        out = dc_pad.func(
            patch,
            mode=self.mode,
            constant_values=self.constant_values,
            expand_coords=self.expand_coords,
            samples=self.samples,
            **(self.kwargs or {}),
        )
        out_tree, out_boundary = tree_boundary_from_patch(out)
        return replace(
            self,
            data_pad_width=tuple(pad_width),
            out_boundary=out_boundary,
            out_coords=out_tree.coord_values,
            out_dtype_codes=out_tree.coord_dtype_codes,
            out_dims=out_tree.dims,
        )

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        return patch_tree.new(
            data=jnp.pad(
                patch_tree.data,
                self.data_pad_width,
                mode=self.mode,
                constant_values=self.constant_values,
            ),
            coords=self.out_coords,
            coord_dtype_codes=self.out_dtype_codes,
            dims=self.out_dims,
        )

    def update_boundary(self, boundary: PatchBoundary) -> PatchBoundary:
        _ = boundary
        assert self.out_boundary is not None
        return self.out_boundary
