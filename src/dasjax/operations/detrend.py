"""Detrend patch operation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from dasjax.core import PatchBoundary, PatchOperation, PatchPyTree

from .. import kernels
from .common import replace


@dataclass(frozen=True)
class Detrend(PatchOperation):
    """Remove a linear or constant trend along one dimension."""

    dim: str
    type: str = "linear"
    axis: int | None = None

    def bind(self, boundary: PatchBoundary) -> Self:
        kernels.validate_detrend_type(self.type)
        return replace(self, axis=boundary.axis(self.dim))

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        assert self.axis is not None
        return patch_tree.new(
            data=kernels.detrend_kernel(patch_tree.data, axis=self.axis, type=self.type)
        )
