"""Normalize patch operation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from dasjax.core import PatchBoundary, PatchOperation, PatchPyTree

from .. import kernels
from .common import replace


@dataclass(frozen=True)
class Normalize(PatchOperation):
    """Normalize patch data along one dimension."""

    dim: str
    norm: str = "l2"
    axis: int | None = None

    def bind(self, boundary: PatchBoundary) -> Self:
        return replace(self, axis=boundary.axis(self.dim))

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        assert self.axis is not None
        return patch_tree.new(
            data=kernels.normalize_kernel(
                patch_tree.data, axis=self.axis, norm=self.norm
            )
        )
