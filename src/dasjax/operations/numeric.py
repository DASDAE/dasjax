"""Numeric DASCore patch operations backed by host callbacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Self

import jax
import numpy as np

from dasjax.core import PatchBoundary, PatchOperation, PatchPyTree

from .common import dummy_patch, replace, tree_boundary_from_patch


def _init_callback(obj: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    object.__setattr__(obj, "args", args)
    object.__setattr__(obj, "kwargs", kwargs)
    object.__setattr__(obj, "boundary", None)
    object.__setattr__(obj, "out_shape", None)
    object.__setattr__(obj, "out_dtype", None)
    object.__setattr__(obj, "out_boundary", None)
    object.__setattr__(obj, "out_coords", None)
    object.__setattr__(obj, "out_dtype_codes", None)
    object.__setattr__(obj, "out_dims", None)


@dataclass(frozen=True)
class _DascoreCallbackOperation(PatchOperation):
    """Run DASCore numeric transforms whose full kernels are not yet ported."""

    register: ClassVar[bool] = False
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] | None = None
    boundary: PatchBoundary | None = None
    out_shape: tuple[int, ...] | None = None
    out_dtype: Any = None
    out_boundary: PatchBoundary | None = None
    out_coords: tuple[Any, ...] | None = None
    out_dtype_codes: tuple[Any, ...] | None = None
    out_dims: tuple[str, ...] | None = None

    def _call(self, patch):
        return getattr(patch, type(self).operation_name())(
            *self.args,
            **(self.kwargs or {}),
        )

    def bind(self, boundary: PatchBoundary) -> Self:
        out = self._call(dummy_patch(boundary))
        out_tree, out_boundary = tree_boundary_from_patch(out)
        return replace(
            self,
            boundary=boundary,
            out_shape=out.shape,
            out_dtype=np.asarray(out.data).dtype,
            out_boundary=out_boundary,
            out_coords=out_tree.coord_values,
            out_dtype_codes=out_tree.coord_dtype_codes,
            out_dims=out_tree.dims,
        )

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        assert self.boundary is not None
        assert self.out_shape is not None and self.out_dtype is not None

        def _callback(data, *coord_leaves):
            coord_count = len(coord_leaves) // 2
            coords = tuple(coord_leaves[:coord_count])
            dtype_codes = tuple(coord_leaves[coord_count:])
            tree = PatchPyTree(
                data=data,
                coord_values=coords,
                coord_dtype_codes=dtype_codes,
                dims=self.boundary.dims,
            )
            patch = self.boundary.to_patch(tree)
            return np.asarray(self._call(patch).data)

        result = jax.pure_callback(
            _callback,
            jax.ShapeDtypeStruct(self.out_shape, self.out_dtype),
            patch_tree.data,
            *patch_tree.coord_values,
            *patch_tree.coord_dtype_codes,
            vmap_method="sequential",
        )
        return patch_tree.new(
            data=result,
            coords=self.out_coords,
            coord_dtype_codes=self.out_dtype_codes,
            dims=self.out_dims,
        )

    def update_boundary(self, boundary: PatchBoundary) -> PatchBoundary:
        _ = boundary
        assert self.out_boundary is not None
        return self.out_boundary


@dataclass(frozen=True)
class Correlate(_DascoreCallbackOperation):
    """Correlate source rows or columns with all other rows or columns."""

    def __init__(self, samples: bool = False, lag=None, **kwargs):
        _init_callback(self, (), {"samples": samples, "lag": lag, **kwargs})


@dataclass(frozen=True)
class Decimate(_DascoreCallbackOperation):
    """Decimate a patch along a dimension."""

    def __init__(self, filter_type: str | None = "iir", copy: bool = True, **kwargs):
        _init_callback(self, (), {"filter_type": filter_type, "copy": copy, **kwargs})


@dataclass(frozen=True)
class Interpolate(_DascoreCallbackOperation):
    """Interpolate along one dimension."""

    def __init__(self, kind: str | int = "linear", **kwargs):
        _init_callback(self, (), {"kind": kind, **kwargs})


@dataclass(frozen=True)
class Iresample(_DascoreCallbackOperation):
    """Deprecated DASCore interpolation-resample operation."""

    def __init__(self, *args, **kwargs):
        _init_callback(self, tuple(args), dict(kwargs))


@dataclass(frozen=True)
class Istft(_DascoreCallbackOperation):
    """Invert a short-time Fourier transform."""

    method_name: ClassVar[str | None] = "istft"


@dataclass(frozen=True)
class Resample(_DascoreCallbackOperation):
    """Resample along one dimension."""

    def __init__(
        self,
        window=None,
        interp_kind: str = "linear",
        samples: bool = False,
        **kwargs,
    ):
        _init_callback(
            self,
            (),
            {
                "window": window,
                "interp_kind": interp_kind,
                "samples": samples,
                **kwargs,
            },
        )


@dataclass(frozen=True)
class Rfft(_DascoreCallbackOperation):
    """Perform DASCore's real FFT transform."""

    method_name: ClassVar[str | None] = "rfft"

    def __init__(self, dim: str = "time"):
        _init_callback(self, (), {"dim": dim})


@dataclass(frozen=True)
class Spectrogram(_DascoreCallbackOperation):
    """Calculate a spectrogram from patch data."""

    def __init__(self, dim: str = "time", **kwargs):
        _init_callback(self, (), {"dim": dim, **kwargs})


@dataclass(frozen=True)
class Stft(_DascoreCallbackOperation):
    """Perform a short-time Fourier transform."""

    method_name: ClassVar[str | None] = "stft"

    def __init__(
        self,
        taper_window: str | np.ndarray | tuple[str, Any, ...] = "hann",
        overlap=None,
        samples: bool = False,
        detrend: bool = False,
        **kwargs,
    ):
        _init_callback(
            self,
            (),
            {
                "taper_window": taper_window,
                "overlap": overlap,
                "samples": samples,
                "detrend": detrend,
                **kwargs,
            },
        )
