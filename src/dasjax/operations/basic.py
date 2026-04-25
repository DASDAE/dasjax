"""Basic DASCore-style patch operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Self

import jax.numpy as jnp
from dascore.utils.misc import iterate

from dasjax.core import PatchBoundary, PatchOperation, PatchPyTree

from .. import kernels
from .common import dummy_patch, replace, tree_boundary_from_patch


@dataclass(frozen=True)
class Identity(PatchOperation):
    """Return a patch unchanged."""

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        return patch_tree.new(data=kernels.identity_kernel(patch_tree.data))


@dataclass(frozen=True)
class Scale(PatchOperation):
    """Scale patch data by a constant factor."""

    factor: float

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        return patch_tree.new(data=kernels.scale_kernel(patch_tree.data, self.factor))


@dataclass(frozen=True)
class Add(PatchOperation):
    """Add a constant value to patch data."""

    value: float

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        return patch_tree.new(data=kernels.add_kernel(patch_tree.data, self.value))


@dataclass(frozen=True)
class Subtract(PatchOperation):
    """Subtract a value from patch data."""

    other: Any

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        return patch_tree.new(data=jnp.asarray(patch_tree.data) - self.other)


@dataclass(frozen=True)
class Multiply(PatchOperation):
    """Multiply patch data by a value."""

    other: Any

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        return patch_tree.new(data=jnp.asarray(patch_tree.data) * self.other)


@dataclass(frozen=True)
class Divide(PatchOperation):
    """Divide patch data by a value."""

    other: Any

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        return patch_tree.new(data=jnp.asarray(patch_tree.data) / self.other)


@dataclass(frozen=True)
class Maximum(PatchOperation):
    """Return elementwise maximum of patch data and a value."""

    other: Any

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        return patch_tree.new(
            data=jnp.maximum(jnp.asarray(patch_tree.data), self.other)
        )


@dataclass(frozen=True)
class Minimum(PatchOperation):
    """Return elementwise minimum of patch data and a value."""

    other: Any

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        return patch_tree.new(
            data=jnp.minimum(jnp.asarray(patch_tree.data), self.other)
        )


@dataclass(frozen=True)
class Abs(PatchOperation):
    """Take absolute value of patch data."""

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        return patch_tree.new(data=kernels.abs_kernel(patch_tree.data))


@dataclass(frozen=True)
class Clip(PatchOperation):
    """Clip patch data to a fixed range."""

    min_value: float
    max_value: float

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        return patch_tree.new(
            data=kernels.clip_kernel(patch_tree.data, self.min_value, self.max_value)
        )


@dataclass(frozen=True)
class Real(PatchOperation):
    """Return the real component of patch data."""

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        return patch_tree.new(data=kernels.real_kernel(patch_tree.data))


@dataclass(frozen=True)
class Imag(PatchOperation):
    """Return the imaginary component of patch data."""

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        return patch_tree.new(data=kernels.imag_kernel(patch_tree.data))


@dataclass(frozen=True)
class Angle(PatchOperation):
    """Return the phase angle of patch data."""

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        return patch_tree.new(data=kernels.angle_kernel(patch_tree.data))


@dataclass(frozen=True)
class Conj(PatchOperation):
    """Return the complex conjugate of patch data."""

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        return patch_tree.new(data=kernels.conj_kernel(patch_tree.data))


@dataclass(frozen=True)
class Exp(PatchOperation):
    """Apply exponential to patch data."""

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        return patch_tree.new(data=jnp.exp(jnp.asarray(patch_tree.data)))


@dataclass(frozen=True)
class Log(PatchOperation):
    """Apply natural logarithm to patch data."""

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        return patch_tree.new(data=jnp.log(jnp.asarray(patch_tree.data)))


@dataclass(frozen=True)
class Log10(PatchOperation):
    """Apply base-10 logarithm to patch data."""

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        return patch_tree.new(data=jnp.log10(jnp.asarray(patch_tree.data)))


@dataclass(frozen=True)
class Log2(PatchOperation):
    """Apply base-2 logarithm to patch data."""

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        return patch_tree.new(data=jnp.log2(jnp.asarray(patch_tree.data)))


@dataclass(frozen=True)
class IsFinite(PatchOperation):
    """Return a finite-value mask for patch data."""

    method_name: ClassVar[str | None] = "is_finite"

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        return patch_tree.new(data=jnp.isfinite(jnp.asarray(patch_tree.data)))


@dataclass(frozen=True)
class IsInf(PatchOperation):
    """Return an infinite-value mask for patch data."""

    method_name: ClassVar[str | None] = "isinf"

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        return patch_tree.new(data=jnp.isinf(jnp.asarray(patch_tree.data)))


@dataclass(frozen=True)
class IsNan(PatchOperation):
    """Return a NaN-value mask for patch data."""

    method_name: ClassVar[str | None] = "isnan"

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        return patch_tree.new(data=jnp.isnan(jnp.asarray(patch_tree.data)))


@dataclass(frozen=True)
class FillNa(PatchOperation):
    """Fill NaN, and optionally infinite, values in patch data."""

    method_name: ClassVar[str | None] = "fillna"
    value: Any
    include_inf: bool = True

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        data = jnp.asarray(patch_tree.data)
        mask = jnp.isnan(data)
        if self.include_inf:
            mask = mask | jnp.isinf(data)
        return patch_tree.new(data=jnp.where(mask, self.value, data))


@dataclass(frozen=True)
class Where(PatchOperation):
    """Select patch data where a condition is true, otherwise use another value."""

    cond: Any
    other: Any = jnp.nan

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        cond = self.cond.data if hasattr(self.cond, "coords") else self.cond
        other = self.other.data if hasattr(self.other, "coords") else self.other
        return patch_tree.new(data=jnp.where(cond, patch_tree.data, other))


@dataclass(frozen=True)
class Flip(PatchOperation):
    """Flip data and, optionally, associated coordinate values."""

    dims: tuple[str, ...] = ()
    flip_coords: bool = True
    axes: tuple[int, ...] = ()
    coord_axes: tuple[tuple[int, tuple[int, ...]], ...] = ()

    def __init__(self, *dims: str, flip_coords: bool = True):
        object.__setattr__(self, "dims", tuple(dims))
        object.__setattr__(self, "flip_coords", flip_coords)
        object.__setattr__(self, "axes", ())
        object.__setattr__(self, "coord_axes", ())

    def bind(self, boundary: PatchBoundary) -> Self:
        dims = self.dims or boundary.dims
        axes = tuple(boundary.axis(dim) for dim in dims)
        coord_axes = []
        if self.flip_coords:
            for name in boundary.coord_names:
                axes_for_coord = tuple(
                    idx
                    for idx, coord_dim in enumerate(boundary.coord_dims(name))
                    if coord_dim in dims
                )
                if axes_for_coord:
                    coord_axes.append((boundary.coord_index(name), axes_for_coord))
        return replace(self, dims=dims, axes=axes, coord_axes=tuple(coord_axes))

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        data = kernels.flip_kernel(patch_tree.data, self.axes)
        coords = patch_tree.coord_values
        if self.flip_coords:
            for index, axes in self.coord_axes:
                coords = tuple(
                    jnp.flip(value, axis=axes) if idx == index else value
                    for idx, value in enumerate(coords)
                )
        return patch_tree.new(data=data, coords=coords)


@dataclass(frozen=True)
class Roll(PatchOperation):
    """Roll data along one dimension."""

    samples: bool = False
    update_coord: bool = False
    kwargs: dict[str, Any] | None = None
    axis: int | None = None
    shift: int | None = None

    def __init__(
        self,
        samples: bool = False,
        update_coord: bool = False,
        **kwargs,
    ):
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "update_coord", update_coord)
        object.__setattr__(self, "kwargs", dict(kwargs))
        object.__setattr__(self, "axis", None)
        object.__setattr__(self, "shift", None)

    def bind(self, boundary: PatchBoundary) -> Self:
        if self.update_coord:
            raise NotImplementedError(
                "Compiled roll currently requires update_coord=False."
            )
        kwargs = self.kwargs or {}
        dim = next(key for key in kwargs if key in boundary.dims)
        coord = boundary.coord(dim)
        return replace(
            self,
            axis=boundary.axis(dim),
            shift=int(coord.get_sample_count(kwargs[dim], samples=self.samples)),
        )

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        assert self.axis is not None and self.shift is not None
        return patch_tree.new(
            data=kernels.roll_kernel(patch_tree.data, self.shift, self.axis)
        )


@dataclass(frozen=True)
class Standardize(PatchOperation):
    """Standardize patch data along one dimension."""

    dim: str
    axis: int | None = None

    def bind(self, boundary: PatchBoundary) -> Self:
        return replace(self, axis=boundary.axis(self.dim))

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        assert self.axis is not None
        return patch_tree.new(
            data=kernels.standardize_kernel(patch_tree.data, self.axis)
        )


_REDUCTIONS = {
    "all": jnp.all,
    "any": jnp.any,
    "max": jnp.max,
    "mean": jnp.mean,
    "median": jnp.median,
    "min": jnp.min,
    "std": jnp.std,
    "sum": jnp.sum,
}


@dataclass(frozen=True)
class Aggregate(PatchOperation):
    """Aggregate values along one or more dimensions."""

    dim: str | tuple[str, ...] | None = None
    method: str | Any = "mean"
    dim_reduce: str | Any = "empty"
    axes: tuple[int, ...] = ()
    keepdims: bool = True
    out_boundary: PatchBoundary | None = None
    out_coords: tuple[Any, ...] | None = None
    out_dtype_codes: tuple[Any, ...] | None = None
    out_dims: tuple[str, ...] | None = None

    def bind(self, boundary: PatchBoundary) -> Self:
        if not isinstance(self.method, str) or self.method not in _REDUCTIONS:
            msg = f"Unsupported aggregate method {self.method!r}."
            raise NotImplementedError(msg)
        if self.dim_reduce not in {"empty", "squeeze"}:
            msg = "Compiled aggregate supports dim_reduce='empty' or 'squeeze'."
            raise NotImplementedError(msg)
        dims = tuple(iterate(self.dim if self.dim is not None else boundary.dims))
        axes = tuple(boundary.axis(dim) for dim in dims)
        patch = dummy_patch(boundary)
        if self.method in {"all", "any"}:
            out = getattr(patch, self.method)(dim=dims, dim_reduce=self.dim_reduce)
        else:
            out = patch.aggregate(
                dim=dims,
                method=self.method,
                dim_reduce=self.dim_reduce,
            )
        out_tree, out_boundary = tree_boundary_from_patch(out)
        return replace(
            self,
            axes=axes,
            keepdims=self.dim_reduce == "empty",
            out_boundary=out_boundary,
            out_coords=out_tree.coord_values,
            out_dtype_codes=out_tree.coord_dtype_codes,
            out_dims=out_tree.dims,
        )

    def kernel(self, patch_tree: PatchPyTree) -> PatchPyTree:
        data = _REDUCTIONS[self.method](
            jnp.asarray(patch_tree.data),
            axis=self.axes,
            keepdims=self.keepdims,
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
class _Reduction(Aggregate):
    """Base class for named reduction methods."""

    register: ClassVar[bool] = False
    method_name: ClassVar[str | None] = None

    def __init__(
        self,
        dim: str | tuple[str, ...] | None = None,
        dim_reduce: str | Any = "empty",
    ):
        object.__setattr__(self, "dim", dim)
        object.__setattr__(self, "method", type(self).operation_name())
        object.__setattr__(self, "dim_reduce", dim_reduce)
        object.__setattr__(self, "axes", ())
        object.__setattr__(self, "keepdims", True)
        object.__setattr__(self, "out_boundary", None)
        object.__setattr__(self, "out_coords", None)
        object.__setattr__(self, "out_dtype_codes", None)
        object.__setattr__(self, "out_dims", None)


@dataclass(frozen=True)
class All(_Reduction):
    """Perform boolean all reduction."""

    def __init__(self, dim=None, dim_reduce="empty"):
        _Reduction.__init__(self, dim=dim, dim_reduce=dim_reduce)


@dataclass(frozen=True)
class AnyOp(_Reduction):
    """Perform boolean any reduction."""

    method_name: ClassVar[str | None] = "any"

    def __init__(self, dim=None, dim_reduce="empty"):
        _Reduction.__init__(self, dim=dim, dim_reduce=dim_reduce)


@dataclass(frozen=True)
class Max(_Reduction):
    """Calculate maximum along one or more dimensions."""

    def __init__(self, dim=None, dim_reduce="empty"):
        _Reduction.__init__(self, dim=dim, dim_reduce=dim_reduce)


@dataclass(frozen=True)
class Mean(_Reduction):
    """Calculate mean along one or more dimensions."""

    def __init__(self, dim=None, dim_reduce="empty"):
        _Reduction.__init__(self, dim=dim, dim_reduce=dim_reduce)


@dataclass(frozen=True)
class Median(_Reduction):
    """Calculate median along one or more dimensions."""

    def __init__(self, dim=None, dim_reduce="empty"):
        _Reduction.__init__(self, dim=dim, dim_reduce=dim_reduce)


@dataclass(frozen=True)
class Min(_Reduction):
    """Calculate minimum along one or more dimensions."""

    def __init__(self, dim=None, dim_reduce="empty"):
        _Reduction.__init__(self, dim=dim, dim_reduce=dim_reduce)


@dataclass(frozen=True)
class Std(_Reduction):
    """Calculate standard deviation along one or more dimensions."""

    def __init__(self, dim=None, dim_reduce="empty"):
        _Reduction.__init__(self, dim=dim, dim_reduce=dim_reduce)


@dataclass(frozen=True)
class Sum(_Reduction):
    """Calculate sum along one or more dimensions."""

    def __init__(self, dim=None, dim_reduce="empty"):
        _Reduction.__init__(self, dim=dim, dim_reduce=dim_reduce)
