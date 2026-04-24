"""Pipeline support for chaining JAX patch operations."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, ClassVar

import jax
import numpy as np
from jax import tree_util

from .core import PatchPyTree, get_patch_operation

_COMPILED_CACHE_MAXSIZE = 128
_SEGMENT_CACHE_MAXSIZE = 128


@dataclass(frozen=True)
class PipelineStep:
    """A recorded operation with its bound arguments."""

    name: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


@dataclass(frozen=True)
class PipelinePlanSegment:
    """One planned execution segment for a patch pipeline."""

    kind: str
    operations: tuple[str, ...]
    fused: bool
    jitted: bool
    input_dims: tuple[str, ...] = ()
    output_dims: tuple[str, ...] = ()
    reason: str | None = None


@dataclass(frozen=True)
class PipelinePlan:
    """Diagnostic execution plan for a patch pipeline."""

    segments: tuple[PipelinePlanSegment, ...]
    uses_core: bool

    @property
    def fused_kernel_count(self) -> int:
        """Return the number of JIT-fused execution segments."""
        return sum(segment.jitted for segment in self.segments)


class _BoundedCache:
    """A small insertion-ordered cache with LRU eviction."""

    def __init__(self, maxsize: int) -> None:
        self.maxsize = maxsize
        self._values: OrderedDict[Any, Any] = OrderedDict()

    def get(self, key: Any, default: Any = None) -> Any:
        value = self._values.get(key, default)
        if key in self._values:
            self._values.move_to_end(key)
        return value

    def add(self, key: Any, value: Any) -> None:
        self._values[key] = value
        self._values.move_to_end(key)
        while len(self._values) > self.maxsize:
            self._values.popitem(last=False)

    def __len__(self) -> int:
        return len(self._values)


def _freeze_device(device: jax.Device | None) -> Any:
    if device is None:
        return None
    return (
        "device",
        device.platform,
        getattr(device, "id", None),
        getattr(device, "process_index", None),
    )


def _resolve_target_device(
    *,
    device: jax.Device | None = None,
    backend: str | None = None,
) -> jax.Device | None:
    if device is not None and backend is not None:
        msg = "Pass either device or backend to compile(), not both."
        raise ValueError(msg)
    if device is not None:
        return device
    if backend is None:
        return None
    devices = jax.local_devices(backend=backend)
    if not devices:
        msg = f"No local JAX devices are available for backend {backend!r}."
        raise ValueError(msg)
    return devices[0]


def _freeze_value(value: Any) -> Any:
    """Convert nested values into hashable cache-key components."""
    if isinstance(value, np.ndarray):
        return (
            "ndarray",
            str(value.dtype),
            value.shape,
            tuple(np.asarray(value).reshape(-1).tolist()),
        )
    if isinstance(value, dict):
        return tuple((key, _freeze_value(val)) for key, val in sorted(value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze_value(item) for item in value))
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _operation_key(operation: Any) -> tuple[Any, ...]:
    """Return a stable cache key for a bound core operation."""
    if is_dataclass(operation):
        entries = tuple(
            (field.name, _freeze_value(getattr(operation, field.name)))
            for field in fields(operation)
        )
    else:
        entries = tuple((key, _freeze_value(value)) for key, value in vars(operation).items())
    return type(operation).operation_name(), entries


class JaxPatchPipeline:
    """A reusable recipe composed of registered dasjax operations."""

    _compiled_cache: ClassVar[_BoundedCache] = _BoundedCache(
        maxsize=_COMPILED_CACHE_MAXSIZE
    )

    def __init__(self, steps: tuple[PipelineStep, ...] = ()) -> None:
        self._steps = steps

    @property
    def steps(self) -> tuple[PipelineStep, ...]:
        """Expose the recorded step list."""
        return self._steps

    def __getattr__(self, name: str):
        """Record only known JAX patch operations."""
        get_patch_operation(name)

        def _record(*args: Any, **kwargs: Any) -> "JaxPatchPipeline":
            step = PipelineStep(name=name, args=args, kwargs=dict(kwargs))
            return self.__class__(steps=(*self._steps, step))

        return _record

    def _compile_cache_key(
        self,
        *,
        device: jax.Device | None = None,
        backend: str | None = None,
        assert_no_fallback: bool = False,
    ) -> tuple[Any, ...]:
        """Return a stable cache key for equivalent pipeline definitions."""
        return (
            tuple(
                (
                    step.name,
                    _freeze_value(step.args),
                    _freeze_value(step.kwargs),
                )
                for step in self._steps
            ),
            _freeze_device(device),
            backend,
            assert_no_fallback,
        )

    @staticmethod
    def _validation_signature(patch) -> tuple[Any, ...]:
        data = np.asarray(patch.data)
        coord_sig = tuple(
            (
                name,
                tuple(patch.coords.dim_map[name]),
                np.asarray(coord.values).dtype.str,
                len(coord),
                getattr(coord, "step", None),
            )
            for name, coord in patch.coords.coord_map.items()
        )
        return (
            tuple(patch.dims),
            tuple(patch.shape),
            data.dtype.str,
            bool(np.isfinite(data).all()),
            coord_sig,
        )

    def _resolve_operations(self) -> tuple[Any, ...]:
        """Return class-authored operation instances."""
        return tuple(
            get_patch_operation(step.name)(*step.args, **step.kwargs)
            for step in self._steps
        )

    @staticmethod
    def _build_core_runner(operations: tuple[Any, ...]):
        """Build a class-authored PatchPyTree operation segment."""

        def _run(patch_tree):
            current = patch_tree
            for operation in operations:
                current = operation.kernel(current)
            return current

        return jax.jit(_run)

    @staticmethod
    def _plan_core_segments(
        operations: tuple[Any, ...],
        boundary: Any,
    ) -> tuple[tuple[tuple[Any, ...], Any, bool], ...]:
        """Bind operations and statically advance boundaries for JAX segments."""
        segments: list[tuple[tuple[Any, ...], Any, bool]] = []
        start_index = 0
        while start_index < len(operations):
            bound_ops, boundary, needs_result_boundary, start_index = (
                JaxPatchPipeline._plan_core_segment(
                    operations,
                    boundary,
                    start_index,
                )
            )
            segments.append((bound_ops, boundary, needs_result_boundary))
            if needs_result_boundary:
                break
        return tuple(segments)

    @staticmethod
    def _plan_core_segment(
        operations: tuple[Any, ...],
        boundary: Any,
        start_index: int,
    ) -> tuple[tuple[Any, ...], Any, bool, int]:
        """Bind one executable segment, stopping at result-dependent metadata."""
        segment_ops: list[Any] = []
        current_boundary = boundary
        op_index = start_index
        while op_index < len(operations):
            operation = operations[op_index]
            bound_operation = operation.bind(current_boundary)
            segment_ops.append(bound_operation)
            if type(bound_operation).overrides("update_boundary"):
                current_boundary = bound_operation.update_boundary(current_boundary)
            needs_result_boundary = type(bound_operation).overrides(
                "update_boundary_from_result"
            )
            stops_segment = needs_result_boundary
            op_index += 1
            if stops_segment:
                return (
                    tuple(segment_ops),
                    current_boundary,
                    needs_result_boundary,
                    op_index,
                )
        return tuple(segment_ops), current_boundary, False, op_index

    def assert_no_fallback(self, patch) -> None:
        """No-op compatibility method; core pipelines have no legacy fallback."""
        _ = patch

    def plan(self, patch) -> PipelinePlan:
        """Return a diagnostic execution plan for this pipeline and patch."""
        return self._plan_core_execution(patch, self._resolve_operations())

    def _plan_core_execution(
        self,
        patch,
        core_operations: tuple[Any, ...],
    ) -> PipelinePlan:
        """Plan class-authored core execution without running kernels."""
        _, boundary = PatchPyTree.from_patch(patch)
        segments: list[PipelinePlanSegment] = []
        op_index = 0
        while op_index < len(core_operations):
            input_dims = boundary.dims
            bound_ops, planned_boundary, needs_result_boundary, op_index = (
                self._plan_core_segment(core_operations, boundary, op_index)
            )
            reason = (
                "requires result-dependent boundary update"
                if needs_result_boundary
                else None
            )
            segments.append(
                PipelinePlanSegment(
                    kind="core",
                    operations=tuple(type(op).operation_name() for op in bound_ops),
                    fused=len(bound_ops) > 1,
                    jitted=True,
                    input_dims=input_dims,
                    output_dims=planned_boundary.dims,
                    reason=reason,
                )
            )
            boundary = planned_boundary
            if needs_result_boundary:
                if op_index < len(core_operations):
                    remaining = core_operations[op_index:]
                    segments.append(
                        PipelinePlanSegment(
                            kind="unplanned",
                            operations=tuple(
                                type(op).operation_name() for op in remaining
                            ),
                            fused=False,
                            jitted=False,
                            input_dims=boundary.dims,
                            output_dims=(),
                            reason="requires executing previous segment first",
                        )
                    )
                break
        return PipelinePlan(segments=tuple(segments), uses_core=True)

    def compile(
        self,
        *,
        device: jax.Device | None = None,
        backend: str | None = None,
        assert_no_fallback: bool = False,
    ):
        """Compile the recorded steps into a callable `Patch -> Patch`.

        Class-authored operations are bound against a patch boundary, grouped
        into JIT segments, and reconstructed through the final boundary.
        """
        target_device = _resolve_target_device(device=device, backend=backend)
        cache_key = self._compile_cache_key(
            device=target_device,
            backend=backend,
            assert_no_fallback=assert_no_fallback,
        )
        cached = self._compiled_cache.get(cache_key)
        if cached is not None:
            return cached

        operations = self._resolve_operations()
        core_segment_cache = _BoundedCache(maxsize=_SEGMENT_CACHE_MAXSIZE)

        def _compiled_patch_fn(patch):
            patch_tree, boundary = PatchPyTree.from_patch(patch)
            current_tree = patch_tree
            current_boundary = boundary
            op_index = 0
            while op_index < len(operations):
                bound_ops, planned_boundary, needs_result_boundary, op_index = (
                    self._plan_core_segment(
                        operations,
                        current_boundary,
                        op_index,
                    )
                )
                segment_key = tuple(_operation_key(bound_operation) for bound_operation in bound_ops)
                compiled = core_segment_cache.get(segment_key)
                if compiled is None:
                    compiled = self._build_core_runner(bound_ops)
                    core_segment_cache.add(segment_key, compiled)
                if target_device is not None:
                    current_tree = tree_util.tree_map(
                        lambda x: jax.device_put(x, device=target_device),
                        current_tree,
                    )
                current_tree = compiled(current_tree)
                current_boundary = planned_boundary
                if needs_result_boundary:
                    current_boundary = bound_ops[-1].update_boundary_from_result(
                        current_tree,
                        current_boundary,
                    )
            return current_boundary.to_patch(current_tree)

        self._compiled_cache.add(cache_key, _compiled_patch_fn)
        return _compiled_patch_fn
