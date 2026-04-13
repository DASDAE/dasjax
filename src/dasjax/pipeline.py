"""Pipeline support for chaining JAX patch operations."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, ClassVar

import jax
import numpy as np
from jax import tree_util

from .operations import ExecutionPolicy, get_operation
from .operations.patch_ops import patch_to_state_spec
from .pytree import patch_to_leaves, patch_from_leaves

_COMPILED_CACHE_MAXSIZE = 128
_SEGMENT_CACHE_MAXSIZE = 128
_VALIDATION_CACHE_MAXSIZE = 64


@dataclass(frozen=True)
class PipelineStep:
    """A recorded operation with its bound arguments."""

    name: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


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
    except TypeError as exc:
        msg = f"Unhashable pipeline argument {value!r} cannot be compiled."
        raise TypeError(msg) from exc
    return value


def _build_segments(
    resolved_steps: tuple[tuple[Any, tuple[Any, ...], dict[str, Any]], ...],
) -> list[tuple[str, Any]]:
    """Split steps into compiled segments and eager patch-executed steps.

    Returns a list of tagged tuples:
      ("patchop", tuple_of_resolved_steps)      — compiled PatchOp segment
      ("leaf",  tuple_of_resolved_steps)        — compiled legacy leaf segment
      ("patch", (op, args, kwargs))             — eager patch_impl step
    """
    segments: list[tuple[str, Any]] = []
    patchop_run: list[Any] = []
    patchop_has_structural = False
    leaf_run: list[Any] = []
    for op, args, kwargs in resolved_steps:
        if op.patch_op_cls is not None:
            if leaf_run:
                segments.append(("leaf", tuple(leaf_run)))
                leaf_run = []
            if (
                patchop_run
                and patchop_has_structural
                and getattr(
                    op.patch_op_cls, "requires_materialized_patch_for_prepare", False
                )
            ):
                segments.append(("patchop", tuple(patchop_run)))
                patchop_run = []
                patchop_has_structural = False
            patchop_run.append((op, args, kwargs))
            patchop_has_structural = patchop_has_structural or getattr(
                op.patch_op_cls, "mutates_spec", False
            )
            if getattr(op.patch_op_cls, "requires_materialized_patch_after", False):
                segments.append(("patchop", tuple(patchop_run)))
                patchop_run = []
                patchop_has_structural = False
            continue
        if op.execution_policy is ExecutionPolicy.PATCH:
            if patchop_run:
                segments.append(("patchop", tuple(patchop_run)))
                patchop_run = []
                patchop_has_structural = False
            if leaf_run:
                segments.append(("leaf", tuple(leaf_run)))
                leaf_run = []
            segments.append(("patch", (op, args, kwargs)))
        else:
            if patchop_run:
                segments.append(("patchop", tuple(patchop_run)))
                patchop_run = []
                patchop_has_structural = False
            leaf_run.append((op, args, kwargs))
    if patchop_run:
        segments.append(("patchop", tuple(patchop_run)))
    if leaf_run:
        segments.append(("leaf", tuple(leaf_run)))
    return segments


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
        get_operation(name)

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

    def _resolve_steps(self) -> tuple[tuple[Any, tuple[Any, ...], dict[str, Any]], ...]:
        """Resolve operation names into operation specs once."""
        return tuple(
            (get_operation(step.name), step.args, step.kwargs) for step in self._steps
        )

    @staticmethod
    def _freeze_resolved_steps(
        resolved_steps: tuple[tuple[Any, tuple[Any, ...], dict[str, Any]], ...],
    ) -> tuple[Any, ...]:
        return tuple(
            (
                op.name,
                _freeze_value(args),
                _freeze_value(kwargs),
            )
            for op, args, kwargs in resolved_steps
        )

    @staticmethod
    def _compile_leaf_runner(
        resolved_steps: tuple[tuple[Any, tuple[Any, ...], dict[str, Any]], ...],
    ):
        """Compile the leaf-native execution path once per pipeline definition."""

        def _run(data, coord_leaves, dims):
            for op, args, kwargs in resolved_steps:
                assert op.leaf_transform is not None
                data, coord_leaves = op.leaf_transform(
                    data,
                    coord_leaves,
                    *args,
                    dims=dims,
                    **kwargs,
                )
            return data, coord_leaves

        return jax.jit(_run, static_argnums=2)

    @staticmethod
    def _compile_patchop_runner(prepared_ops: tuple[Any, ...]):
        """Compile one unified PatchOp segment."""

        def _run(state):
            current_state = state
            for prepared_op in prepared_ops:
                current_state = prepared_op.apply(current_state)
            return current_state

        return jax.jit(_run)

    def _get_compiled_fallback_reasons(self, patch) -> tuple[str, ...]:
        resolved_steps = self._resolve_steps()
        reasons = []
        for op, args, kwargs in resolved_steps:
            if op.compiled_fallback_reason is None:
                continue
            reason = op.compiled_fallback_reason(patch, args, kwargs)
            if reason is not None:
                reasons.append(f"{op.name}: {reason}")
        return tuple(reasons)

    def assert_no_fallback(self, patch) -> None:
        """Raise if any step would use a host fallback for this patch."""
        reasons = self._get_compiled_fallback_reasons(patch)
        if reasons:
            msg = "Compiled pipeline would use host fallbacks:\n" + "\n".join(reasons)
            raise AssertionError(msg)

    def compile(
        self,
        *,
        device: jax.Device | None = None,
        backend: str | None = None,
        assert_no_fallback: bool = False,
    ):
        """Compile the recorded steps into a callable `Patch -> Patch`.

        Leaf-native operations are fused into jax.jit-compiled segments.
        Patch-native operations run eagerly via patch_impl between those
        segments, so compiled pipelines can mix both execution modes.
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

        resolved_steps = self._resolve_steps()
        segments = _build_segments(resolved_steps)
        # One inner cache dict per leaf segment (None for patch slots).
        compiled_by_segment: list[_BoundedCache | None] = [
            _BoundedCache(maxsize=_SEGMENT_CACHE_MAXSIZE)
            if kind in {"leaf", "patchop"}
            else None
            for kind, _ in segments
        ]
        validated_signatures = _BoundedCache(maxsize=_VALIDATION_CACHE_MAXSIZE)
        fallback_checked_signatures = _BoundedCache(maxsize=_VALIDATION_CACHE_MAXSIZE)

        def _compiled_patch_fn(patch):
            patch_signature = self._validation_signature(patch)
            if (
                assert_no_fallback
                and fallback_checked_signatures.get(patch_signature) is None
            ):
                self.assert_no_fallback(patch)
                fallback_checked_signatures.add(patch_signature, True)
            if validated_signatures.get(patch_signature) is None:
                for op, args, kwargs in resolved_steps:
                    if op.validate_patch is not None:
                        op.validate_patch(patch, *args, **kwargs)
                    if op.validate_compiled_patch is not None:
                        op.validate_compiled_patch(patch, args, kwargs)
                validated_signatures.add(patch_signature, True)

            current_patch = patch
            for seg_idx, (kind, seg_data) in enumerate(segments):
                if kind == "patch":
                    op, args, kwargs = seg_data
                    current_patch = op.patch_impl(current_patch, *args, **kwargs)
                elif kind == "patchop":
                    prepared_ops = tuple(
                        op.patch_op_cls.prepare(current_patch, *args, **kwargs)
                        for op, args, kwargs in seg_data
                    )
                    prepared_key = tuple(
                        prepared_op.compile_key() for prepared_op in prepared_ops
                    )
                    seg_cache = compiled_by_segment[seg_idx]
                    compiled = seg_cache.get(prepared_key)
                    if compiled is None:
                        compiled = self._compile_patchop_runner(prepared_ops)
                        seg_cache.add(prepared_key, compiled)
                    state, spec = patch_to_state_spec(current_patch)
                    if target_device is not None:
                        state = tree_util.tree_map(
                            lambda x: jax.device_put(x, device=target_device), state
                        )
                    out_state = compiled(state)
                    out_spec = spec
                    for prepared_op in prepared_ops:
                        out_spec = out_spec.apply_meta(prepared_op.meta_delta(out_spec))
                    current_patch = prepared_ops[-1].reconstruct(
                        current_patch,
                        out_spec,
                        out_state,
                    )
                else:
                    leaf_steps = seg_data
                    prepared_steps = tuple(
                        (op, *op.prepare_call(current_patch, args, kwargs))
                        if op.prepare_call is not None
                        else (op, args, kwargs)
                        for op, args, kwargs in leaf_steps
                    )
                    prepared_key = self._freeze_resolved_steps(prepared_steps)
                    seg_cache = compiled_by_segment[seg_idx]
                    compiled = seg_cache.get(prepared_key)
                    if compiled is None:
                        compiled = self._compile_leaf_runner(prepared_steps)
                        seg_cache.add(prepared_key, compiled)
                    data, coord_leaves, aux_data = patch_to_leaves(current_patch)
                    if target_device is not None:
                        data = jax.device_put(data, device=target_device)
                        coord_leaves = tree_util.tree_map(
                            lambda x: jax.device_put(x, device=target_device),
                            coord_leaves,
                        )
                    out_data, out_coord_leaves = compiled(
                        data,
                        coord_leaves,
                        aux_data["dims"],
                    )
                    next_kind = (
                        segments[seg_idx + 1][0]
                        if seg_idx + 1 < len(segments)
                        else None
                    )
                    current_patch = patch_from_leaves(
                        out_data,
                        out_coord_leaves,
                        aux_data,
                        coerce_numpy=next_kind != "leaf",
                    )

            return current_patch

        self._compiled_cache.add(cache_key, _compiled_patch_fn)
        return _compiled_patch_fn
