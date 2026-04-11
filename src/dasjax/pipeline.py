"""Pipeline support for chaining JAX patch operations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

import jax
import numpy as np
from jax import tree_util

from .operations import get_operation
from .pytree import patch_to_leaves, patch_from_leaves


@dataclass(frozen=True)
class PipelineStep:
    """A recorded operation with its bound arguments."""

    name: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


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
    """Split resolved steps into JIT segments and eager (shape-changing) steps.

    Returns a list of tagged tuples:
      ("jit",   tuple_of_resolved_steps)   — compiled as a single jax.jit block
      ("eager", (op, args, kwargs))         — executed via op.patch_impl
    """
    segments: list[tuple[str, Any]] = []
    jit_run: list[Any] = []
    for op, args, kwargs in resolved_steps:
        if op.shape_changing:
            if jit_run:
                segments.append(("jit", tuple(jit_run)))
                jit_run = []
            segments.append(("eager", (op, args, kwargs)))
        else:
            jit_run.append((op, args, kwargs))
    if jit_run:
        segments.append(("jit", tuple(jit_run)))
    return segments


class JaxPatchPipeline:
    """A reusable recipe composed of registered dasjax operations."""

    _compiled_cache: ClassVar[dict[tuple[Any, ...], Callable[[Any], Any]]] = {}

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
                data, coord_leaves = op.leaf_transform(
                    data,
                    coord_leaves,
                    *args,
                    dims=dims,
                    **kwargs,
                )
            return data, coord_leaves

        return jax.jit(_run, static_argnums=2)

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

        Shape-preserving operations are fused into jax.jit-compiled segments.
        Shape-changing operations (e.g. fbe) run eagerly via patch_impl between
        those segments, so the compiled function correctly handles pipelines like
        ``scale → fbe → normalize``.
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
        # One inner cache dict per JIT segment (None for eager slots).
        compiled_by_segment: list[
            dict[tuple[Any, ...], Callable[[Any, Any, Any], Any]] | None
        ] = [{} if kind == "jit" else None for kind, _ in segments]

        def _compiled_patch_fn(patch):
            if assert_no_fallback:
                self.assert_no_fallback(patch)
            for op, args, kwargs in resolved_steps:
                if op.validate_patch is not None:
                    op.validate_patch(patch, *args, **kwargs)
                if op.validate_compiled_patch is not None:
                    op.validate_compiled_patch(patch, args, kwargs)

            current_patch = patch
            for seg_idx, (kind, seg_data) in enumerate(segments):
                if kind == "eager":
                    op, args, kwargs = seg_data
                    current_patch = op.patch_impl(current_patch, *args, **kwargs)
                else:
                    jit_steps = seg_data
                    prepared_steps = tuple(
                        (op, *op.prepare_call(current_patch, args, kwargs))
                        if op.prepare_call is not None
                        else (op, args, kwargs)
                        for op, args, kwargs in jit_steps
                    )
                    prepared_key = self._freeze_resolved_steps(prepared_steps)
                    seg_cache = compiled_by_segment[seg_idx]
                    compiled = seg_cache.get(prepared_key)
                    if compiled is None:
                        compiled = self._compile_leaf_runner(prepared_steps)
                        seg_cache[prepared_key] = compiled
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
                    current_patch = patch_from_leaves(
                        out_data, out_coord_leaves, aux_data
                    )

            return current_patch

        self._compiled_cache[cache_key] = _compiled_patch_fn
        return _compiled_patch_fn
