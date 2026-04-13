"""Shared types for operation specs and registry assembly."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from dascore.constants import PatchType

LeafTransform = Callable[..., tuple[Any, tuple[Any, ...]]]
PatchValidator = Callable[..., None]
CompiledPatchValidator = Callable[[PatchType, tuple[Any, ...], dict[str, Any]], None]
CompiledFallbackReason = Callable[
    [PatchType, tuple[Any, ...], dict[str, Any]], str | None
]
CompiledCaseGuard = Callable[
    [PatchType, tuple[Any, ...], dict[str, Any]],
    tuple[type[Exception], str] | None,
]
CallResolver = Callable[[PatchType], tuple[tuple[Any, ...], dict[str, Any]]]
Baseline = Callable[[PatchType, tuple[Any, ...], dict[str, Any]], PatchType]
PatchPreparer = Callable[[PatchType], PatchType]
CallPreparer = Callable[
    [PatchType, tuple[Any, ...], dict[str, Any]],
    tuple[tuple[Any, ...], dict[str, Any]],
]


class ExecutionPolicy(str, Enum):
    """How one operation executes inside compiled pipelines."""

    LEAF = "leaf"
    PATCH = "patch"


def _identity_prepare(patch: PatchType) -> PatchType:
    return patch


def empty_call(_: PatchType) -> tuple[tuple[Any, ...], dict[str, Any]]:
    return (), {}


@dataclass(frozen=True)
class OperationCase:
    """One shared test case for eager and compiled parity."""

    case_id: str
    resolve_call: CallResolver
    baseline: Baseline
    prepare_patch: PatchPreparer = _identity_prepare
    compiled_baseline: Baseline | None = None
    compiled_error: type[Exception] | None = None
    compiled_error_message: str | None = None
    compiled_guard: CompiledCaseGuard | None = None


@dataclass(frozen=True)
class OperationSpec:
    """Read-only operation spec for patch namespace and pipeline support."""

    name: str
    execution_policy: ExecutionPolicy
    patch_impl: Callable[..., PatchType]
    leaf_transform: LeafTransform | None = None
    prepare_call: CallPreparer | None = None
    validate_patch: PatchValidator | None = None
    validate_compiled_patch: CompiledPatchValidator | None = None
    compiled_fallback_reason: CompiledFallbackReason | None = None
    test_cases: tuple[OperationCase, ...] = field(default_factory=tuple)

