"""Operation registry package."""

from .patch_ops import EagerPatchOp, OpResult, PatchOp, PatchSpec, PatchState
from .registry import (
    get_operation,
    iter_operations,
    list_operations,
)
from .types import ExecutionPolicy, OperationCase, OperationSpec

__all__ = [
    "ExecutionPolicy",
    "EagerPatchOp",
    "OpResult",
    "OperationCase",
    "OperationSpec",
    "PatchOp",
    "PatchSpec",
    "PatchState",
    "get_operation",
    "iter_operations",
    "list_operations",
]
