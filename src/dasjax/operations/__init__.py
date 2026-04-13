"""Operation registry package."""

from .registry import get_operation, iter_operations, list_operations
from .types import ExecutionPolicy, OperationCase, OperationSpec

__all__ = [
    "ExecutionPolicy",
    "OperationCase",
    "OperationSpec",
    "get_operation",
    "iter_operations",
    "list_operations",
]
