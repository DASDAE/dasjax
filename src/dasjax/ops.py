"""Compatibility wrappers for the refactored operation registry."""

from .operations import (
    JaxPatchOperation,
    get_operation,
    iter_operations,
    list_operations,
)

__all__ = ["JaxPatchOperation", "get_operation", "iter_operations", "list_operations"]
