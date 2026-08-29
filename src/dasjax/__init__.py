"""DASCore extensions for JAX-backed patch processing."""

# Ruff would normally require imports before executable statements, but this
# flag must be set before importing package modules that initialize JAX objects.
# ruff: noqa: E402

from importlib.metadata import PackageNotFoundError, version

import jax

jax.config.update("jax_enable_x64", True)

from .core import (
    PatchBoundary,
    PatchOperation,
    PatchPyTree,
    get_patch_operation,
    iter_patch_operations,
    list_patch_operations,
)
from . import operations as operations
from . import core_ops as core_ops
from .pipeline import JaxPatchPipeline

__all__ = [
    "JaxPatchPipeline",
    "PatchBoundary",
    "PatchOperation",
    "PatchPyTree",
    "get_patch_operation",
    "iter_patch_operations",
    "list_patch_operations",
]

try:
    __version__ = version("dasjax")
except PackageNotFoundError:
    __version__ = "0+unknown"
