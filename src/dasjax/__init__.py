"""DASCore extensions for JAX-backed patch processing."""

# Ruff would normally require imports before executable statements, but this
# flag must be set before importing package modules that initialize JAX objects.
# ruff: noqa: E402

from importlib.metadata import PackageNotFoundError, version

import jax

jax.config.update("jax_enable_x64", True)

from . import pytree as pytree
from .operations import JaxPatchOperation, list_operations
from .pipeline import JaxPatchPipeline

__all__ = ["JaxPatchOperation", "JaxPatchPipeline", "list_operations"]

try:
    __version__ = version("dasjax")
except PackageNotFoundError:
    __version__ = "0+unknown"
