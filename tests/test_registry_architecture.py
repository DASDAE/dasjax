"""Architecture-level checks for the core operation registry."""

from __future__ import annotations

import dasjax
from dasjax import PatchOperation, get_patch_operation, iter_patch_operations


EXPECTED_OPERATIONS = (
    "identity",
    "scale",
    "add",
    "abs",
    "clip",
    "real",
    "imag",
    "angle",
    "conj",
    "flip",
    "roll",
    "standardize",
    "detrend",
    "normalize",
    "differentiate",
    "integrate",
    "taper",
    "taper_range",
    "gaussian_filter",
    "hampel_filter",
    "pass_filter",
    "pad",
    "hilbert",
    "envelope",
    "dft",
    "idft",
    "whiten",
    "fbe",
)


def test_core_operation_names_are_unique() -> None:
    names = [operation.operation_name() for operation in iter_patch_operations()]
    assert len(names) == len(set(names))


def test_all_operations_are_patch_operation_subclasses() -> None:
    for operation in iter_patch_operations():
        assert issubclass(operation, PatchOperation)


def test_list_patch_operations_preserves_registry_order() -> None:
    assert dasjax.list_patch_operations() == EXPECTED_OPERATIONS


def test_get_patch_operation_resolves_all_registered_names() -> None:
    for name in EXPECTED_OPERATIONS:
        assert get_patch_operation(name).operation_name() == name


def test_package_root_exports_only_runtime_surface() -> None:
    assert sorted(dasjax.__all__) == [
        "JaxPatchPipeline",
        "PatchBoundary",
        "PatchOperation",
        "PatchPyTree",
        "get_patch_operation",
        "iter_patch_operations",
        "list_patch_operations",
    ]
