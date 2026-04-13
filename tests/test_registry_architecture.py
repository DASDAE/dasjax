"""Architecture-level checks for the operation registry rewrite."""

from __future__ import annotations

import inspect

import dasjax

from dasjax.operations.basic import impl as basic_impl
from dasjax.operations.filters import impl as filters_impl
from dasjax.operations import ExecutionPolicy, iter_operations
from dasjax.operations import basic, filters, signal, spectral
from dasjax.operations.signal import impl as signal_impl
from dasjax.operations.registry import OPERATIONS, list_operations
from dasjax.operations.spectral import impl as spectral_impl


def test_registry_operation_names_are_unique() -> None:
    names = [operation.name for operation in iter_operations()]
    assert len(names) == len(set(names))


def test_leaf_operations_define_leaf_transform() -> None:
    for operation in iter_operations():
        if operation.execution_policy is ExecutionPolicy.LEAF:
            assert operation.leaf_transform is not None
        else:
            assert operation.leaf_transform is None


def test_package_root_exports_only_runtime_surface() -> None:
    assert sorted(dasjax.__all__) == ["JaxPatchPipeline", "list_operations"]


def test_list_operations_preserves_registry_order() -> None:
    assert list_operations() == tuple(operation.name for operation in OPERATIONS)


def test_operation_families_export_specs_from_packages() -> None:
    for module in (basic, signal, filters, spectral):
        assert hasattr(module, "OPERATIONS")
        assert module.OPERATIONS
        assert module.__file__ is not None
        assert module.__file__.endswith("__init__.py")


def test_operation_impl_modules_avoid_private_dascore_imports() -> None:
    for module in (
        basic_impl,
        signal_impl,
        filters_impl,
        spectral_impl,
    ):
        source = inspect.getsource(module)
        assert "from dascore" in source
        assert " import _" not in source
