"""Architecture-level checks for the operation registry rewrite."""

from __future__ import annotations

import inspect

import dasjax

from dasjax.operations.basic import impl as basic_impl
from dasjax.operations.filters import impl as filters_impl
from dasjax.operations import ExecutionPolicy, PatchOp, iter_operations
from dasjax.operations import basic, filters, signal, spectral
from dasjax.operations.signal import impl as signal_impl
from dasjax.operations.registry import OPERATIONS, list_operations
from dasjax.operations.spectral import impl as spectral_impl


def test_registry_operation_names_are_unique() -> None:
    """Keep registered operation names unique."""
    names = [operation.name for operation in iter_operations()]
    assert len(names) == len(set(names))


def test_leaf_operations_define_leaf_transform() -> None:
    """Require leaf operations to expose only the leaf transform path."""
    for operation in iter_operations():
        if operation.execution_policy is ExecutionPolicy.LEAF:
            assert operation.leaf_transform is not None
        else:
            assert operation.leaf_transform is None


def test_all_operations_define_patch_op_class() -> None:
    """Ensure every registered operation exposes a PatchOp class."""
    for operation in iter_operations():
        assert operation.patch_op_cls is not None


def test_patch_op_subclass_registry_is_populated() -> None:
    """Register concrete PatchOp subclasses for runtime introspection."""
    subclass_names = {cls.__name__ for cls in PatchOp.iter_subclasses()}
    assert "ScaleOp" in subclass_names
    assert "PadOp" in subclass_names


def test_patch_op_compile_category_table_contains_expected_groups() -> None:
    """Expose the expected compile categories through the PatchOp table."""
    table = PatchOp.compile_category_table()
    assert "kernel_fusible" in table
    assert "compiled_boundary" in table
    assert "eager_boundary" in table


def test_package_root_exports_only_runtime_surface() -> None:
    """Keep the root package export surface intentionally small."""
    assert sorted(dasjax.__all__) == ["JaxPatchPipeline", "list_operations"]


def test_list_operations_preserves_registry_order() -> None:
    """Preserve registry order when listing operation names."""
    assert list_operations() == tuple(operation.name for operation in OPERATIONS)


def test_operation_families_export_specs_from_packages() -> None:
    """Expose operation spec collections from each family package."""
    for module in (basic, signal, filters, spectral):
        assert hasattr(module, "OPERATIONS")
        assert module.OPERATIONS
        assert module.__file__ is not None
        assert module.__file__.endswith("__init__.py")


def test_operation_impl_modules_avoid_private_dascore_imports() -> None:
    """Avoid private DASCore imports inside operation implementations."""
    for module in (
        basic_impl,
        signal_impl,
        filters_impl,
        spectral_impl,
    ):
        source = inspect.getsource(module)
        assert "from dascore" in source
        assert " import _" not in source
