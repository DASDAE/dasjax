"""Compare internal dasjax operations against DASCore baselines."""

from __future__ import annotations

import numpy as np
import pytest

from dasjax import JaxPatchPipeline, list_operations
from dasjax.operations import get_operation, iter_operations


def _assert_patch_data_close(left, right) -> None:
    """Assert two patches have numerically close data and identical coords."""
    assert np.allclose(left.data, right.data, equal_nan=True, rtol=1e-5, atol=1e-6)
    assert left.coords == right.coords


def _assert_same_exception(left_exc: Exception, right_exc: Exception) -> None:
    """Assert two operations failed in the same way."""
    assert type(left_exc) is type(right_exc)
    assert str(left_exc) == str(right_exc)


def _iter_operation_cases():
    """Yield every public operation together with its declared test cases."""
    for operation in iter_operations():
        for case in operation.test_cases:
            yield operation, case


@pytest.mark.parametrize("operation_name", list_operations())
def test_list_operations_matches_registry(operation_name: str) -> None:
    """Expose every registered operation through list_operations."""
    assert get_operation(operation_name).name == operation_name


@pytest.mark.parametrize(
    ("operation", "case"),
    list(_iter_operation_cases()),
    ids=lambda item: getattr(item, "case_id", getattr(item, "name", repr(item))),
)
def test_operation_patch_impl_matches_declared_baseline(
    example_patch,
    operation,
    case,
) -> None:
    """Match each eager operation implementation to its declared baseline."""
    patch = case.prepare_patch(example_patch)
    args, kwargs = case.resolve_call(patch)

    try:
        expected = case.baseline(patch, args, kwargs)
    except Exception as expected_exc:
        try:
            operation.patch_impl(patch, *args, **kwargs)
        except Exception as out_exc:
            _assert_same_exception(out_exc, expected_exc)
            return
        raise AssertionError("Expected internal patch operation to fail like DASCore.")

    out = operation.patch_impl(patch, *args, **kwargs)
    _assert_patch_data_close(out, expected)


@pytest.mark.parametrize(
    ("operation", "case"),
    list(_iter_operation_cases()),
    ids=lambda item: getattr(item, "case_id", getattr(item, "name", repr(item))),
)
def test_compiled_pipeline_matches_declared_baseline(
    example_patch,
    operation,
    case,
) -> None:
    """Match each compiled pipeline operation to its declared baseline."""
    patch = case.prepare_patch(example_patch)
    args, kwargs = case.resolve_call(patch)
    pipeline = getattr(JaxPatchPipeline(), operation.name)(*args, **kwargs)

    if case.compiled_guard is not None:
        guarded = case.compiled_guard(patch, args, kwargs)
        if guarded is not None:
            error_type, error_message = guarded
            with pytest.raises(error_type, match=error_message):
                pipeline.compile()(patch)
            return

    if case.compiled_error is not None:
        with pytest.raises(
            case.compiled_error, match=case.compiled_error_message or ""
        ):
            pipeline.compile()(patch)
        return

    try:
        baseline = case.compiled_baseline or case.baseline
        expected = baseline(patch, args, kwargs)
    except Exception as expected_exc:
        try:
            pipeline.compile()(patch)
        except Exception as out_exc:
            _assert_same_exception(out_exc, expected_exc)
            return
        raise AssertionError("Expected compiled pipeline to fail like DASCore.")

    out = pipeline.compile()(patch)
    _assert_patch_data_close(out, expected)
