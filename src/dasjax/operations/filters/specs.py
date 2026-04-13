"""Registry specs for filter operations."""

from __future__ import annotations

import dascore as dc

from ..common import baseline_patch_method, prepare_even_patch_for_dim
from ..types import ExecutionPolicy, OperationCase, OperationSpec
from . import impl

OPERATIONS: tuple[OperationSpec, ...] = (
    OperationSpec(
        name="gaussian_filter",
        execution_policy=ExecutionPolicy.LEAF,
        patch_impl=dc.patch_function(history="method_name")(impl.gaussian_filter_patch),
        leaf_transform=impl.gaussian_filter_leaves,
        prepare_call=impl.prepare_gaussian_call,
        compiled_fallback_reason=impl.fallback_gaussian_filter_reason,
        test_cases=(
            OperationCase(
                case_id="gaussian-filter-default",
                resolve_call=lambda patch: ((), {patch.dims[-1]: 3, "samples": True}),
                baseline=lambda patch, args, kwargs: baseline_patch_method(patch, args, kwargs, "gaussian_filter"),
            ),
            OperationCase(
                case_id="gaussian-filter-constant",
                resolve_call=lambda patch: ((), {patch.dims[0]: 3, "samples": True, "mode": "constant", "cval": 0.25}),
                baseline=lambda patch, args, kwargs: baseline_patch_method(patch, args, kwargs, "gaussian_filter"),
            ),
        ),
    ),
    OperationSpec(
        name="hampel_filter",
        execution_policy=ExecutionPolicy.LEAF,
        patch_impl=dc.patch_function(history="method_name")(impl.hampel_filter_patch),
        leaf_transform=impl.hampel_filter_leaves,
        prepare_call=impl.prepare_hampel_call,
        validate_patch=impl.validate_hampel_filter_patch_input,
        validate_compiled_patch=impl.validate_hampel_filter_compiled_input,
        compiled_fallback_reason=impl.fallback_hampel_filter_reason,
        test_cases=(
            OperationCase(
                case_id="hampel-filter-approximate",
                resolve_call=lambda patch: ((), {patch.dims[-1]: 3, "samples": True, "threshold": 3.5, "approximate": True}),
                baseline=lambda patch, args, kwargs: baseline_patch_method(patch, args, kwargs, "hampel_filter"),
                compiled_guard=impl.guard_compiled_hampel_case,
            ),
            OperationCase(
                case_id="hampel-filter-exact",
                resolve_call=lambda patch: ((), {patch.dims[-1]: 3, "samples": True, "threshold": 3.5, "approximate": False}),
                baseline=lambda patch, args, kwargs: baseline_patch_method(patch, args, kwargs, "hampel_filter"),
                compiled_guard=impl.guard_compiled_hampel_case,
            ),
        ),
    ),
    OperationSpec(
        name="pass_filter",
        execution_policy=ExecutionPolicy.LEAF,
        patch_impl=dc.patch_function(history="method_name")(impl.pass_filter_patch),
        leaf_transform=impl.pass_filter_leaves,
        prepare_call=impl.prepare_pass_filter_call,
        test_cases=(
            OperationCase(
                case_id="pass-filter-bandpass",
                resolve_call=lambda patch: ((), {patch.dims[-1]: (1.0, 10.0), "corners": 4, "zerophase": True}),
                prepare_patch=prepare_even_patch_for_dim,
                baseline=lambda patch, args, kwargs: baseline_patch_method(patch, args, kwargs, "pass_filter"),
            ),
            OperationCase(
                case_id="pass-filter-lowpass",
                resolve_call=lambda patch: ((), {patch.dims[-1]: (None, 12.0), "corners": 2, "zerophase": False}),
                prepare_patch=prepare_even_patch_for_dim,
                baseline=lambda patch, args, kwargs: baseline_patch_method(patch, args, kwargs, "pass_filter"),
            ),
        ),
    ),
)
