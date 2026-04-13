"""Registry specs for signal operations."""

from __future__ import annotations

import dascore as dc
from dascore.proc.taper import taper as dc_taper
from dascore.proc.taper import taper_range as dc_taper_range
from dascore.transform.differentiate import differentiate as dc_differentiate
from dascore.transform.integrate import integrate as dc_integrate

from ..common import baseline_patch_method
from ..types import ExecutionPolicy, OperationCase, OperationSpec
from . import impl

OPERATIONS: tuple[OperationSpec, ...] = (
    OperationSpec(
        name="detrend",
        execution_policy=ExecutionPolicy.LEAF,
        patch_impl=dc.patch_function(history="method_name")(impl.detrend_patch),
        patch_op_cls=impl.DetrendOp,
        leaf_transform=impl.detrend_leaves,
        validate_patch=impl.validate_detrend_patch_input,
        test_cases=tuple(
            OperationCase(
                case_id=f"detrend-{kind}",
                resolve_call=lambda patch, kind=kind: (
                    (),
                    {"dim": patch.dims[-1], "type": kind},
                ),
                baseline=lambda patch, args, kwargs: baseline_patch_method(
                    patch, args, kwargs, "detrend"
                ),
            )
            for kind in ("constant", "linear")
        ),
    ),
    OperationSpec(
        name="normalize",
        execution_policy=ExecutionPolicy.LEAF,
        patch_impl=dc.patch_function(history="method_name")(impl.normalize_patch),
        patch_op_cls=impl.NormalizeOp,
        leaf_transform=impl.normalize_leaves,
        test_cases=tuple(
            OperationCase(
                case_id=f"normalize-{norm}",
                resolve_call=lambda patch, norm=norm: (
                    (),
                    {"dim": patch.dims[-1], "norm": norm},
                ),
                baseline=lambda patch, args, kwargs: baseline_patch_method(
                    patch, args, kwargs, "normalize"
                ),
            )
            for norm in ("l1", "l2", "max", "bit")
        ),
    ),
    OperationSpec(
        name="differentiate",
        execution_policy=ExecutionPolicy.PATCH,
        patch_impl=dc.patch_function(history="method_name")(impl.differentiate_patch),
        patch_op_cls=impl.DifferentiateOp,
        test_cases=(
            OperationCase(
                case_id="differentiate-default",
                resolve_call=lambda patch: ((), {"dim": patch.dims[-1]}),
                baseline=lambda patch, args, kwargs: dc_differentiate.func(
                    patch, *args, **kwargs
                ),
            ),
            OperationCase(
                case_id="differentiate-step",
                resolve_call=lambda patch: ((), {"dim": patch.dims[-1], "step": 2}),
                baseline=lambda patch, args, kwargs: dc_differentiate.func(
                    patch, *args, **kwargs
                ),
            ),
        ),
    ),
    OperationSpec(
        name="integrate",
        execution_policy=ExecutionPolicy.PATCH,
        patch_impl=dc.patch_function(history="method_name")(impl.integrate_patch),
        patch_op_cls=impl.IntegrateOp,
        test_cases=(
            OperationCase(
                case_id="integrate-indefinite",
                resolve_call=lambda patch: ((), {"dim": patch.dims[-1]}),
                baseline=lambda patch, args, kwargs: dc_integrate.func(
                    patch, *args, **kwargs
                ),
            ),
            OperationCase(
                case_id="integrate-definite",
                resolve_call=lambda patch: (
                    (),
                    {"dim": patch.dims[-1], "definite": True},
                ),
                baseline=lambda patch, args, kwargs: dc_integrate.func(
                    patch, *args, **kwargs
                ),
            ),
        ),
    ),
    OperationSpec(
        name="taper",
        execution_policy=ExecutionPolicy.LEAF,
        patch_impl=dc.patch_function(history="method_name")(impl.taper_patch),
        patch_op_cls=impl.TaperOp,
        leaf_transform=impl.taper_leaves,
        prepare_call=impl.prepare_taper_call,
        test_cases=(
            OperationCase(
                case_id="taper-time",
                resolve_call=lambda patch: (
                    (),
                    {"window_type": "hann", patch.dims[-1]: 0.05},
                ),
                baseline=lambda patch, args, kwargs: dc_taper.func(
                    patch, *args, **kwargs
                ),
            ),
        ),
    ),
    OperationSpec(
        name="taper_range",
        execution_policy=ExecutionPolicy.LEAF,
        patch_impl=dc.patch_function(history="method_name")(impl.taper_range_patch),
        patch_op_cls=impl.TaperRangeOp,
        leaf_transform=impl.taper_range_leaves,
        prepare_call=impl.prepare_taper_range_call,
        test_cases=(
            OperationCase(
                case_id="taper-range",
                resolve_call=lambda patch: (
                    (),
                    {"window_type": "hann", patch.dims[-1]: (10, 20), "samples": True},
                ),
                baseline=lambda patch, args, kwargs: dc_taper_range.func(
                    patch, *args, **kwargs
                ),
            ),
        ),
    ),
)
