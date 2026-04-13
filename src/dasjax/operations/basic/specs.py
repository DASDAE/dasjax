"""Registry specs for basic operations."""

from __future__ import annotations

import dascore as dc
from dascore.proc.basic import angle as dc_angle
from dascore.proc.basic import conj as dc_conj
from dascore.proc.basic import flip as dc_flip
from dascore.proc.basic import imag as dc_imag
from dascore.proc.basic import real as dc_real
from dascore.proc.basic import roll as dc_roll
from dascore.proc.basic import standardize as dc_standardize

from dasjax import kernels

from ..common import prepare_complex_patch
from ..types import ExecutionPolicy, OperationCase, OperationSpec, empty_call
from . import impl

OPERATIONS: tuple[OperationSpec, ...] = (
    OperationSpec(
        name="identity",
        execution_policy=ExecutionPolicy.LEAF,
        patch_impl=dc.patch_function(history=None)(impl.identity_patch),
        patch_op_cls=impl.IdentityOp,
        leaf_transform=impl.identity_leaves,
        test_cases=(
            OperationCase(
                case_id="identity",
                resolve_call=empty_call,
                baseline=lambda patch, args, kwargs: patch,
            ),
        ),
    ),
    OperationSpec(
        name="scale",
        execution_policy=ExecutionPolicy.LEAF,
        patch_impl=dc.patch_function(history="method_name")(impl.scale_patch),
        patch_op_cls=impl.ScaleOp,
        leaf_transform=impl.scale_leaves,
        test_cases=tuple(
            OperationCase(
                case_id=f"scale-{value}",
                resolve_call=lambda patch, value=value: ((value,), {}),
                baseline=lambda patch, args, kwargs: patch * args[0],
            )
            for value in (2.0, -0.75)
        ),
    ),
    OperationSpec(
        name="add",
        execution_policy=ExecutionPolicy.LEAF,
        patch_impl=dc.patch_function(history="method_name")(impl.add_patch),
        patch_op_cls=impl.AddOp,
        leaf_transform=impl.add_leaves,
        test_cases=tuple(
            OperationCase(
                case_id=f"add-{value}",
                resolve_call=lambda patch, value=value: ((value,), {}),
                baseline=lambda patch, args, kwargs: patch + args[0],
            )
            for value in (2.0, -0.75)
        ),
    ),
    OperationSpec(
        name="abs",
        execution_policy=ExecutionPolicy.LEAF,
        patch_impl=dc.patch_function(history="method_name")(impl.abs_patch),
        patch_op_cls=impl.AbsOp,
        leaf_transform=impl.abs_leaves,
        test_cases=(
            OperationCase(
                case_id="abs",
                resolve_call=empty_call,
                prepare_patch=prepare_complex_patch,
                baseline=lambda patch, args, kwargs: patch.abs(),
            ),
        ),
    ),
    OperationSpec(
        name="clip",
        execution_policy=ExecutionPolicy.LEAF,
        patch_impl=dc.patch_function(history="method_name")(impl.clip_patch),
        patch_op_cls=impl.ClipOp,
        leaf_transform=impl.clip_leaves,
        test_cases=tuple(
            OperationCase(
                case_id=f"clip-{low}-{high}",
                resolve_call=lambda patch, low=low, high=high: ((low, high), {}),
                baseline=lambda patch, args, kwargs, low=low, high=high: patch.update(
                    data=kernels.clip_kernel(patch.data, low, high)
                ),
            )
            for low, high in ((-0.25, 0.25), (-1.0, 1.5))
        ),
    ),
    OperationSpec(
        name="real",
        execution_policy=ExecutionPolicy.LEAF,
        patch_impl=dc.patch_function(history="method_name")(impl.real_patch),
        patch_op_cls=impl.RealOp,
        leaf_transform=impl.real_leaves,
        test_cases=(
            OperationCase(
                case_id="real",
                resolve_call=empty_call,
                prepare_patch=prepare_complex_patch,
                baseline=lambda patch, args, kwargs: dc_real.func(patch),
            ),
        ),
    ),
    OperationSpec(
        name="imag",
        execution_policy=ExecutionPolicy.LEAF,
        patch_impl=dc.patch_function(history="method_name")(impl.imag_patch),
        patch_op_cls=impl.ImagOp,
        leaf_transform=impl.imag_leaves,
        test_cases=(
            OperationCase(
                case_id="imag",
                resolve_call=empty_call,
                prepare_patch=prepare_complex_patch,
                baseline=lambda patch, args, kwargs: dc_imag.func(patch),
            ),
        ),
    ),
    OperationSpec(
        name="angle",
        execution_policy=ExecutionPolicy.LEAF,
        patch_impl=dc.patch_function(history="method_name")(impl.angle_patch),
        patch_op_cls=impl.AngleOp,
        leaf_transform=impl.angle_leaves,
        test_cases=(
            OperationCase(
                case_id="angle",
                resolve_call=empty_call,
                prepare_patch=prepare_complex_patch,
                baseline=lambda patch, args, kwargs: dc_angle.func(patch),
            ),
        ),
    ),
    OperationSpec(
        name="conj",
        execution_policy=ExecutionPolicy.LEAF,
        patch_impl=dc.patch_function(history="method_name")(impl.conj_patch),
        patch_op_cls=impl.ConjOp,
        leaf_transform=impl.conj_leaves,
        test_cases=(
            OperationCase(
                case_id="conj",
                resolve_call=empty_call,
                prepare_patch=prepare_complex_patch,
                baseline=lambda patch, args, kwargs: dc_conj.func(patch),
            ),
        ),
    ),
    OperationSpec(
        name="flip",
        execution_policy=ExecutionPolicy.PATCH,
        patch_impl=dc.patch_function(history="method_name")(impl.flip_patch),
        patch_op_cls=impl.FlipOp,
        test_cases=(
            OperationCase(
                case_id="flip-time",
                resolve_call=lambda patch: ((patch.dims[-1],), {"flip_coords": True}),
                baseline=lambda patch, args, kwargs: dc_flip.func(
                    patch, *args, **kwargs
                ),
            ),
            OperationCase(
                case_id="flip-all-no-coords",
                resolve_call=lambda patch: (tuple(patch.dims), {"flip_coords": False}),
                baseline=lambda patch, args, kwargs: dc_flip.func(
                    patch, *args, **kwargs
                ),
            ),
        ),
    ),
    OperationSpec(
        name="roll",
        execution_policy=ExecutionPolicy.LEAF,
        patch_impl=dc.patch_function(history="method_name")(impl.roll_patch),
        patch_op_cls=impl.RollOp,
        leaf_transform=impl.roll_leaves,
        prepare_call=impl.prepare_roll_call,
        validate_compiled_patch=impl.validate_roll_compiled_input,
        test_cases=(
            OperationCase(
                case_id="roll-samples",
                resolve_call=lambda patch: ((), {"samples": True, patch.dims[-1]: 5}),
                baseline=lambda patch, args, kwargs: dc_roll.func(
                    patch, *args, **kwargs
                ),
                compiled_guard=impl.guard_compiled_roll_case,
            ),
            OperationCase(
                case_id="roll-coord-update",
                resolve_call=lambda patch: (
                    (),
                    {"samples": True, "update_coord": True, patch.dims[-1]: 3},
                ),
                baseline=lambda patch, args, kwargs: dc_roll.func(
                    patch, *args, **kwargs
                ),
                compiled_guard=impl.guard_compiled_roll_case,
            ),
        ),
    ),
    OperationSpec(
        name="standardize",
        execution_policy=ExecutionPolicy.PATCH,
        patch_impl=dc.patch_function(history="method_name")(impl.standardize_patch),
        patch_op_cls=impl.StandardizeOp,
        test_cases=(
            OperationCase(
                case_id="standardize-time",
                resolve_call=lambda patch: ((), {"dim": patch.dims[-1]}),
                baseline=lambda patch, args, kwargs: dc_standardize.func(
                    patch, *args, **kwargs
                ),
            ),
        ),
    ),
)
