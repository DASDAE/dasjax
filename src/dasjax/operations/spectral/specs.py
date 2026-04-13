"""Registry specs for spectral operations."""

from __future__ import annotations

import dascore as dc
from dascore.proc.basic import pad as dc_pad
from dascore.proc.whiten import whiten as dc_whiten
from dascore.transform.fourier import dft as dc_dft
from dascore.transform.fourier import idft as dc_idft
from dascore.transform.hilbert import envelope as dc_envelope
from dascore.transform.hilbert import hilbert as dc_hilbert

from ..common import (
    get_preferred_transform_dim,
    prepare_even_patch_for_dim,
    prepare_fourier_patch,
    resolve_even_dim_call,
)
from ..types import ExecutionPolicy, OperationCase, OperationSpec
from . import impl

OPERATIONS: tuple[OperationSpec, ...] = (
    OperationSpec(
        name="pad",
        execution_policy=ExecutionPolicy.PATCH,
        patch_impl=dc.patch_function(history="method_name")(impl.pad_patch),
        test_cases=(
            OperationCase(
                case_id="pad-constant",
                resolve_call=lambda patch: ((), {"mode": "constant", "samples": True, get_preferred_transform_dim(patch): (2, 3)}),
                prepare_patch=prepare_even_patch_for_dim,
                baseline=lambda patch, args, kwargs: impl.pad_patch(patch, *args, **kwargs),
            ),
            OperationCase(
                case_id="pad-fft",
                resolve_call=lambda patch: ((), {get_preferred_transform_dim(patch): "fft"}),
                baseline=lambda patch, args, kwargs: dc_pad.func(patch, *args, **kwargs),
            ),
        ),
    ),
    OperationSpec(
        name="dft",
        execution_policy=ExecutionPolicy.PATCH,
        patch_impl=dc.patch_function(history="method_name")(impl.dft_patch),
        test_cases=(
            OperationCase(
                case_id="dft-real",
                resolve_call=lambda patch: ((), {"dim": patch.dims[-1], "real": True}),
                baseline=lambda patch, args, kwargs: dc_dft.func(patch, *args, **kwargs),
            ),
            OperationCase(
                case_id="dft-full",
                resolve_call=lambda patch: ((), {"dim": patch.dims[-1], "real": None, "pad": False}),
                baseline=lambda patch, args, kwargs: dc_dft.func(patch, *args, **kwargs),
            ),
        ),
    ),
    OperationSpec(
        name="idft",
        execution_policy=ExecutionPolicy.PATCH,
        patch_impl=dc.patch_function(history="method_name")(impl.idft_patch),
        test_cases=(
            OperationCase(
                case_id="idft-real",
                resolve_call=lambda patch: ((), {}),
                prepare_patch=prepare_fourier_patch,
                baseline=lambda patch, args, kwargs: dc_idft.func(patch, *args, **kwargs),
            ),
        ),
    ),
    OperationSpec(
        name="hilbert",
        execution_policy=ExecutionPolicy.PATCH,
        patch_impl=dc.patch_function(history="method_name")(impl.hilbert_patch),
        test_cases=(
            OperationCase(
                case_id="hilbert-time",
                resolve_call=lambda patch: resolve_even_dim_call(patch),
                baseline=lambda patch, args, kwargs: dc_hilbert.func(patch, *args, **kwargs),
            ),
        ),
    ),
    OperationSpec(
        name="envelope",
        execution_policy=ExecutionPolicy.PATCH,
        patch_impl=dc.patch_function(history="method_name")(impl.envelope_patch),
        test_cases=(
            OperationCase(
                case_id="envelope-time",
                resolve_call=lambda patch: resolve_even_dim_call(patch),
                baseline=lambda patch, args, kwargs: dc_envelope.func(patch, *args, **kwargs),
            ),
        ),
    ),
    OperationSpec(
        name="whiten",
        execution_policy=ExecutionPolicy.PATCH,
        patch_impl=dc.patch_function(history="method_name")(impl.whiten_patch),
        test_cases=(
            OperationCase(
                case_id="whiten-default",
                resolve_call=lambda patch: ((), {get_preferred_transform_dim(patch): None}),
                baseline=lambda patch, args, kwargs: dc_whiten.func(patch, *args, **kwargs),
            ),
            OperationCase(
                case_id="whiten-band-limited",
                resolve_call=lambda patch: ((), {"smooth_size": 3.0, get_preferred_transform_dim(patch): (5.0, 10.0, 20.0, 25.0)}),
                baseline=lambda patch, args, kwargs: dc_whiten.func(patch, *args, **kwargs),
            ),
        ),
    ),
    OperationSpec(
        name="fbe",
        execution_policy=ExecutionPolicy.PATCH,
        patch_impl=dc.patch_function(history="method_name")(impl.fbe_patch),
        prepare_call=impl.prepare_fbe_call,
        test_cases=(
            OperationCase(
                case_id="fbe-default",
                resolve_call=lambda patch: ((), {patch.dims[-1]: 64, "samples": True, "overlap": 32, "fmin": 2.0, "fmax": 10.0}),
                prepare_patch=prepare_even_patch_for_dim,
                baseline=impl.baseline_fbe_patch,
            ),
        ),
    ),
)
