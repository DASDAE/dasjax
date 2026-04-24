"""Parity tests for filter operations."""

from __future__ import annotations

import dascore as dc
import jax.numpy as jnp
import numpy as np
import pytest
from dascore.exceptions import FilterValueError, ParameterError
from scipy.signal import wiener

from dasjax import JaxPatchPipeline
from dasjax.kernels import (
    design_notch_filter,
    design_pass_filter_sos,
    gaussian_filter_kernel,
    hampel_filter_kernel,
    is_finite_array,
    median_filter_kernel,
    notch_filter_kernel,
    pass_filter_default_padlen,
    pass_filter_initial_state,
    pass_filter_kernel,
    savgol_coefficients,
    savgol_filter_kernel,
    sobel_filter_kernel,
    wiener_filter_kernel,
)
from dasjax.kernels.filters import _iirfilt_rows

from .helpers import assert_compiled_matches_dascore, even_patch


CASES = (
    (
        "gaussian_filter",
        (),
        {"time": 3, "samples": True},
        lambda p: p.gaussian_filter(time=3, samples=True),
    ),
    (
        "hampel_filter",
        (),
        {"time": 3, "samples": True, "threshold": 3.5, "approximate": True},
        lambda p: p.hampel_filter(
            time=3, samples=True, threshold=3.5, approximate=True
        ),
    ),
    (
        "median_filter",
        (),
        {"time": 3, "samples": True},
        lambda p: p.median_filter(time=3, samples=True),
    ),
    (
        "notch_filter",
        (),
        {"time": 10.0, "q": 30.0},
        lambda p: p.notch_filter(time=10.0, q=30.0),
        even_patch,
    ),
    (
        "pass_filter",
        (),
        {"time": (1.0, 10.0), "corners": 4, "zerophase": True},
        lambda p: p.pass_filter(time=(1.0, 10.0), corners=4, zerophase=True),
        even_patch,
    ),
    (
        "savgol_filter",
        (2,),
        {"time": 5, "samples": True},
        lambda p: p.savgol_filter(2, time=5, samples=True),
    ),
    (
        "sobel_filter",
        ("time",),
        {},
        lambda p: p.sobel_filter("time"),
    ),
)


@pytest.mark.parametrize("case", CASES, ids=lambda item: item[0])
def test_compiled_filter_operation_matches_dascore(case) -> None:
    """Match compiled filter operations against DASCore baselines."""
    assert_compiled_matches_dascore(case)


def test_filter_kernel_and_design_edges() -> None:
    """Cover pass-filter validation and lower-level kernel branches."""
    data = np.sin(np.linspace(0, 1, 20))[None, :]
    sos = design_pass_filter_sos(100.0, None, 10.0, 2)

    assert is_finite_array(data)
    assert not is_finite_array([1.0, np.nan])
    assert design_pass_filter_sos(100.0, 10.0, None, 2).shape[1] == 6
    assert design_pass_filter_sos(100.0, 5.0, 20.0, 2).shape[1] == 6
    for args in (
        (100.0, -1.0, None, 2),
        (100.0, None, 60.0, 2),
        (100.0, 20.0, 10.0, 2),
        (100.0, None, None, 2),
    ):
        with pytest.raises(FilterValueError):
            design_pass_filter_sos(*args)

    zi = pass_filter_initial_state(sos)
    assert pass_filter_default_padlen(sos) > 0
    assert np.asarray(
        pass_filter_kernel(data, sos, axis=1, zi=zi, zerophase=False)
    ).shape == data.shape
    assert np.asarray(
        pass_filter_kernel(data, sos, axis=1, zi=zi, padlen=0)
    ).shape == data.shape
    assert np.asarray(
        pass_filter_kernel(data, sos, axis=1, zerophase=False)
    ).shape == data.shape
    with pytest.raises(ValueError, match="requires zi"):
        pass_filter_kernel(data, jnp.asarray(sos), axis=1, zi=None)
    with pytest.raises(ValueError, match="greater than padlen"):
        pass_filter_kernel(data[:, :2], sos, axis=1, zi=zi, padlen=3)

    patch = even_patch(dc.get_example_patch())
    with pytest.raises(ParameterError, match="greater than zero"):
        JaxPatchPipeline().hampel_filter(time=3, samples=True, threshold=0).compile()(
            patch
        )


def test_jax_filter_kernel_edge_modes_and_validation() -> None:
    """Cover JAX-native filter modes and validation paths."""
    data = np.arange(12.0).reshape(3, 4)

    assert np.asarray(gaussian_filter_kernel(data, sigma=(), axes=())).shape == data.shape
    assert np.asarray(
        gaussian_filter_kernel(data, sigma=(-1.0,), axes=(1,), truncate=4.0)
    ).shape == data.shape
    assert np.asarray(
        gaussian_filter_kernel(data, sigma=(0.1,), axes=(1,), truncate=0.1)
    ).shape == data.shape
    for mode in ("nearest", "constant", "wrap", "mirror"):
        assert np.asarray(
            gaussian_filter_kernel(data, sigma=(1.0,), axes=(1,), mode=mode)
        ).shape == data.shape
    with pytest.raises(ValueError, match="Unsupported gaussian_filter mode"):
        gaussian_filter_kernel(data, sigma=(1.0,), axes=(1,), mode="bad")

    assert np.asarray(median_filter_kernel(data, size=(1, 1))).shape == data.shape
    for mode in ("grid-constant", "grid-wrap", "mirror"):
        assert np.asarray(median_filter_kernel(data, size=(1, 3), mode=mode)).shape == (
            data.shape
        )
    with pytest.raises(ValueError, match="Unsupported filter mode"):
        median_filter_kernel(data, size=(1, 3), mode="bad")

    assert np.allclose(
        np.asarray(wiener_filter_kernel(data, size=(1, 3), noise=0.1)),
        wiener(data, mysize=(1, 3), noise=0.1),
        equal_nan=True,
    )
    assert np.asarray(sobel_filter_kernel(data, axis=1, mode="nearest")).shape == (
        data.shape
    )


def test_savgol_and_notch_kernel_edges() -> None:
    """Cover Savitzky-Golay and notch kernel edge branches."""
    data = np.arange(12.0).reshape(3, 4)
    coeff, left, right = savgol_coefficients(3, 1)

    assert np.asarray(
        savgol_filter_kernel(
            data,
            size=(1, 3),
            axes=(1,),
            coeffs=(coeff,),
            left_coeffs=(left,),
            right_coeffs=(right,),
            mode="nearest",
        )
    ).shape == data.shape
    with pytest.raises(ValueError, match="window_length"):
        savgol_filter_kernel(
            data[:, :2],
            size=(1, 3),
            axes=(1,),
            coeffs=(coeff,),
            left_coeffs=(left,),
            right_coeffs=(right,),
        )

    with pytest.raises(FilterValueError):
        design_notch_filter(100.0, 60.0, 30.0)
    b, a, zi, padlen = design_notch_filter(100.0, 10.0, 30.0)
    assert np.asarray(notch_filter_kernel(data, b, a, zi, padlen=0, axis=1)).shape == (
        data.shape
    )
    with pytest.raises(ValueError, match="greater than padlen"):
        notch_filter_kernel(data[:, :2], b, a, zi, padlen=padlen, axis=1)

    filtered, states = _iirfilt_rows(
        jnp.asarray(data[:1]),
        jnp.asarray(b),
        jnp.asarray(a),
    )
    assert filtered.shape == data[:1].shape
    assert states.shape == (1, len(b) - 1)


def test_hampel_integer_and_exact_rejection_paths() -> None:
    """Cover Hampel integer round-trip and exact-mode rejection."""
    data = np.asarray([[1, 100, 1, 1, 1]], dtype=np.int32)

    native = np.asarray(hampel_filter_kernel(data, size=(1, 3), threshold=1.0))
    approx = np.asarray(
        hampel_filter_kernel(
            data.astype(float), size=(1, 3), threshold=1.0, approximate=True
        )
    )

    assert native.dtype == data.dtype
    assert approx.shape == data.shape
    with pytest.raises(NotImplementedError, match="pure JAX"):
        hampel_filter_kernel(
            data.astype(float), size=(1, 3), threshold=1.0, approximate=False
        )
    with pytest.raises(NotImplementedError, match="pure JAX"):
        JaxPatchPipeline().hampel_filter(
            time=3, samples=True, approximate=False
        ).compile()


def test_filter_operation_validation_edges() -> None:
    """Cover operation-level validation branches."""
    patch = even_patch(dc.get_example_patch())

    with pytest.raises(FilterValueError, match="dim parameter"):
        JaxPatchPipeline().sobel_filter(1).compile()(patch)
    with pytest.raises(ParameterError, match="wiener_filter"):
        JaxPatchPipeline().wiener_filter().compile()(patch)
    with pytest.raises(ParameterError, match="sorted length 4"):
        JaxPatchPipeline().slope_filter([1.0, 3.0, 2.0, 4.0]).compile()(patch)

    out = JaxPatchPipeline().notch_filter(time=10.0 * dc.get_unit("Hz"), q=30.0)
    assert out.compile()(patch).shape == patch.shape
