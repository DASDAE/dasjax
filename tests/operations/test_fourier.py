"""Parity tests for Fourier operations."""

from __future__ import annotations

import dascore as dc
import numpy as np
import pytest

from dasjax import JaxPatchPipeline

from .helpers import assert_compiled_matches_dascore, assert_patch_close, fourier_patch


CASES = (
    ("dft", (), {"dim": "time", "real": True}, lambda p: p.dft(dim="time", real=True)),
    ("idft", (), {}, lambda p: p.idft(), fourier_patch),
    (
        "correlate_shift",
        ("time",),
        {},
        lambda p: p.correlate_shift("time"),
    ),
)


@pytest.mark.parametrize("case", CASES, ids=lambda item: item[0])
def test_compiled_fourier_operation_matches_dascore(case) -> None:
    """Match compiled Fourier operations against DASCore baselines."""
    assert_compiled_matches_dascore(case)


def test_fourier_complex_paths() -> None:
    """Cover complex DFT and IDFT kernel branches."""
    patch = dc.get_example_patch()
    complex_patch = patch.update(data=patch.data + 1j * patch.data)
    transformed = (
        JaxPatchPipeline().dft(dim="time", real=False).compile()(complex_patch)
    )
    expected = complex_patch.dft(dim="time", real=False)
    assert_patch_close(transformed, expected)

    restored = JaxPatchPipeline().idft().compile()(transformed)
    assert_patch_close(restored, expected.idft())

    odd = patch.select(time=(..., patch.get_coord("time").values[-2]))
    padded = JaxPatchPipeline().dft(dim="time", real=True).compile()(odd)
    assert_patch_close(padded, odd.dft(dim="time", real=True))


def test_fourier_preserves_float32_precision() -> None:
    """Do not promote float32 Fourier data through float64 scale factors."""
    patch = dc.get_example_patch()
    patch = patch.update(data=np.asarray(patch.data, dtype=np.float32))

    transformed = JaxPatchPipeline().dft(dim="time", real=True).compile()(patch)
    restored = JaxPatchPipeline().idft().compile()(transformed)

    assert transformed.data.dtype == np.dtype("complex64")
    assert restored.data.dtype == np.dtype("float32")
