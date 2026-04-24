"""Parity tests for detrend operations."""

from __future__ import annotations

import dascore as dc
import numpy as np
import pytest

from dasjax import JaxPatchPipeline
from dasjax.kernels import detrend_kernel, validate_detrend_type

from .helpers import assert_compiled_matches_dascore, assert_patch_close


CASES = (
    (
        "detrend",
        (),
        {"dim": "time", "type": "constant"},
        lambda p: p.detrend(dim="time", type="constant"),
    ),
)


@pytest.mark.parametrize("case", CASES, ids=lambda item: item[0])
def test_compiled_detrend_operation_matches_dascore(case) -> None:
    """Match compiled detrend operations against DASCore baselines."""
    assert_compiled_matches_dascore(case)


@pytest.mark.parametrize("detrend_type", ["linear", "l", "c"])
def test_detrend_aliases_match_dascore(detrend_type: str) -> None:
    """Cover detrend aliases and the linear kernel path."""
    patch = dc.get_example_patch()
    pipeline = JaxPatchPipeline().detrend(dim="time", type=detrend_type)

    assert_patch_close(
        pipeline.compile()(patch),
        patch.detrend(dim="time", type=detrend_type),
    )


def test_detrend_rejects_invalid_type() -> None:
    """Cover invalid detrend validation."""
    with pytest.raises(ValueError, match="Trend type"):
        validate_detrend_type("curve")
    with pytest.raises(ValueError, match="Trend type"):
        detrend_kernel(np.arange(5.0), axis=0, type="curve")
