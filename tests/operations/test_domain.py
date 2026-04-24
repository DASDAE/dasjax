"""Parity tests for domain transform operations."""

from __future__ import annotations

import numpy as np
import pytest
import dascore as dc
from dascore.exceptions import ParameterError

from dasjax import JaxPatchPipeline

from .helpers import assert_patch_close


def small_patch():
    """Return a compact patch for domain transform parity tests."""
    patch = dc.get_example_patch()
    time = patch.get_coord("time").values
    return patch.select(distance=(0, 20), time=(time[0], time[200]))


CASES = (
    (
        "line_mute",
        (),
        {"time": (0, 0.1)},
        lambda p: p.line_mute(time=(0, 0.1)),
        small_patch,
    ),
    (
        "slope_mute",
        ((1000.0, 3000.0),),
        {},
        lambda p: p.slope_mute((1000.0, 3000.0)),
        small_patch,
    ),
    (
        "wiener_filter",
        (),
        {"time": 5, "samples": True},
        lambda p: p.wiener_filter(time=5, samples=True),
        small_patch,
    ),
    (
        "phase_weighted_stack",
        ("distance",),
        {"transform_dim": "time"},
        lambda p: p.phase_weighted_stack("distance", transform_dim="time"),
        small_patch,
    ),
    (
        "slope_filter",
        ([100.0, 200.0, 500.0, 1000.0],),
        {},
        lambda p: p.slope_filter([100.0, 200.0, 500.0, 1000.0]),
        small_patch,
    ),
    (
        "tau_p",
        (np.asarray([1000.0, 2000.0]),),
        {},
        lambda p: p.tau_p(np.asarray([1000.0, 2000.0])),
        lambda: small_patch().set_units(distance="m", time="s"),
    ),
    (
        "dispersion_phase_shift",
        (np.asarray([100.0, 200.0]),),
        {"approx_freq": (1.0, 10.0)},
        lambda p: p.dispersion_phase_shift(
            np.asarray([100.0, 200.0]), approx_freq=(1.0, 10.0)
        ),
        lambda: small_patch().set_units(distance="m", time="s"),
    ),
)


@pytest.mark.parametrize("case", CASES, ids=lambda item: item[0])
def test_compiled_domain_operation_matches_dascore(case) -> None:
    """Match compiled domain operations against DASCore baselines."""
    name, args, kwargs, baseline, patch_factory = case
    patch = patch_factory()
    pipeline = getattr(JaxPatchPipeline(), name)(*args, **kwargs)

    assert_patch_close(pipeline.compile()(patch), baseline(patch))


def test_compiled_strain_operations_match_dascore() -> None:
    """Match compiled strain transforms against DASCore baselines."""
    patch = dc.get_example_patch("deformation_rate_event_1")

    cases = (
        (
            JaxPatchPipeline().velocity_to_strain_rate(),
            patch.velocity_to_strain_rate(),
            patch,
        ),
        (
            JaxPatchPipeline().velocity_to_strain_rate_edgeless(step_multiple=2),
            patch.velocity_to_strain_rate_edgeless(step_multiple=2),
            patch,
        ),
    )
    for pipeline, expected, source in cases:
        assert_patch_close(pipeline.compile()(source), expected)

    radians_patch = patch.update(
        data=patch.data,
        attrs=patch.attrs.update(data_units="rad", gauge_length=10.0),
    )
    assert_patch_close(
        JaxPatchPipeline().radians_to_strain().compile()(radians_patch),
        radians_patch.radians_to_strain(),
    )

    non_radian_patch = patch.update(
        data=patch.data,
        attrs=patch.attrs.update(data_units="m/s", gauge_length=10.0),
    )
    assert_patch_close(
        JaxPatchPipeline().radians_to_strain().compile()(non_radian_patch),
        non_radian_patch,
    )


def test_domain_operation_validation_edges() -> None:
    """Cover validation paths for domain transforms."""
    patch = small_patch().set_units(distance="m", time="s")

    no_freq_out = JaxPatchPipeline().dispersion_phase_shift(
        np.asarray([100.0, 200.0])
    ).compile()(patch)
    assert no_freq_out.shape == patch.dispersion_phase_shift(
        np.asarray([100.0, 200.0])
    ).shape

    for velocities, message in (
        (np.asarray([200.0, 100.0]), "monotonically increasing"),
        (np.asarray([-100.0, 200.0]), "positive"),
    ):
        with pytest.raises(ParameterError, match=message):
            JaxPatchPipeline().dispersion_phase_shift(velocities).compile()(patch)

    for kwargs, message in (
        ({"approx_resolution": 0.0}, "Frequency resolution"),
        ({"approx_freq": (0.0, 10.0)}, "Minimal and maximal"),
        ({"approx_freq": (10.0, 1.0)}, "Maximal frequency"),
        ({"approx_freq": (1.0, 1000.0)}, "Nyquist"),
        (
            {"approx_resolution": 100.0, "approx_freq": (1.0, 2.0)},
            "not an array",
        ),
    ):
        with pytest.raises(ParameterError, match=message):
            JaxPatchPipeline().dispersion_phase_shift(
                np.asarray([100.0, 200.0]), **kwargs
            ).compile()(patch)

    for velocities, message in (
        (np.asarray([0.0, 1000.0]), "positive"),
        (np.asarray([2000.0, 1000.0]), "monotonically increasing"),
    ):
        with pytest.raises(ParameterError, match=message):
            JaxPatchPipeline().tau_p(velocities).compile()(patch)


def test_mute_and_strain_validation_edges() -> None:
    """Cover mute and strain transform edge paths."""
    patch = small_patch().set_units(distance="m", time="s")
    strain_patch = dc.get_example_patch("deformation_rate_event_1")

    assert_patch_close(
        JaxPatchPipeline().slope_mute((0.0, np.inf)).compile()(patch),
        patch.slope_mute((0.0, np.inf)),
    )
    for slopes, message in (
        ((1.0,), "length 2"),
        ((-1.0, 1.0), "positive"),
    ):
        with pytest.raises(ParameterError, match=message):
            JaxPatchPipeline().slope_mute(slopes).compile()(patch)

    with pytest.raises(ParameterError, match="positive"):
        JaxPatchPipeline().velocity_to_strain_rate(step_multiple=0).compile()(
            strain_patch
        )
    with pytest.raises(ParameterError, match="even"):
        JaxPatchPipeline().velocity_to_strain_rate(step_multiple=3).compile()(
            strain_patch
        )
    with pytest.raises(ParameterError, match="positive"):
        JaxPatchPipeline().velocity_to_strain_rate_edgeless(step_multiple=0).compile()(
            strain_patch
        )
    with pytest.raises(ParameterError, match="Gauge length"):
        JaxPatchPipeline().radians_to_strain().compile()(strain_patch)
