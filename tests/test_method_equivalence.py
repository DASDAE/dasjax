"""Compare core pipeline operations against DASCore baselines."""

from __future__ import annotations

import dascore as dc
import numpy as np
import pytest

from dasjax import JaxPatchPipeline, list_patch_operations


def _assert_patch_close(left, right) -> None:
    assert np.allclose(left.data, right.data, equal_nan=True, rtol=1e-5, atol=1e-6)
    assert left.coords == right.coords


def _complex_patch(patch):
    return patch.update(data=patch.data + 1j * patch.data)


def _even_patch(patch):
    return patch.select(time=(..., patch.get_coord("time").values[-2]))


def _fourier_patch(patch):
    return patch.dft("time", real=True)


def _fbe_baseline(patch):
    out = patch.stft(time=64, samples=True, overlap=32).abs()
    ft_dim = next(dim for dim in out.dims if dim.startswith("ft_"))
    out = out.select(**{ft_dim: (2.0, 10.0)})
    return out.sum(dim=ft_dim, dim_reduce="squeeze")


CASES = (
    ("identity", (), {}, lambda p: p),
    ("scale", (2.0,), {}, lambda p: p * 2.0),
    ("add", (2.0,), {}, lambda p: p + 2.0),
    ("abs", (), {}, lambda p: p.abs(), _complex_patch),
    ("clip", (-0.25, 0.25), {}, lambda p: p.update(data=np.clip(p.data, -0.25, 0.25))),
    ("real", (), {}, lambda p: p.real(), _complex_patch),
    ("imag", (), {}, lambda p: p.imag(), _complex_patch),
    ("angle", (), {}, lambda p: p.angle(), _complex_patch),
    ("conj", (), {}, lambda p: p.conj(), _complex_patch),
    ("flip", ("time",), {}, lambda p: p.flip("time")),
    ("roll", (), {"time": 5, "samples": True}, lambda p: p.roll(time=5, samples=True)),
    ("standardize", (), {"dim": "time"}, lambda p: p.standardize(dim="time")),
    ("detrend", (), {"dim": "time", "type": "constant"}, lambda p: p.detrend(dim="time", type="constant")),
    ("normalize", (), {"dim": "time", "norm": "l2"}, lambda p: p.normalize(dim="time", norm="l2")),
    ("differentiate", (), {"dim": "time"}, lambda p: p.differentiate(dim="time")),
    ("integrate", (), {"dim": "time"}, lambda p: p.integrate(dim="time")),
    ("taper", (), {"time": 0.05, "window_type": "hann"}, lambda p: p.taper(time=0.05, window_type="hann")),
    (
        "taper_range",
        (),
        {"time": (10, 20), "samples": True, "window_type": "hann"},
        lambda p: p.taper_range(time=(10, 20), samples=True, window_type="hann"),
    ),
    ("gaussian_filter", (), {"time": 3, "samples": True}, lambda p: p.gaussian_filter(time=3, samples=True)),
    (
        "hampel_filter",
        (),
        {"time": 3, "samples": True, "threshold": 3.5, "approximate": True},
        lambda p: p.hampel_filter(time=3, samples=True, threshold=3.5, approximate=True),
    ),
    (
        "pass_filter",
        (),
        {"time": (1.0, 10.0), "corners": 4, "zerophase": True},
        lambda p: p.pass_filter(time=(1.0, 10.0), corners=4, zerophase=True),
        _even_patch,
    ),
    ("pad", (), {"time": (2, 3), "samples": True}, lambda p: p.pad(time=(2, 3), samples=True)),
    ("dft", (), {"dim": "time", "real": True}, lambda p: p.dft(dim="time", real=True)),
    ("idft", (), {}, lambda p: p.idft(), _fourier_patch),
    ("hilbert", (), {"dim": "time"}, lambda p: p.hilbert(dim="time"), _even_patch),
    ("envelope", (), {"dim": "time"}, lambda p: p.envelope(dim="time"), _even_patch),
    ("whiten", (), {"time": None}, lambda p: p.whiten(time=None), _even_patch),
    (
        "fbe",
        (),
        {"time": 64, "samples": True, "overlap": 32, "fmin": 2.0, "fmax": 10.0},
        _fbe_baseline,
        lambda p: dc.get_example_patch("chirp"),
    ),
)


def test_case_table_covers_registered_operations() -> None:
    assert {case[0] for case in CASES} == set(list_patch_operations())


@pytest.mark.parametrize("case", CASES, ids=lambda item: item[0])
def test_compiled_pipeline_matches_dascore(case) -> None:
    name, args, kwargs, baseline, *rest = case
    example_patch = dc.get_example_patch()
    patch = rest[0](example_patch) if rest else example_patch
    pipeline = getattr(JaxPatchPipeline(), name)(*args, **kwargs)

    out = pipeline.compile()(patch)
    expected = baseline(patch)

    _assert_patch_close(out, expected)
