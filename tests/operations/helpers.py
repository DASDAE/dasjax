"""Shared helpers for operation parity tests."""

from __future__ import annotations

import dascore as dc
import numpy as np

from dasjax import JaxPatchPipeline


def assert_patch_close(left, right) -> None:
    """Assert data closeness and exact coordinate preservation."""
    assert np.allclose(left.data, right.data, equal_nan=True, rtol=1e-5, atol=1e-6)
    assert left.coords == right.coords


def complex_patch(patch):
    """Return a patch with complex-valued data."""
    return patch.update(data=patch.data + 1j * patch.data)


def even_patch(patch):
    """Return a patch with an even number of time samples."""
    return patch.select(time=(..., patch.get_coord("time").values[-2]))


def fourier_patch(patch):
    """Return a patch transformed along time."""
    return patch.dft("time", real=True)


def fbe_baseline(patch):
    """Return the DASCore baseline for frequency-band energy."""
    out = patch.stft(time=64, samples=True, overlap=32).abs()
    ft_dim = next(dim for dim in out.dims if dim.startswith("ft_"))
    out = out.select(**{ft_dim: (2.0, 10.0)})
    return out.sum(dim=ft_dim, dim_reduce="squeeze")


def assert_compiled_matches_dascore(case) -> None:
    """Compile one pipeline case and compare it to the DASCore baseline."""
    name, args, kwargs, baseline, *rest = case
    example_patch = dc.get_example_patch()
    patch = rest[0](example_patch) if rest else example_patch
    pipeline = getattr(JaxPatchPipeline(), name)(*args, **kwargs)

    out = pipeline.compile()(patch)
    expected = baseline(patch)

    assert_patch_close(out, expected)
