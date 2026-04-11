"""Benchmarks for comparing compiled dasjax pipelines against DASCore chains."""

from __future__ import annotations

from collections.abc import Callable

import dascore as dc
import pytest

from dasjax import JaxPatchPipeline


def _scale_patch(patch, factor: float):
    """Return a patch with scaled data."""
    return patch.update(data=patch.data * factor)


def _add_patch(patch, value: float):
    """Return a patch with shifted data."""
    return patch.update(data=patch.data + value)


def _run_dascore_fbe(
    patch,
    *,
    time: int,
    overlap: int,
    samples: bool,
    fmin: float | None = None,
    fmax: float | None = None,
):
    """Run the DASCore baseline used to compare against dasjax fbe."""
    out = patch.stft(time=time, overlap=overlap, samples=samples).abs()
    ft_dim = next(dim for dim in out.dims if dim.startswith("ft_"))
    if fmin is not None or fmax is not None:
        out = out.select(**{ft_dim: (fmin, fmax)})
    return out.sum(dim=ft_dim, dim_reduce="squeeze")


@pytest.fixture(
    scope="module",
    params=[
        ((600, 4000), "float64"),
        ((600, 4000), "float32"),
        ((1200, 8000), "float64"),
        ((1200, 8000), "float32"),
    ],
    ids=["medium-f64", "medium-f32", "large-f64", "large-f32"],
)
def example_patch(request):
    """Return larger patches in both float64 and float32 variants."""
    shape, dtype = request.param
    patch = dc.get_example_patch("random_das", shape=shape)
    if dtype == "float32":
        patch = patch.update(data=patch.data.astype("float32"))
    return patch


@pytest.fixture(scope="module")
def dasjax_scale_add_detrend_normalize(example_patch) -> Callable[[], object]:
    """Return a warmed compiled dasjax arithmetic pipeline."""
    pipeline = (
        JaxPatchPipeline()
        .scale(2.0)
        .add(1.0)
        .detrend(dim="time", type="constant")
        .normalize(dim="time")
    )
    compiled = pipeline.compile()
    compiled(example_patch)
    return lambda: compiled(example_patch)


@pytest.fixture(scope="module")
def dascore_scale_add_detrend_normalize(example_patch) -> Callable[[], object]:
    """Return the equivalent DASCore arithmetic chain."""
    return lambda: (
        _add_patch(_scale_patch(example_patch, 2.0), 1.0)
        .detrend(dim="time", type="constant")
        .normalize(dim="time")
    )


@pytest.fixture(scope="module")
def dasjax_scale_pass_filter_abs(example_patch) -> Callable[[], object]:
    """Return a warmed compiled dasjax filtering pipeline."""
    pipeline = (
        JaxPatchPipeline().scale(2.0).pass_filter(time=(2.0, 10.0)).abs()
    )
    compiled = pipeline.compile()
    compiled(example_patch)
    return lambda: compiled(example_patch)


@pytest.fixture(scope="module")
def dascore_scale_pass_filter_abs(example_patch) -> Callable[[], object]:
    """Return the equivalent DASCore filtering chain."""
    return lambda: _scale_patch(example_patch, 2.0).pass_filter(time=(2.0, 10.0)).abs()


@pytest.fixture(scope="module")
def dasjax_scale_fbe(example_patch) -> Callable[[], object]:
    """Return a warmed compiled dasjax fbe pipeline."""
    pipeline = (
        JaxPatchPipeline()
        .scale(2.0)
        .fbe(time=64, samples=True, overlap=32, fmin=2.0, fmax=10.0)
    )
    compiled = pipeline.compile()
    compiled(example_patch)
    return lambda: compiled(example_patch)


@pytest.fixture(scope="module")
def dascore_scale_fbe(example_patch) -> Callable[[], object]:
    """Return the equivalent DASCore fbe baseline."""
    return lambda: _run_dascore_fbe(
        _scale_patch(example_patch, 2.0),
        time=64,
        overlap=32,
        samples=True,
        fmin=2.0,
        fmax=10.0,
    )


@pytest.fixture(scope="module")
def dasjax_scale_fbe_normalize(example_patch) -> Callable[[], object]:
    """Return a warmed compiled dasjax fbe+normalize pipeline."""
    pipeline = (
        JaxPatchPipeline()
        .scale(2.0)
        .fbe(time=64, samples=True, overlap=32, fmin=2.0, fmax=10.0)
        .normalize(dim="time")
    )
    compiled = pipeline.compile()
    compiled(example_patch)
    return lambda: compiled(example_patch)


@pytest.fixture(scope="module")
def dascore_scale_fbe_normalize(example_patch) -> Callable[[], object]:
    """Return the equivalent DASCore fbe+normalize baseline."""
    return lambda: _run_dascore_fbe(
        _scale_patch(example_patch, 2.0),
        time=64,
        overlap=32,
        samples=True,
        fmin=2.0,
        fmax=10.0,
    ).normalize(dim="time")


class TestCompiledPipelineBenchmarks:
    """Benchmarks for compiled dasjax pipelines and DASCore baselines."""

    @pytest.mark.benchmark(group="scale_add_detrend_normalize")
    def test_dasjax_compiled_scale_add_detrend_normalize(
        self, benchmark, dasjax_scale_add_detrend_normalize
    ) -> None:
        """Benchmark warmed compiled dasjax arithmetic pipeline execution."""
        benchmark(dasjax_scale_add_detrend_normalize)

    @pytest.mark.benchmark(group="scale_add_detrend_normalize")
    def test_dascore_scale_add_detrend_normalize(
        self, benchmark, dascore_scale_add_detrend_normalize
    ) -> None:
        """Benchmark DASCore arithmetic chain execution."""
        benchmark(dascore_scale_add_detrend_normalize)

    @pytest.mark.benchmark(group="scale_pass_filter_abs")
    def test_dasjax_compiled_scale_pass_filter_abs(
        self, benchmark, dasjax_scale_pass_filter_abs
    ) -> None:
        """Benchmark warmed compiled dasjax filtering pipeline execution."""
        benchmark(dasjax_scale_pass_filter_abs)

    @pytest.mark.benchmark(group="scale_pass_filter_abs")
    def test_dascore_scale_pass_filter_abs(
        self, benchmark, dascore_scale_pass_filter_abs
    ) -> None:
        """Benchmark DASCore filtering chain execution."""
        benchmark(dascore_scale_pass_filter_abs)

    @pytest.mark.benchmark(group="scale_fbe")
    def test_dasjax_compiled_scale_fbe(self, benchmark, dasjax_scale_fbe) -> None:
        """Benchmark warmed compiled dasjax fbe pipeline execution."""
        benchmark(dasjax_scale_fbe)

    @pytest.mark.benchmark(group="scale_fbe")
    def test_dascore_scale_fbe(self, benchmark, dascore_scale_fbe) -> None:
        """Benchmark DASCore fbe-equivalent chain execution."""
        benchmark(dascore_scale_fbe)

    @pytest.mark.benchmark(group="scale_fbe_normalize")
    def test_dasjax_compiled_scale_fbe_normalize(
        self, benchmark, dasjax_scale_fbe_normalize
    ) -> None:
        """Benchmark warmed compiled dasjax fbe+normalize pipeline execution."""
        benchmark(dasjax_scale_fbe_normalize)

    @pytest.mark.benchmark(group="scale_fbe_normalize")
    def test_dascore_scale_fbe_normalize(
        self, benchmark, dascore_scale_fbe_normalize
    ) -> None:
        """Benchmark DASCore fbe+normalize chain execution."""
        benchmark(dascore_scale_fbe_normalize)
