"""Benchmarks for comparing compiled dasjax pipelines against DASCore chains."""

from __future__ import annotations

from collections.abc import Callable

import dascore as dc
import numpy as np
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
def dasjax_scale(example_patch) -> Callable[[], object]:
    """Return a warmed compiled dasjax scale operation."""
    compiled = JaxPatchPipeline().scale(2.0).compile()
    compiled(example_patch)
    return lambda: compiled(example_patch)


@pytest.fixture(scope="module")
def dascore_scale(example_patch) -> Callable[[], object]:
    """Return the equivalent DASCore scale operation."""
    return lambda: _scale_patch(example_patch, 2.0)


@pytest.fixture(scope="module")
def dasjax_add(example_patch) -> Callable[[], object]:
    """Return a warmed compiled dasjax add operation."""
    compiled = JaxPatchPipeline().add(1.0).compile()
    compiled(example_patch)
    return lambda: compiled(example_patch)


@pytest.fixture(scope="module")
def dascore_add(example_patch) -> Callable[[], object]:
    """Return the equivalent DASCore add operation."""
    return lambda: _add_patch(example_patch, 1.0)


@pytest.fixture(scope="module")
def dasjax_abs(example_patch) -> Callable[[], object]:
    """Return a warmed compiled dasjax abs operation."""
    compiled = JaxPatchPipeline().abs().compile()
    compiled(example_patch)
    return lambda: compiled(example_patch)


@pytest.fixture(scope="module")
def dascore_abs(example_patch) -> Callable[[], object]:
    """Return the equivalent DASCore abs operation."""
    return lambda: example_patch.abs()


@pytest.fixture(scope="module")
def dasjax_clip(example_patch) -> Callable[[], object]:
    """Return a warmed compiled dasjax clip operation."""
    compiled = JaxPatchPipeline().clip(-0.25, 0.25).compile()
    compiled(example_patch)
    return lambda: compiled(example_patch)


@pytest.fixture(scope="module")
def dascore_clip(example_patch) -> Callable[[], object]:
    """Return the equivalent DASCore clip operation."""
    return lambda: example_patch.update(data=np.clip(example_patch.data, -0.25, 0.25))


@pytest.fixture(scope="module")
def dasjax_detrend(example_patch) -> Callable[[], object]:
    """Return a warmed compiled dasjax detrend operation."""
    compiled = JaxPatchPipeline().detrend(dim="time", type="constant").compile()
    compiled(example_patch)
    return lambda: compiled(example_patch)


@pytest.fixture(scope="module")
def dascore_detrend(example_patch) -> Callable[[], object]:
    """Return the equivalent DASCore detrend operation."""
    return lambda: example_patch.detrend(dim="time", type="constant")


@pytest.fixture(scope="module")
def dasjax_normalize(example_patch) -> Callable[[], object]:
    """Return a warmed compiled dasjax normalize operation."""
    compiled = JaxPatchPipeline().normalize(dim="time").compile()
    compiled(example_patch)
    return lambda: compiled(example_patch)


@pytest.fixture(scope="module")
def dascore_normalize(example_patch) -> Callable[[], object]:
    """Return the equivalent DASCore normalize operation."""
    return lambda: example_patch.normalize(dim="time")


@pytest.fixture(scope="module")
def dasjax_standardize(example_patch) -> Callable[[], object]:
    """Return a warmed compiled dasjax standardize operation."""
    compiled = JaxPatchPipeline().standardize(dim="time").compile()
    compiled(example_patch)
    return lambda: compiled(example_patch)


@pytest.fixture(scope="module")
def dascore_standardize(example_patch) -> Callable[[], object]:
    """Return the equivalent DASCore standardize operation."""
    return lambda: example_patch.standardize(dim="time")


@pytest.fixture(scope="module")
def dasjax_differentiate(example_patch) -> Callable[[], object]:
    """Return a warmed compiled dasjax differentiate operation."""
    compiled = JaxPatchPipeline().differentiate(dim="time").compile()
    compiled(example_patch)
    return lambda: compiled(example_patch)


@pytest.fixture(scope="module")
def dascore_differentiate(example_patch) -> Callable[[], object]:
    """Return the equivalent DASCore differentiate operation."""
    return lambda: example_patch.differentiate(dim="time")


@pytest.fixture(scope="module")
def dasjax_integrate(example_patch) -> Callable[[], object]:
    """Return a warmed compiled dasjax integrate operation."""
    compiled = JaxPatchPipeline().integrate(dim="time").compile()
    compiled(example_patch)
    return lambda: compiled(example_patch)


@pytest.fixture(scope="module")
def dascore_integrate(example_patch) -> Callable[[], object]:
    """Return the equivalent DASCore integrate operation."""
    return lambda: example_patch.integrate(dim="time")


@pytest.fixture(scope="module")
def dasjax_taper(example_patch) -> Callable[[], object]:
    """Return a warmed compiled dasjax taper operation."""
    compiled = JaxPatchPipeline().taper(time=0.05, window_type="hann").compile()
    compiled(example_patch)
    return lambda: compiled(example_patch)


@pytest.fixture(scope="module")
def dascore_taper(example_patch) -> Callable[[], object]:
    """Return the equivalent DASCore taper operation."""
    return lambda: example_patch.taper(time=0.05, window_type="hann")


@pytest.fixture(scope="module")
def dasjax_pad(example_patch) -> Callable[[], object]:
    """Return a warmed compiled dasjax pad operation."""
    compiled = JaxPatchPipeline().pad(time=(16, 16), samples=True).compile()
    compiled(example_patch)
    return lambda: compiled(example_patch)


@pytest.fixture(scope="module")
def dascore_pad(example_patch) -> Callable[[], object]:
    """Return the equivalent DASCore pad operation."""
    return lambda: example_patch.pad(time=(16, 16), samples=True)


@pytest.fixture(scope="module")
def dasjax_mean(example_patch) -> Callable[[], object]:
    """Return a warmed compiled dasjax mean reduction."""
    compiled = JaxPatchPipeline().mean(dim="time").compile()
    compiled(example_patch)
    return lambda: compiled(example_patch)


@pytest.fixture(scope="module")
def dascore_mean(example_patch) -> Callable[[], object]:
    """Return the equivalent DASCore mean reduction."""
    return lambda: example_patch.mean(dim="time")


@pytest.fixture(scope="module")
def dasjax_pass_filter(example_patch) -> Callable[[], object]:
    """Return a warmed compiled dasjax pass_filter operation."""
    compiled = JaxPatchPipeline().pass_filter(time=(2.0, 10.0)).compile()
    compiled(example_patch)
    return lambda: compiled(example_patch)


@pytest.fixture(scope="module")
def dascore_pass_filter(example_patch) -> Callable[[], object]:
    """Return the equivalent DASCore pass_filter operation."""
    return lambda: example_patch.pass_filter(time=(2.0, 10.0))


@pytest.fixture(scope="module")
def dasjax_fbe(example_patch) -> Callable[[], object]:
    """Return a warmed compiled dasjax fbe operation."""
    compiled = (
        JaxPatchPipeline()
        .fbe(time=64, samples=True, overlap=32, fmin=2.0, fmax=10.0)
        .compile()
    )
    compiled(example_patch)
    return lambda: compiled(example_patch)


@pytest.fixture(scope="module")
def dascore_fbe(example_patch) -> Callable[[], object]:
    """Return the equivalent DASCore fbe baseline operation."""
    return lambda: _run_dascore_fbe(
        example_patch,
        time=64,
        overlap=32,
        samples=True,
        fmin=2.0,
        fmax=10.0,
    )


@pytest.fixture(scope="module")
def dasjax_scale_pass_filter_abs(example_patch) -> Callable[[], object]:
    """Return a warmed compiled dasjax filtering pipeline."""
    pipeline = JaxPatchPipeline().scale(2.0).pass_filter(time=(2.0, 10.0)).abs()
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


class TestIndividualOperationBenchmarks:
    """Benchmarks for individual dasjax operations and DASCore baselines."""

    @pytest.mark.benchmark(group="operation_scale")
    def test_dasjax_compiled_operation_scale(self, benchmark, dasjax_scale) -> None:
        """Benchmark warmed compiled dasjax scale execution."""
        benchmark(dasjax_scale)

    @pytest.mark.benchmark(group="operation_scale")
    def test_dascore_operation_scale(self, benchmark, dascore_scale) -> None:
        """Benchmark DASCore scale execution."""
        benchmark(dascore_scale)

    @pytest.mark.benchmark(group="operation_add")
    def test_dasjax_compiled_operation_add(self, benchmark, dasjax_add) -> None:
        """Benchmark warmed compiled dasjax add execution."""
        benchmark(dasjax_add)

    @pytest.mark.benchmark(group="operation_add")
    def test_dascore_operation_add(self, benchmark, dascore_add) -> None:
        """Benchmark DASCore add execution."""
        benchmark(dascore_add)

    @pytest.mark.benchmark(group="operation_abs")
    def test_dasjax_compiled_operation_abs(self, benchmark, dasjax_abs) -> None:
        """Benchmark warmed compiled dasjax abs execution."""
        benchmark(dasjax_abs)

    @pytest.mark.benchmark(group="operation_abs")
    def test_dascore_operation_abs(self, benchmark, dascore_abs) -> None:
        """Benchmark DASCore abs execution."""
        benchmark(dascore_abs)

    @pytest.mark.benchmark(group="operation_clip")
    def test_dasjax_compiled_operation_clip(self, benchmark, dasjax_clip) -> None:
        """Benchmark warmed compiled dasjax clip execution."""
        benchmark(dasjax_clip)

    @pytest.mark.benchmark(group="operation_clip")
    def test_dascore_operation_clip(self, benchmark, dascore_clip) -> None:
        """Benchmark DASCore-equivalent clip execution."""
        benchmark(dascore_clip)

    @pytest.mark.benchmark(group="operation_detrend")
    def test_dasjax_compiled_operation_detrend(self, benchmark, dasjax_detrend) -> None:
        """Benchmark warmed compiled dasjax detrend execution."""
        benchmark(dasjax_detrend)

    @pytest.mark.benchmark(group="operation_detrend")
    def test_dascore_operation_detrend(self, benchmark, dascore_detrend) -> None:
        """Benchmark DASCore detrend execution."""
        benchmark(dascore_detrend)

    @pytest.mark.benchmark(group="operation_normalize")
    def test_dasjax_compiled_operation_normalize(
        self, benchmark, dasjax_normalize
    ) -> None:
        """Benchmark warmed compiled dasjax normalize execution."""
        benchmark(dasjax_normalize)

    @pytest.mark.benchmark(group="operation_normalize")
    def test_dascore_operation_normalize(self, benchmark, dascore_normalize) -> None:
        """Benchmark DASCore normalize execution."""
        benchmark(dascore_normalize)

    @pytest.mark.benchmark(group="operation_standardize")
    def test_dasjax_compiled_operation_standardize(
        self, benchmark, dasjax_standardize
    ) -> None:
        """Benchmark warmed compiled dasjax standardize execution."""
        benchmark(dasjax_standardize)

    @pytest.mark.benchmark(group="operation_standardize")
    def test_dascore_operation_standardize(
        self, benchmark, dascore_standardize
    ) -> None:
        """Benchmark DASCore standardize execution."""
        benchmark(dascore_standardize)

    @pytest.mark.benchmark(group="operation_differentiate")
    def test_dasjax_compiled_operation_differentiate(
        self, benchmark, dasjax_differentiate
    ) -> None:
        """Benchmark warmed compiled dasjax differentiate execution."""
        benchmark(dasjax_differentiate)

    @pytest.mark.benchmark(group="operation_differentiate")
    def test_dascore_operation_differentiate(
        self, benchmark, dascore_differentiate
    ) -> None:
        """Benchmark DASCore differentiate execution."""
        benchmark(dascore_differentiate)

    @pytest.mark.benchmark(group="operation_integrate")
    def test_dasjax_compiled_operation_integrate(
        self, benchmark, dasjax_integrate
    ) -> None:
        """Benchmark warmed compiled dasjax integrate execution."""
        benchmark(dasjax_integrate)

    @pytest.mark.benchmark(group="operation_integrate")
    def test_dascore_operation_integrate(self, benchmark, dascore_integrate) -> None:
        """Benchmark DASCore integrate execution."""
        benchmark(dascore_integrate)

    @pytest.mark.benchmark(group="operation_taper")
    def test_dasjax_compiled_operation_taper(self, benchmark, dasjax_taper) -> None:
        """Benchmark warmed compiled dasjax taper execution."""
        benchmark(dasjax_taper)

    @pytest.mark.benchmark(group="operation_taper")
    def test_dascore_operation_taper(self, benchmark, dascore_taper) -> None:
        """Benchmark DASCore taper execution."""
        benchmark(dascore_taper)

    @pytest.mark.benchmark(group="operation_pad")
    def test_dasjax_compiled_operation_pad(self, benchmark, dasjax_pad) -> None:
        """Benchmark warmed compiled dasjax pad execution."""
        benchmark(dasjax_pad)

    @pytest.mark.benchmark(group="operation_pad")
    def test_dascore_operation_pad(self, benchmark, dascore_pad) -> None:
        """Benchmark DASCore pad execution."""
        benchmark(dascore_pad)

    @pytest.mark.benchmark(group="operation_mean")
    def test_dasjax_compiled_operation_mean(self, benchmark, dasjax_mean) -> None:
        """Benchmark warmed compiled dasjax mean reduction execution."""
        benchmark(dasjax_mean)

    @pytest.mark.benchmark(group="operation_mean")
    def test_dascore_operation_mean(self, benchmark, dascore_mean) -> None:
        """Benchmark DASCore mean reduction execution."""
        benchmark(dascore_mean)

    @pytest.mark.benchmark(group="operation_pass_filter")
    def test_dasjax_compiled_operation_pass_filter(
        self, benchmark, dasjax_pass_filter
    ) -> None:
        """Benchmark warmed compiled dasjax pass_filter execution."""
        benchmark(dasjax_pass_filter)

    @pytest.mark.benchmark(group="operation_pass_filter")
    def test_dascore_operation_pass_filter(
        self, benchmark, dascore_pass_filter
    ) -> None:
        """Benchmark DASCore pass_filter execution."""
        benchmark(dascore_pass_filter)

    @pytest.mark.benchmark(group="operation_fbe")
    def test_dasjax_compiled_operation_fbe(self, benchmark, dasjax_fbe) -> None:
        """Benchmark warmed compiled dasjax fbe execution."""
        benchmark(dasjax_fbe)

    @pytest.mark.benchmark(group="operation_fbe")
    def test_dascore_operation_fbe(self, benchmark, dascore_fbe) -> None:
        """Benchmark DASCore fbe-equivalent execution."""
        benchmark(dascore_fbe)
