# dasjax

![dasjax logo](https://raw.githubusercontent.com/dasdae/dasjax/main/docs/static/dasjax_logo.png)

An experimental package for accelerating [DASCore](https://dascore.org) with [JAX](https://github.com/jax-ml/jax).

## Installation

```bash
python -m pip install -e ".[dev]"
```

## Usage

`dasjax`'s main feature is the ability to create compiled DAS pipelines that can run on CPU, GPU, or TPU. These pipelines fuse adjacent JAX-backed operations where possible and cache metadata planning for repeated calls with the same static patch boundary.

### Compiled pipeline

Use `JaxPatchPipeline` when you want to build a reusable callable once and run it across many compatible patches.

```python
import dascore as dc
from dasjax import JaxPatchPipeline

patch = dc.get_example_patch("example_event_1")

pipeline = (
    JaxPatchPipeline()
    .scale(2.0)
    .add(1.0)
    .detrend(dim="time", type="constant")
    .normalize(dim="time")
)
compiled = pipeline.compile()

out = patch.pipe(compiled)

print(out.shape)
```

## Development

### Architecture

`dasjax` is organized around one core operation model:

1. Pipeline layer: `src/dasjax/pipeline.py` records operation chains, plans metadata boundaries, and compiles reusable patch transforms. This is the main user-facing API.
2. Operation layer: `src/dasjax/core.py` defines `PatchOperation`, `PatchBoundary`, `PatchPyTree`, and registry helpers. Registered operation classes live under `src/dasjax/operations/`, grouped by DASCore-style domains.
3. Kernel layer: `src/dasjax/kernels/` contains the array-level JAX kernels that actually do the numerical work, grouped by domain (`basic`, `signal`, `filters`, `spectral`).

Operation authors use `bind(boundary)` for Python-side metadata planning, `kernel(patch_tree)` for JAX-side data transforms, and `update_boundary(boundary)` for static metadata changes.


### Operation Coverage

`dasjax` currently registers 72 pipeline operations. Most operations use native JAX kernels; a smaller set of DASCore-compatible numeric transforms still use host callbacks where a fully static JAX kernel is not practical yet. The current operation set includes:

- Elementwise math and masks: `abs`, `clip`, `real`, `imag`, `angle`, `conj`, `exp`, `log`, `log10`, `log2`, `is_finite`, `isinf`, `isnan`, `fillna`, `where`, and scalar arithmetic operations.
- Reductions and aggregation: `aggregate`, `all`, `any`, `max`, `mean`, `median`, `min`, `std`, and `sum`.
- Coordinate-aware array transforms: `flip`, `roll`, `pad`, `taper`, `taper_range`, `detrend`, `standardize`, `differentiate`, and `integrate`.
- Spectral and signal operations: `dft`, `idft`, `stft`, `istft`, `hilbert`, `envelope`, `phase_weighted_stack`, `whiten`, `fbe`, and `correlate_shift`.
- Filters, mutes, and DAS-domain operations: `pass_filter`, `gaussian_filter`, `hampel_filter`, `median_filter`, `notch_filter`, `savgol_filter`, `sobel_filter`, `slope_filter`, `wiener_filter`, `line_mute`, `slope_mute`, `correlate`, `decimate`, `interpolate`, `resample`, `dispersion_phase_shift`, `tau_p`, `velocity_to_strain_rate`, `velocity_to_strain_rate_edgeless`, and `radians_to_strain`.

Remaining DASCore patch methods are mostly metadata, selection, convenience, or data-dependent shape operations. `rolling` returns a roller object rather than a patch, and `dropna` has data-dependent output shape, so neither fits the current static compiled-pipeline model directly.

## Performance Notes

- The intended fast path is to build a `JaxPatchPipeline`, call `.compile()` once, and reuse the returned callable. Patch-specific metadata binding and JIT segment creation happen lazily on the first call for a static boundary, then cached plans and segment runners are reused for subsequent calls with matching dims, dynamic coordinate values, coordinate units, and attrs.
- Equivalent pipeline definitions reuse cached compiled callables automatically.
- Callback-backed operations preserve DASCore compatibility but execute their operation body on the host, so they generally do not benefit as much from JAX fusion as native kernels.
- Benchmarks live under `benchmarks/` and compare compiled `dasjax` pipelines against equivalent DASCore operation chains.

## Documentation

Documentation is built with Zensical. The public API reference is generated at build time from the installed `dasjax` package, so run the API generation script before building or serving the site.

```bash
uv run python scripts/build_api_docs.py
uv run --extra docs zensical build --clean
```

For local preview, run:

```bash
uv run python scripts/build_api_docs.py
uv run --extra docs zensical serve
```

Generated files under `docs/api/` and `site/` are ignored by version control. GitHub Pages builds the same generated API docs and static site on pushes to `main`, then deploys the `site/` artifact through the `github-pages` environment.

## Development Guidelines

- Add new JAX patch methods by defining an array kernel in `src/dasjax/kernels/` and one `PatchOperation` subclass in the appropriate `src/dasjax/operations/` module.
- The `PatchOperation` subclass is the single source of truth for pipeline support, metadata binding, and boundary updates.
- Every new patch method must be tested against a DASCore baseline across the shared mixed-patch fixture in `tests/conftest.py`.
- Prefer comparing internal operation behavior and compiled pipeline outputs against the closest native DASCore method or operator. If DASCore has no direct method, compare against an equivalent `Patch.update(...)` baseline.
- Method-equivalence assertions should check data closeness with `equal_nan=True` when needed and should also verify coordinate preservation.
- Compiled pipeline parity should compare `JaxPatchPipeline` output against DASCore baselines for each registered operation.
- Install Git hooks locally with `prek install`.
