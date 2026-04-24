# dasjax

An experimental package for accelerating [DASCore](dascore.org) with [JAX](https://github.com/jax-ml/jax).

## Installation

```bash
python -m pip install -e ".[dev]"
```

## Usage

`dasjax`'s main feature is the ability to create compiled DAS pipelines that can run on CPU, GPU, or TPU. These also perform kernel fusions for increased efficiency.

### Compiled pipeline

Use `JaxPatchPipeline` when you want to compile a reusable sequence once and run it across many compatible patches.

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

1. Pipeline layer:
   `src/dasjax/pipeline.py` records operation chains, plans metadata boundaries, and compiles reusable patch transforms. This is the main user-facing API.
2. Operation layer:
   `src/dasjax/core.py` and `src/dasjax/core_ops.py` define `PatchOperation`, `PatchBoundary`, `PatchPyTree`, and the registered operation classes.
3. Kernel layer:
   `src/dasjax/kernels/` contains the array-level JAX and callback-backed kernels that actually do the numerical work, grouped by domain (`basic`, `signal`, `filters`, `spectral`).

Operation authors use `bind(boundary)` for Python-side metadata planning, `kernel(patch_tree)` for JAX-side data transforms, and `update_boundary(boundary)` for static metadata changes.


### Roadmap

The table below tracks what is missing and roughly how much effort each addition requires.

#### Near-term — straightforward pure-JAX array ops

Implemented in the current package:

- `real`, `imag`, `angle`, `conj`
- `flip`, `roll`, `pad`
- `standardize`, `differentiate`, `integrate`
- `dft`, `idft`
- `hilbert`, `envelope`
- `taper`, `taper_range`
- `whiten`

#### Medium-term — moderate effort or shape-changing

These need either more work in the kernel layer or are shape-changing (segmented pipeline execution, same mechanism as `fbe`).

| Method | Implementation notes |
|---|---|
| `notch_filter` | SOS filter; same pattern as `pass_filter` |
| `savgol_filter` | polynomial fitting per frame; JAX-doable |
| `rolling` | rolling-window reductions (mean, std, …); needs strided views |
| `correlate` | cross-correlation via `jnp.fft` |
| `stft` / `istft` | expose the STFT kernel already used by `fbe` |
| `decimate` | anti-aliased downsampling; shape-changing |
| `aggregate` / `mean` / `std` / `sum` | axis reductions; shape-changing |

## Performance Notes

- The intended fast path is to build a `JaxPatchPipeline`, call `.compile()` once, and reuse the returned callable across many patches of compatible shape and dtype.
- Equivalent pipeline definitions reuse cached compiled callables automatically.
- Benchmarks live under `benchmarks/` and compare compiled `dasjax` pipelines against equivalent DASCore operation chains.

## Development Guidelines

- Add new JAX patch methods by defining an array kernel in `src/dasjax/kernels/` and one `PatchOperation` subclass in `src/dasjax/core_ops.py`.
- The `PatchOperation` subclass is the single source of truth for pipeline support, metadata binding, and boundary updates.
- Every new patch method must be tested against a DASCore baseline across the shared mixed-patch fixture in `tests/conftest.py`.
- Prefer comparing internal operation behavior and compiled pipeline outputs against the closest native DASCore method or operator. If DASCore has no direct method, compare against an equivalent `Patch.update(...)` baseline.
- Method-equivalence assertions should check data closeness with `equal_nan=True` when needed and should also verify coordinate preservation.
- Compiled pipeline parity should compare `JaxPatchPipeline` output against DASCore baselines for each registered operation.
- Install Git hooks locally with `prek install`.
