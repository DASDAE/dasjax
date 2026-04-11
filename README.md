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

### Three-Tier Architecture

`dasjax` is organized as a small three-tier stack:

1. Pipeline layer:
   `src/dasjax/pipeline.py` records operation chains and compiles reusable patch transforms. This is the main user-facing API.
2. Operation and pipeline layer:
   `src/dasjax/operations.py` defines the operation registry, validation rules, internal eager implementations, and compiled behavior.
3. Kernel layer:
   `src/dasjax/kernels.py` contains the array-level JAX and callback-backed kernels that actually do the numerical work.

This split keeps the package easier to extend: add or update numerical behavior in the kernel layer, describe how it plugs into compiled execution in the operation layer, and expose it through the pipeline layer.


### Roadmap

The table below tracks what is missing and roughly how much effort each addition requires.

#### Near-term — straightforward pure-JAX array ops

No new infrastructure needed; each maps directly to one or two `jnp` calls.

| Method | Implementation notes |
|---|---|
| `real`, `imag`, `angle`, `conj` | `jnp` one-liners for complex patches |
| `flip` | `jnp.flip` along a named dim |
| `roll` | `jnp.roll` circular shift along a dim |
| `pad` | `jnp.pad` with DASCore coordinate extension |
| `standardize` | zero-mean + unit-std (compare `normalize`) |
| `differentiate` | `jnp.diff` finite differences along a dim |
| `integrate` | `jnp.cumsum` / trapezoid along a dim |
| `dft` / `idft` | `jnp.fft.rfft` / `irfft` with coord reconstruction |
| `hilbert` / `envelope` | hilbert via FFT; `envelope = abs(hilbert(data))` |
| `taper` / `taper_range` | hann / cosine windows broadcast along axis |
| `whiten` | spectral divide-by-amplitude via FFT |

#### Medium-term — moderate effort or shape-changing

These need either more work in kernels.py or are shape-changing (segmented pipeline execution, same mechanism as `fbe`).

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

- Add new JAX patch methods by defining an array kernel in `src/dasjax/kernels.py` and one operation spec in `src/dasjax/operations.py`.
- The operation spec is the single source of truth for pipeline support, validation, and shared parity test cases.
- Every new patch method must be tested against a DASCore baseline across the shared mixed-patch fixture in `tests/conftest.py`.
- Prefer comparing internal operation behavior and compiled pipeline outputs against the closest native DASCore method or operator. If DASCore has no direct method, compare against an equivalent `Patch.update(...)` baseline.
- Method-equivalence assertions should check data closeness with `equal_nan=True` when needed and should also verify coordinate preservation.
- Compiled pipeline parity should come from the same declared operation cases rather than a separate hand-maintained test matrix.
- Install Git hooks locally with `prek install`.
