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

