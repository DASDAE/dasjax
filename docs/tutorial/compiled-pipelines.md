# Compiled Pipelines

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

## Workflow

1. Create a `JaxPatchPipeline`.
2. Add DASCore-style patch operations.
3. Call `.compile()` once.
4. Reuse the compiled callable with `patch.pipe(compiled)` or `compiled(patch)`.

## Compatibility

The compiled callable can be reused for patches with matching static metadata: dims, dynamic coordinate values, coordinate units, and attrs. When those metadata inputs change, `dasjax` plans and caches a new compatible execution path.
