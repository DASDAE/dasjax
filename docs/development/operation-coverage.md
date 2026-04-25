# Operation Coverage

The package currently registers 72 pipeline operations. Most are implemented as JAX-backed kernels with static metadata planning; heavier DASCore-compatible numeric transforms can use host callbacks when a full static JAX kernel is not yet practical. Callback-backed operations are compatibility paths and generally offer less fusion benefit than native kernels.

Remaining DASCore patch methods are mostly metadata, selection, convenience, or data-dependent shape operations. In particular, `rolling` returns a roller object and `dropna` has data-dependent output shape, so they do not map directly to the current compiled `Patch -> Patch` pipeline model.

The generated [API Reference](../api/index.md) lists the current operation registry.
