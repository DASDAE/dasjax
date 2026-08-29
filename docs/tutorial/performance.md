# Performance

The intended fast path is to call `.compile()` once and reuse the returned callable.

## Caching

Patch-specific metadata binding and JIT segment creation happen lazily on the first call for a static patch boundary. Cached plans and segment runners are reused for later calls with matching dims, dynamic coordinate values, coordinate units, and attrs.

Equivalent pipeline definitions reuse cached compiled callables automatically.

## Native Kernels And Callbacks

Native JAX-backed operations can be fused into compiled segments. Some heavier DASCore-compatible numeric transforms still use host callbacks; those operations are useful for compatibility, but they do not usually benefit as much from JAX fusion as native kernels.

Benchmarks live under `benchmarks/` and compare compiled `dasjax` pipelines against equivalent DASCore operation chains.
