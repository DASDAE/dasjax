# Architecture

`dasjax` is organized around one core operation model.

## Pipeline Layer

`src/dasjax/pipeline.py` records operation chains, plans metadata boundaries, and compiles reusable patch transforms. This is the main user-facing API. Compiled callables cache both JIT segment runners and bound metadata plans for repeated calls with the same static boundary.

## Operation Layer

`src/dasjax/core.py` defines `PatchOperation`, `PatchBoundary`, `PatchPyTree`, and registry helpers. Registered operation classes live under `src/dasjax/operations/`, grouped by DASCore-style domains.

Operation authors use `bind(boundary)` for Python-side metadata planning, `kernel(patch_tree)` for JAX-side data transforms, and `update_boundary(boundary)` for static metadata changes.

## Kernel Layer

`src/dasjax/kernels/` contains the array-level JAX kernels that do the numerical work, grouped by domain.
