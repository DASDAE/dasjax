# Agent Notes

- Prefer explanatory comments in numerical kernels, coordinate math, and callback boundaries.
- Target roughly one meaningful comment for every 5-10 lines in dense array code.
- Comment the reason for a layout transform, cached constant, broadcast shape, or numerical safeguard.
- Do not add filler comments that simply restate the next line.
- When splitting modules, keep package-level re-exports stable unless the task explicitly narrows the public API.
