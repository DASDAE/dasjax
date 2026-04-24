# Agent Notes

- Prefer explanatory comments in numerical kernels, coordinate math, and callback boundaries.
- Target roughly one meaningful comment for every 5-10 lines in dense array code.
- Comment the reason for a layout transform, cached constant, broadcast shape, or numerical safeguard.
- Do not add filler comments that simply restate the next line.
- Write tests, follow red green TDD.
- Ensure all methods, functions, modules, and classes have a docstring. Private objects can have a single line, public objects should have a full numpy docstring with examples (following doctest).
