# dasjax Agent Guide

This file gives AI/code agents a practical checklist for contributing safely to dasjax.

Keep Markdown prose unwrapped. Do not hard-wrap paragraphs or list items in `.md` files; let editors soft-wrap them. Code blocks and formats that require line breaks are exceptions.

## Scope and priorities

1. Keep changes minimal, targeted, and test-backed.
2. Preserve DASCore conventions over personal preferences.
3. Prefer consistency with existing code/tests/docs in this repo.
4. Treat `JaxPatchPipeline` as the primary public API.

## Linting and formatting

- Run `prek` hooks before finalizing changes.
- Project lint/format is driven by `.pre-commit-config.yaml` and Ruff config in `pyproject.toml`.

```bash
prek run --all-files
```

Tip: running twice can apply auto-fixes on first pass.

## Testing requirements

Run targeted tests for changed behavior, then broader tests as needed:

```bash
pytest tests/path/to/affected_test.py
pytest tests
```

For performance-sensitive changes, run the relevant benchmark slice:

```bash
pytest benchmarks/test_pipeline_benchmarks.py
```

For coverage checks:

```bash
pytest tests --cov dasjax --cov-report term-missing
```

For doctests:

```bash
pytest src/dasjax --doctest-modules
```

Unless otherwise specified, a job is not finished until the relevant checks pass.

## Test authoring conventions

- Put tests under `tests/` mirroring package structure.
- Group tests in classes.
- Place fixtures as close as practical to usage (class, module, then `conftest.py`).
- Write tests that focus on boundaries, not implementation details.
- Every test function or method should include a short docstring.
- Keep test names short; put extra detail in the docstring.


## Code conventions


- Prefer `pathlib.Path` over raw path strings when practical.
- Add type hints for public functions/methods.
- Use NumPy-style docstrings for public APIs.
- Add a short explanatory comment for private helpers when intent is not obvious.
- Keep comments meaningful; do not restate obvious code.
- Prefer pipeline-based APIs and tests over ad hoc eager wrappers.

## Documentation changes

If behavior or the public API changes, update `README.md` in the same change unless there is a better local documentation target.

## Quality bar for agent changes

Before handing off:

1. Code compiles/runs for changed paths.
2. Relevant tests pass locally.
3. Lint/format checks pass.
4. Benchmarks updated or rerun for performance-sensitive changes.
5. No unrelated refactors bundled with bug fixes.

## When uncertain

- Prefer existing patterns in nearby `src/dasjax` modules and `tests/`.
- Call out assumptions explicitly in PR notes.
- Choose the simpler behavior-preserving implementation first.
