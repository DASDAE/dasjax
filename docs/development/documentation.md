# Documentation

Documentation is built with Zensical. The API reference page is generated from the public `dasjax` package API immediately before the site build. Benchmark documentation is generated from the checked-in benchmark snapshot.

```bash
uv run python scripts/build_api_docs.py
uv run python scripts/build_benchmark_docs.py
uv run --extra docs zensical build --clean
```

Preview the site locally with:

```bash
uv run python scripts/build_api_docs.py
uv run python scripts/build_benchmark_docs.py
uv run --extra docs zensical serve
```

Generated API reference pages are written under `docs/api/`, generated benchmark docs are written under `docs/benchmarks/`, and the static site is written to `site/`. These outputs are ignored by version control.

Refresh the benchmark snapshot before building benchmark docs with:

```bash
uv run dasjax-benchmark refresh
```

## Local Checks

```bash
uv run python scripts/build_api_docs.py
uv run python scripts/build_benchmark_docs.py
uv run --extra docs zensical build --clean
uv run pytest
```
