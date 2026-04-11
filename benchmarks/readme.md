# Benchmarks

`dasjax` uses pytest-style benchmarks so the suite matches DASCore's benchmark structure and is easy to compare locally.

## Running Benchmarks

```bash
# Install test dependencies
python -m pip install -e ".[test]"

# Run all dasjax benchmarks with pytest-benchmark
pytest benchmarks/

# Run the compiled pipeline comparisons only
pytest benchmarks/test_pipeline_benchmarks.py
```

## Benchmark Structure

The first benchmark suite focuses on side-by-side comparisons between:

- compiled `dasjax` pipelines
- equivalent DASCore-native operation chains

Each comparison is exposed as a separate benchmark test per engine so CodSpeed output is easy to read.

To export benchmark results for ratio comparisons, use:

```bash
pytest benchmarks/test_pipeline_benchmarks.py --benchmark-json=.benchmarks.json
```
