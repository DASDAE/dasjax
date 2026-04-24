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

The benchmark suite focuses on side-by-side comparisons between:

- compiled `dasjax` pipelines
- equivalent DASCore-native operation chains
- individual compiled `dasjax` operations
- equivalent individual DASCore operations

Each comparison is exposed as a separate benchmark test per engine so CodSpeed output is easy to read. Pipeline benchmark groups use names like `scale_fbe`; individual operation benchmark groups use names like `operation_fbe`.

To export benchmark results for ratio comparisons, use:

```bash
pytest benchmarks/test_pipeline_benchmarks.py --benchmark-json=.benchmarks.json
```
