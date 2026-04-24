"""Tests for the dasjax benchmark CLI helpers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dasjax.benchmark_cli import (
    build_pytest_command,
    normalize_benchmark_json,
    render_benchmark_docs,
    render_benchmark_table_page,
)


def test_build_pytest_command_with_filters() -> None:
    """Build a filtered pytest-benchmark command."""
    args = argparse.Namespace(
        benchmark_file=Path("benchmarks/test_pipeline_benchmarks.py"),
        output=Path(".benchmarks/current.json"),
        benchmark=["scale_fbe"],
        case=["medium-f64"],
        pytest_arg=["--benchmark-min-rounds=1"],
    )

    command = build_pytest_command(args)

    assert command[:3] == [sys.executable, "-m", "pytest"]
    assert "--benchmark-json=.benchmarks/current.json" in command
    assert command[command.index("-k") + 1] == "(scale_fbe) and (medium-f64)"
    assert command[-1] == "--benchmark-min-rounds=1"


def test_normalize_benchmark_json_pairs_dasjax_and_dascore() -> None:
    """Normalize raw pytest-benchmark JSON into speedup rows."""
    raw = {
        "benchmarks": [
            {
                "name": "test_dasjax_compiled_scale_fbe[medium-f64]",
                "group": "scale_fbe",
                "params": {"example_patch": [[600, 4000], "float64"]},
                "stats": {"mean": 0.25},
            },
            {
                "name": "test_dascore_scale_fbe[medium-f64]",
                "group": "scale_fbe",
                "params": {"example_patch": [[600, 4000], "float64"]},
                "stats": {"mean": 1.0},
            },
        ]
    }

    snapshot = normalize_benchmark_json(raw, command="pytest benchmarks", source="test")

    assert snapshot["source"] == "test"
    assert snapshot["rows"] == [
        {
            "group": "scale_fbe",
            "case": "medium-f64",
            "dascore_mean_s": 1.0,
            "dasjax_mean_s": 0.25,
            "speedup": 4.0,
        }
    ]


def test_render_benchmark_docs_includes_results_table() -> None:
    """Render benchmark docs overview from a normalized snapshot."""
    snapshot = {
        "generated_at": "2026-04-24T00:00:00+00:00",
        "source": "test",
        "command": "pytest benchmarks/test_pipeline_benchmarks.py",
        "environment": {"python": "3.12", "platform": "linux"},
        "rows": [
            {
                "group": "scale_fbe",
                "case": "medium-f64",
                "dascore_mean_s": 1.0,
                "dasjax_mean_s": 0.25,
                "speedup": 4.0,
            }
        ],
    }

    text = render_benchmark_docs(snapshot)

    assert "# Benchmarks" in text
    assert "[1 rows](pipelines.md)" in text
    assert "[0 rows](methods.md)" in text


def test_render_benchmark_table_page_includes_filterable_rows() -> None:
    """Render a filterable rich table for benchmark rows."""
    snapshot = {
        "generated_at": "2026-04-24T00:00:00+00:00",
        "rows": [],
    }
    rows = [
        {
            "group": "operation_fbe",
            "case": "large-f32",
            "dascore_mean_s": 1.0,
            "dasjax_mean_s": 0.25,
            "speedup": 4.0,
        }
    ]

    text = render_benchmark_table_page(snapshot, title="Methods", rows=rows)

    assert 'data-benchmark-filter type="search"' in text
    assert "<td><code>fbe</code></td>" in text
    assert "<td><code>large-f32</code></td>" in text
    assert "4.00x" in text
