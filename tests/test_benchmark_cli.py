"""Tests for the dasjax benchmark CLI helpers."""

from __future__ import annotations

import argparse
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

from dasjax.benchmark_cli import (
    _benchmark_case,
    _benchmark_category,
    _benchmark_engine,
    _benchmark_group,
    _case_label,
    _format_ms,
    _format_speedup,
    _mean_seconds,
    _version,
    build_parser,
    build_pytest_command,
    docs_command,
    load_json,
    main,
    normalize_command,
    normalize_benchmark_json,
    render_benchmark_docs,
    render_benchmark_table_page,
    refresh_command,
    run_benchmarks,
    write_json,
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


def test_benchmark_parsing_fallbacks() -> None:
    """Cover helper fallbacks used for raw pytest-benchmark data."""
    assert _version("package-that-should-not-exist-for-dasjax-tests") == "unknown"
    assert _benchmark_group({"name": "test_dasjax_compiled_operation_add[small]"}) == (
        "operation_add"
    )
    assert _benchmark_group({"name": "test_custom_case"}) == "custom_case"
    assert _benchmark_engine({"name": "unmatched"}) is None
    assert _case_label("custom") == "custom"
    assert _case_label(((3, 5), "float32")) == "3x5-f32"
    assert _case_label(("bad", "float32")) == "('bad', 'float32')"
    assert _benchmark_case({"name": "bench[medium-f32]"}) == "medium-f32"
    assert _benchmark_case({"name": "bench"}) == "default"
    assert _mean_seconds({"stats": {}}) is None
    assert _format_ms(None) == "n/a"
    assert _format_speedup(float("nan")) == "n/a"
    assert _benchmark_category("operation_add") == "Method"
    assert _benchmark_category("scale_add") == "Pipeline"


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


def test_normalize_skips_unpaired_or_unusable_benchmarks() -> None:
    """Ignore rows without a recognized engine or mean value."""
    raw = {
        "benchmarks": [
            {"name": "test_unknown", "group": "unknown", "stats": {"mean": 1.0}},
            {"name": "test_dasjax_compiled_add", "group": "add", "stats": {}},
            {
                "name": "test_dascore_add",
                "group": "add",
                "params": {"case": "custom"},
                "stats": {"mean": 2.0},
            },
        ]
    }

    snapshot = normalize_benchmark_json(raw, command="pytest")

    assert snapshot["rows"] == [
        {
            "group": "add",
            "case": "custom",
            "dascore_mean_s": 2.0,
            "dasjax_mean_s": None,
            "speedup": None,
        }
    ]


def test_json_helpers_round_trip(tmp_path: Path) -> None:
    """Write and load stable JSON payloads."""
    path = tmp_path / "nested" / "payload.json"
    payload = {"b": 2, "a": 1}

    write_json(path, payload)

    assert load_json(path) == payload
    assert path.read_text(encoding="utf-8").startswith('{\n  "a"')


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
    assert 'data-sort-index="4" data-sort-type="number"' in text
    assert 'data-sort-value="4"' in text
    assert "<td><code>fbe</code></td>" in text
    assert "<td><code>large-f32</code></td>" in text
    assert "4.00x" in text


def test_run_benchmarks_invokes_pytest(monkeypatch, tmp_path: Path) -> None:
    """Run command creates the output directory and invokes subprocess."""
    calls = []

    def fake_run(command, check):
        calls.append((command, check))
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr("dasjax.benchmark_cli.subprocess.run", fake_run)
    args = argparse.Namespace(
        benchmark_file=Path("bench.py"),
        output=tmp_path / "raw" / "bench.json",
        benchmark=None,
        case=None,
        pytest_arg=[],
    )

    assert run_benchmarks(args) == 7
    assert args.output.parent.exists()
    assert calls[0][1] is False
    assert str(args.output) in calls[0][0][-1]


def test_normalize_and_docs_commands_write_outputs(tmp_path: Path) -> None:
    """Exercise normalize and docs subcommands without invoking pytest."""
    raw_path = tmp_path / "raw.json"
    snapshot_path = tmp_path / "snapshot.json"
    docs_path = tmp_path / "docs" / "index.md"
    raw_path.write_text(
        """
        {
          "benchmarks": [
            {
              "name": "test_dasjax_compiled_operation_add[case]",
              "group": "operation_add",
              "stats": {"mean": 0.5}
            },
            {
              "name": "test_dascore_operation_add[case]",
              "group": "operation_add",
              "stats": {"mean": 1.0}
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    normalize_args = argparse.Namespace(
        input=raw_path,
        output=snapshot_path,
        source="unit",
        command=None,
    )
    assert normalize_command(normalize_args) == 0

    docs_args = argparse.Namespace(snapshot=snapshot_path, output=docs_path)
    assert docs_command(docs_args) == 0

    assert "Benchmarks" in docs_path.read_text(encoding="utf-8")
    assert (docs_path.parent / "methods.md").exists()
    assert (docs_path.parent / "pipelines.md").exists()


def test_refresh_command_stops_when_benchmark_run_fails(monkeypatch, tmp_path: Path):
    """Refresh returns early when pytest-benchmark fails."""
    monkeypatch.setattr("dasjax.benchmark_cli.run_benchmarks", lambda args: 3)
    args = argparse.Namespace(
        benchmark_file=Path("bench.py"),
        raw_output=tmp_path / "raw.json",
        snapshot=tmp_path / "snapshot.json",
        docs_output=tmp_path / "docs" / "index.md",
        source="unit",
        benchmark=None,
        case=None,
        pytest_arg=[],
    )

    assert refresh_command(args) == 3


def test_refresh_command_runs_all_steps(monkeypatch, tmp_path: Path) -> None:
    """Refresh wires run, normalize, and docs commands together."""
    calls = []

    def fake_run(args):
        calls.append(("run", args.output))
        return 0

    def fake_normalize(args):
        calls.append(("normalize", args.input, args.output, args.command))
        return 0

    def fake_docs(args):
        calls.append(("docs", args.snapshot, args.output))
        return 0

    monkeypatch.setattr("dasjax.benchmark_cli.run_benchmarks", fake_run)
    monkeypatch.setattr("dasjax.benchmark_cli.normalize_command", fake_normalize)
    monkeypatch.setattr("dasjax.benchmark_cli.docs_command", fake_docs)
    args = argparse.Namespace(
        benchmark_file=Path("bench.py"),
        raw_output=tmp_path / "raw.json",
        snapshot=tmp_path / "snapshot.json",
        docs_output=tmp_path / "docs" / "index.md",
        source="unit",
        benchmark=["add"],
        case=["medium"],
        pytest_arg=["--quiet"],
    )

    assert refresh_command(args) == 0
    assert [call[0] for call in calls] == ["run", "normalize", "docs"]
    assert "-k" in calls[1][3]


def test_parser_and_main_dispatch(monkeypatch, tmp_path: Path) -> None:
    """Cover parser construction and main dispatch."""
    parser = build_parser()
    args = parser.parse_args(["docs", "--snapshot", str(tmp_path / "s.json")])
    assert args.snapshot == tmp_path / "s.json"

    def fake_run(args):
        assert args.benchmark == ["add"]
        return 4

    monkeypatch.setattr("dasjax.benchmark_cli.run_benchmarks", fake_run)

    assert main(["run", "--benchmark", "add"]) == 4


def test_main_defaults_to_refresh(monkeypatch) -> None:
    """Bare CLI invocation runs the full refresh workflow."""
    calls = []

    def fake_refresh(args):
        calls.append(args.command)
        return 0

    monkeypatch.setattr("dasjax.benchmark_cli.refresh_command", fake_refresh)

    assert main([]) == 0
    assert calls == ["refresh"]


def test_parser_requires_subcommand() -> None:
    """Argparse exits when no subcommand is supplied."""
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
