"""Command line helpers for running and publishing dasjax benchmarks."""

from __future__ import annotations

import argparse
import html
import json
import math
import platform
import re
import subprocess
import sys
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

DEFAULT_BENCHMARK_FILE = Path("benchmarks/test_pipeline_benchmarks.py")
DEFAULT_RAW_OUTPUT = Path(".benchmarks/current.json")
DEFAULT_SNAPSHOT = Path("benchmarks/results/current.json")
DEFAULT_DOCS_OUTPUT = Path("docs/benchmarks/index.md")
DEFAULT_DOCS_DIR = Path("docs/benchmarks")


def _version(package: str) -> str:
    """Return an installed package version or a fallback label."""
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "unknown"


def _now_iso() -> str:
    """Return the current UTC timestamp as an ISO string."""
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat()


def build_pytest_command(args: argparse.Namespace) -> list[str]:
    """Build the pytest-benchmark command from parsed CLI args."""
    command = [
        sys.executable,
        "-m",
        "pytest",
        str(args.benchmark_file),
        f"--benchmark-json={args.output}",
    ]
    filters = []
    if args.benchmark:
        filters.append("(" + " or ".join(args.benchmark) + ")")
    if args.case:
        filters.append("(" + " or ".join(args.case) + ")")
    if filters:
        command.extend(["-k", " and ".join(filters)])
    command.extend(args.pytest_arg or ())
    return command


def run_benchmarks(args: argparse.Namespace) -> int:
    """Run pytest-benchmark and write raw benchmark JSON."""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = build_pytest_command(args)
    print("Running:", " ".join(command))
    return subprocess.run(command, check=False).returncode


def _benchmark_group(benchmark: dict[str, Any]) -> str:
    """Return a benchmark group name from pytest-benchmark data."""
    if group := benchmark.get("group"):
        return str(group)
    name = str(benchmark.get("name") or benchmark.get("fullname") or "")
    name = re.sub(r"\[.*\]$", "", name)
    for prefix in ("test_dasjax_compiled_", "test_dascore_"):
        if prefix in name:
            return name.rsplit(prefix, 1)[1]
    return name.removeprefix("test_")


def _benchmark_engine(benchmark: dict[str, Any]) -> str | None:
    """Return dasjax/dascore for known benchmark functions."""
    text = " ".join(
        str(benchmark.get(key, "")) for key in ("name", "fullname", "fullname")
    )
    if "dasjax" in text:
        return "dasjax"
    if "dascore" in text:
        return "dascore"
    return None


def _case_label(value: Any) -> str:
    """Return a readable label for a pytest benchmark parameter."""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 2:
        shape, dtype = value
        if isinstance(shape, (list, tuple)) and len(shape) == 2:
            shape_label = {
                (600, 4000): "medium",
                (1200, 8000): "large",
            }.get(tuple(int(axis) for axis in shape), "x".join(str(axis) for axis in shape))
            dtype_label = str(dtype).removeprefix("float")
            return f"{shape_label}-f{dtype_label}"
    return str(value)


def _benchmark_case(benchmark: dict[str, Any]) -> str:
    """Return the parametrized patch case label."""
    params = benchmark.get("params") or {}
    if isinstance(params, dict):
        for key in ("example_patch", "patch", "case"):
            if key in params:
                return _case_label(params[key])
    text = str(benchmark.get("name") or benchmark.get("fullname") or "")
    match = re.search(r"\[([^\]]+)\]", text)
    return match.group(1) if match else "default"


def _mean_seconds(benchmark: dict[str, Any]) -> float | None:
    """Return benchmark mean in seconds."""
    stats = benchmark.get("stats") or {}
    mean = stats.get("mean") if isinstance(stats, dict) else None
    return float(mean) if mean is not None else None


def _environment(raw: dict[str, Any]) -> dict[str, Any]:
    """Build environment metadata for a normalized snapshot."""
    machine = raw.get("machine_info") or {}
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or machine.get("processor") or "unknown",
        "machine": platform.machine(),
        "jax": _version("jax"),
        "dascore": _version("dascore"),
        "dasjax": _version("dasjax"),
    }


def normalize_benchmark_json(
    raw: dict[str, Any],
    *,
    command: str,
    source: str = "local",
) -> dict[str, Any]:
    """Normalize pytest-benchmark JSON into a docs-friendly snapshot."""
    grouped: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for benchmark in raw.get("benchmarks", ()):
        engine = _benchmark_engine(benchmark)
        mean = _mean_seconds(benchmark)
        if engine is None or mean is None:
            continue
        key = (_benchmark_group(benchmark), _benchmark_case(benchmark))
        grouped[key][engine] = mean

    rows = []
    for (group, case), values in sorted(grouped.items()):
        dascore = values.get("dascore")
        dasjax = values.get("dasjax")
        speedup = dascore / dasjax if dascore and dasjax else None
        rows.append(
            {
                "group": group,
                "case": case,
                "dascore_mean_s": dascore,
                "dasjax_mean_s": dasjax,
                "speedup": speedup,
            }
        )

    return {
        "schema_version": 1,
        "generated_at": _now_iso(),
        "source": source,
        "command": command,
        "environment": _environment(raw),
        "rows": rows,
    }


def load_json(path: Path) -> dict[str, Any]:
    """Load JSON from a path."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write stable, pretty JSON to a path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _format_ms(value: float | None) -> str:
    """Format seconds as milliseconds."""
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value * 1000:.3f}"


def _format_speedup(value: float | None) -> str:
    """Format a speedup ratio."""
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.2f}x"


def _benchmark_category(group: str) -> str:
    """Return a human category for a benchmark group."""
    return "Method" if group.startswith("operation_") else "Pipeline"


def _split_rows(snapshot: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return pipeline rows and method rows from a benchmark snapshot."""
    rows = snapshot.get("rows") or []
    pipelines = [row for row in rows if not str(row.get("group", "")).startswith("operation_")]
    methods = [row for row in rows if str(row.get("group", "")).startswith("operation_")]
    return pipelines, methods


def render_benchmark_index(snapshot: dict[str, Any]) -> str:
    """Render the benchmark section overview page."""
    env = snapshot.get("environment") or {}
    pipelines, methods = _split_rows(snapshot)
    lines = [
        "# Benchmarks",
        "",
        "<!-- This file is generated by dasjax.benchmark_cli. -->",
        "",
        "Published benchmark results compare warmed compiled `dasjax` pipelines and "
        "methods against equivalent DASCore operation chains.",
        "",
        '<div class="grid cards" markdown>',
        "",
        f"- __Source__  \n  {snapshot.get('source', 'unknown')}",
        f"- __Generated__  \n  {snapshot.get('generated_at', 'unknown')}",
        f"- __Python__  \n  {env.get('python', 'unknown')}",
        f"- __Platform__  \n  {env.get('platform', 'unknown')}",
        f"- __Pipeline results__  \n  [{len(pipelines)} rows](pipelines.md)",
        f"- __Method results__  \n  [{len(methods)} rows](methods.md)",
        "",
        "</div>",
        "",
        '!!! warning "Benchmark numbers are environment-specific"',
        "",
        "    Treat these numbers as a snapshot for this machine and dependency "
        "set, not as universal performance guarantees.",
        "",
        "## Result Tables",
        "",
        "- [Pipelines](pipelines.md)",
        "- [Methods](methods.md)",
        "",
        "## Environment",
        "",
        "| Key | Value |",
        "|---|---|",
    ]
    for key, value in env.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(
        [
            "",
            "## Refreshing Results",
            "",
            "```bash",
            "dasjax-benchmark refresh",
            "```",
            "",
            "To run the steps separately:",
            "",
            "```bash",
            "dasjax-benchmark run --output .benchmarks/current.json",
            "dasjax-benchmark normalize --input .benchmarks/current.json --output benchmarks/results/current.json",
            "dasjax-benchmark docs --snapshot benchmarks/results/current.json --output docs/benchmarks/index.md",
            "```",
            "",
            "Raw pytest-benchmark command:",
            "",
            "```bash",
            f"{snapshot.get('command', 'pytest benchmarks/test_pipeline_benchmarks.py')}",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _benchmark_display_name(group: str) -> str:
    """Return a display name for a benchmark group."""
    return group.removeprefix("operation_")


def _render_filter_script() -> str:
    """Return client-side table filtering script."""
    return """
<script>
(() => {
  for (const table of document.querySelectorAll("[data-benchmark-table]")) {
    const input = table.querySelector("[data-benchmark-filter]");
    const rows = Array.from(table.querySelectorAll("tbody tr"));
    const count = table.querySelector("[data-benchmark-count]");
    const update = () => {
      const query = input.value.trim().toLowerCase();
      let visible = 0;
      for (const row of rows) {
        const match = row.textContent.toLowerCase().includes(query);
        row.hidden = !match;
        if (match) visible += 1;
      }
      if (count) count.textContent = `${visible} of ${rows.length} rows`;
    };
    input.addEventListener("input", update);
    update();
  }
})();
</script>
""".strip()


def _render_table_style() -> str:
    """Return page-local styles for benchmark tables."""
    return """
<style>
.dasjax-benchmark-table {
  margin-top: 1rem;
}
.dasjax-benchmark-table input[type="search"] {
  width: min(100%, 32rem);
  margin-bottom: 0.75rem;
  padding: 0.55rem 0.7rem;
  border: 1px solid var(--md-default-fg-color--lighter);
  border-radius: 0.2rem;
  background: var(--md-default-bg-color);
  color: var(--md-default-fg-color);
}
.dasjax-benchmark-count {
  margin: 0 0 0.5rem;
  color: var(--md-default-fg-color--light);
  font-size: 0.8rem;
}
.dasjax-benchmark-table table {
  display: table;
  width: 100%;
}
.dasjax-benchmark-table tbody tr:hover {
  background: var(--md-code-bg-color);
}
</style>
""".strip()


def render_benchmark_table_page(
    snapshot: dict[str, Any],
    *,
    title: str,
    rows: Sequence[dict[str, Any]],
) -> str:
    """Render a filterable benchmark table page."""
    generated = html.escape(str(snapshot.get("generated_at", "unknown")))
    placeholder = html.escape(f"Filter {title.lower()}...")
    lines = [
        f"# {title}",
        "",
        "<!-- This file is generated by dasjax.benchmark_cli. -->",
        "",
        f"Generated from the benchmark snapshot at `{generated}`.",
        "",
        _render_table_style(),
        "",
        '<div class="dasjax-benchmark-table" data-benchmark-table>',
        f'<input data-benchmark-filter type="search" placeholder="{placeholder}" aria-label="{placeholder}">',
        '<p class="dasjax-benchmark-count" data-benchmark-count></p>',
        "<table>",
        "<thead>",
        "<tr>",
        "<th>Benchmark</th>",
        "<th>Case</th>",
        '<th style="text-align: right;">DASCore mean (ms)</th>',
        '<th style="text-align: right;">dasjax mean (ms)</th>',
        '<th style="text-align: right;">Speedup</th>',
        "</tr>",
        "</thead>",
        "<tbody>",
    ]
    for row in rows:
        group = html.escape(_benchmark_display_name(str(row.get("group", "unknown"))))
        case = html.escape(str(row.get("case", "default")))
        lines.extend(
            [
                "<tr>",
                f"<td><code>{group}</code></td>",
                f"<td><code>{case}</code></td>",
                f'<td style="text-align: right;">{html.escape(_format_ms(row.get("dascore_mean_s")))}</td>',
                f'<td style="text-align: right;">{html.escape(_format_ms(row.get("dasjax_mean_s")))}</td>',
                f'<td style="text-align: right;">{html.escape(_format_speedup(row.get("speedup")))}</td>',
                "</tr>",
            ]
        )
    lines.extend(
        [
            "</tbody>",
            "</table>",
            "</div>",
            "",
            _render_filter_script(),
            "",
        ]
    )
    return "\n".join(lines)


def render_benchmark_docs(snapshot: dict[str, Any]) -> str:
    """Render benchmark docs Markdown from a normalized snapshot."""
    return render_benchmark_index(snapshot)


def normalize_command(args: argparse.Namespace) -> int:
    """Normalize a raw pytest-benchmark JSON file."""
    raw = load_json(args.input)
    command = args.command or f"pytest {DEFAULT_BENCHMARK_FILE} --benchmark-json={args.input}"
    snapshot = normalize_benchmark_json(raw, command=command, source=args.source)
    write_json(args.output, snapshot)
    print(f"Wrote {args.output}")
    return 0


def docs_command(args: argparse.Namespace) -> int:
    """Render benchmark docs from a normalized snapshot."""
    snapshot = load_json(args.snapshot)
    docs_dir = args.output.parent
    docs_dir.mkdir(parents=True, exist_ok=True)
    pipelines, methods = _split_rows(snapshot)
    pages = {
        args.output: render_benchmark_index(snapshot),
        docs_dir / "pipelines.md": render_benchmark_table_page(
            snapshot, title="Pipelines", rows=pipelines
        ),
        docs_dir / "methods.md": render_benchmark_table_page(
            snapshot, title="Methods", rows=methods
        ),
    }
    for path, text in pages.items():
        path.write_text(text, encoding="utf-8")
        print(f"Wrote {path}")
    return 0


def refresh_command(args: argparse.Namespace) -> int:
    """Run benchmarks, normalize results, and render docs."""
    run_args = argparse.Namespace(
        benchmark_file=args.benchmark_file,
        output=args.raw_output,
        benchmark=args.benchmark,
        case=args.case,
        pytest_arg=args.pytest_arg,
    )
    code = run_benchmarks(run_args)
    if code:
        return code
    command = " ".join(build_pytest_command(run_args))
    normalize_args = argparse.Namespace(
        input=args.raw_output,
        output=args.snapshot,
        source=args.source,
        command=command,
    )
    normalize_command(normalize_args)
    docs_args = argparse.Namespace(snapshot=args.snapshot, output=args.docs_output)
    return docs_command(docs_args)


def build_parser() -> argparse.ArgumentParser:
    """Build the benchmark CLI parser."""
    parser = argparse.ArgumentParser(prog="dasjax-benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run pytest-benchmark suite")
    run.add_argument("--benchmark-file", type=Path, default=DEFAULT_BENCHMARK_FILE)
    run.add_argument("--output", type=Path, default=DEFAULT_RAW_OUTPUT)
    run.add_argument("--benchmark", action="append", help="filter benchmark group/name")
    run.add_argument("--case", action="append", help="filter fixture case label")
    run.add_argument("--pytest-arg", action="append", default=[], help="extra pytest arg")
    run.set_defaults(func=run_benchmarks)

    normalize = subparsers.add_parser("normalize", help="normalize raw JSON")
    normalize.add_argument("--input", type=Path, required=True)
    normalize.add_argument("--output", type=Path, default=DEFAULT_SNAPSHOT)
    normalize.add_argument("--source", default="local")
    normalize.add_argument("--command")
    normalize.set_defaults(func=normalize_command)

    docs = subparsers.add_parser("docs", help="render benchmark docs")
    docs.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    docs.add_argument("--output", type=Path, default=DEFAULT_DOCS_OUTPUT)
    docs.set_defaults(func=docs_command)

    refresh = subparsers.add_parser("refresh", help="run, normalize, and render docs")
    refresh.add_argument("--benchmark-file", type=Path, default=DEFAULT_BENCHMARK_FILE)
    refresh.add_argument("--raw-output", type=Path, default=DEFAULT_RAW_OUTPUT)
    refresh.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    refresh.add_argument("--docs-output", type=Path, default=DEFAULT_DOCS_OUTPUT)
    refresh.add_argument("--source", default="local")
    refresh.add_argument("--benchmark", action="append", help="filter benchmark group/name")
    refresh.add_argument("--case", action="append", help="filter fixture case label")
    refresh.add_argument("--pytest-arg", action="append", default=[], help="extra pytest arg")
    refresh.set_defaults(func=refresh_command)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
