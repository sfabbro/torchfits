#!/usr/bin/env python3
"""Splice rendered benchmark sections into docs/benchmarks.md."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def _render(script: str, *args: str) -> str:
    out = subprocess.check_output(
        [sys.executable, str(_REPO / "scripts" / script), *args],
        text=True,
    )
    return out.rstrip() + "\n"


def _replace_block(text: str, begin: str, end: str, body: str) -> str:
    pattern = re.compile(
        rf"({re.escape(begin)})\r?\n.*?\r?\n?({re.escape(end)})",
        re.DOTALL,
    )
    if not pattern.search(text):
        raise SystemExit(f"missing markers {begin!r} .. {end!r} in docs")
    return pattern.sub(rf"\1\n{body.rstrip()}\n\2", text, count=1)


def _host_label(csv_path: Path) -> str:
    """OS / arch / accelerator — never raw hostname.

    Device signal comes from the actual benchmark data (per-row ``metadata``
    ``device`` field, then the ``host`` column token), not the run-id tag: the
    local bench script names every run ``exhaustive_mps_*`` regardless of the
    platform actually used, and a run with no device observed is a CPU run.
    """
    devices: set[str] = set()
    host_tokens: set[str] = set()
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            md = row.get("metadata") or ""
            m = re.search(r"['\"]device['\"]\s*:\s*['\"]([^'\"]+)['\"]", md)
            if m:
                devices.add(m.group(1).lower())
            elif "cuda" in md.lower():
                devices.add("cuda")
            elif "mps" in md.lower():
                devices.add("mps")
            host = (row.get("host") or "").lower()
            if "cuda" in host:
                host_tokens.add("cuda")
            elif "mps" in host:
                host_tokens.add("mps")
            elif "cpu" in host:
                host_tokens.add("cpu")

    device = "cpu"
    for signal in ("cuda", "mps"):
        if signal in devices or signal in host_tokens:
            device = signal
            break

    if device == "cuda":
        return "Linux x86_64 / CUDA"
    if device == "mps":
        return "macOS arm64 / MPS"
    return "Linux x86_64 / CPU"


def _median_rss(csv_path: Path) -> str:
    vals: list[float] = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            raw = (row.get("peak_rss_mb") or "").strip()
            if not raw:
                continue
            try:
                vals.append(float(raw))
            except ValueError:
                continue
    if not vals:
        return "-"
    vals.sort()
    return f"{vals[len(vals) // 2]:.1f}"


def _run_stats(run_id: str, csv_path: Path, deficits_path: Path) -> dict[str, str]:
    with csv_path.open(newline="", encoding="utf-8") as fh:
        row_count = sum(1 for _ in csv.DictReader(fh))
    deficit_count = 0
    if deficits_path.is_file():
        with deficits_path.open(newline="", encoding="utf-8") as fh:
            deficit_count = sum(1 for _ in csv.DictReader(fh))
    has_mmap_off = False
    has_gpu = False
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            mmap = (row.get("mmap_target") or "").lower()
            if mmap in {"off", "false", "0"}:
                has_mmap_off = True
            md = row.get("metadata") or ""
            if "io_transport" in md and row.get("status") == "OK":
                has_gpu = True
    notes = ["lab"]
    if has_mmap_off:
        notes.append("mmap-matrix")
    if has_gpu:
        notes.append("GPU")
    return {
        "run_id": run_id,
        "host": _host_label(csv_path),
        "rows": str(row_count),
        "deficits": str(deficit_count),
        "median_rss_mb": _median_rss(csv_path),
        "notes": " + ".join(notes),
    }


def _hosts_table(runs: list[dict[str, str]]) -> str:
    lines = []
    for r in runs:
        lines.append(
            f"| {r['host']} | `{r['run_id']}` | {r['rows']} | {r['deficits']} | "
            f"{r['median_rss_mb']} | {r['notes']} |"
        )
    return "\n".join(lines) + "\n"


def _concat_deficits(paths: list[Path]) -> Path:
    rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    for path in paths:
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if fieldnames is None:
                fieldnames = list(reader.fieldnames or [])
            for row in reader:
                rows.append(dict(row))
    if fieldnames is None:
        fieldnames = ["run_id", "domain", "case_id", "lag_ratio"]
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix="_deficits.csv",
        delete=False,
        encoding="utf-8",
        newline="",
    )
    with tmp:
        writer = csv.DictWriter(tmp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return Path(tmp.name)


def _resolve_runs(args: argparse.Namespace) -> list[tuple[str, Path, Path]]:
    runs: list[tuple[str, Path, Path]] = []
    if args.run_dir:
        for d in args.run_dir:
            run_dir = Path(d)
            csv_path = run_dir / "results.csv"
            deficits = run_dir / "torchfits_deficits.csv"
            if not csv_path.is_file():
                raise SystemExit(f"missing results.csv under {run_dir}")
            runs.append((run_dir.name, csv_path, deficits))
    if args.csv:
        if not args.run_id or not args.deficits:
            raise SystemExit("--csv requires --run-id and --deficits")
        runs.append((args.run_id, Path(args.csv), Path(args.deficits)))
    if not runs:
        raise SystemExit("provide --run-dir and/or --csv/--deficits/--run-id")
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, default=_REPO / "docs" / "benchmarks.md")
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        help="benchmarks_results/<run_id> directory (repeatable for multi-host)",
    )
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--deficits", type=Path, default=None)
    parser.add_argument("--run-id", default="")
    parser.add_argument(
        "--quick-dir",
        type=Path,
        default=_REPO / "benchmarks_results" / "quick",
        help="Directory containing per-scope quick-bench JSON files "
        "(`fits.json`, `fitstable.json`). Missing files produce empty rows.",
    )
    args = parser.parse_args()
    resolved = _resolve_runs(args)
    primary_csv = resolved[0][1]
    primary_dir = primary_csv.parent
    deficit_paths = [d for _rid, _c, d in resolved]
    concat_deficits = _concat_deficits(deficit_paths)

    iopath = _render("render_bench_iopath_table.py", "--csv", str(primary_csv))
    deficits = _render("render_bench_deficits.py", "--csv", str(concat_deficits))
    highlights = _render(
        "render_bench_highlights.py", "--results-dir", str(primary_dir)
    )
    full_table = _render(
        "render_full_benchmarks_table.py", "--results-dir", str(primary_dir)
    )
    quick = _render("render_bench_quick.py", "--quick-dir", str(args.quick_dir))
    ml_csv = primary_dir / "ml_results.csv"
    ml = (
        _render("render_bench_ml.py", "--csv", str(ml_csv))
        if ml_csv.is_file()
        else "_Run `pixi run bench-ml` to populate ML loader throughput._\n"
    )
    if deficits.startswith("## Performance deficits"):
        deficits = deficits.split("\n", 2)[-1] if "\n\n" in deficits else ""
        if deficits.startswith("\n"):
            deficits = deficits[1:]
    if highlights.startswith("## Performance Highlights"):
        highlights = highlights.split("\n", 2)[-1] if "\n\n" in highlights else ""
        if highlights.startswith("\n"):
            highlights = highlights[1:]
    if full_table.startswith("## Exhaustive Benchmark Results"):
        full_table = full_table.split("\n", 2)[-1] if "\n\n" in full_table else ""
        if full_table.startswith("\n"):
            full_table = full_table[1:]

    stats = [_run_stats(rid, csv_p, def_p) for rid, csv_p, def_p in resolved]
    hosts = _hosts_table(stats)

    text = args.docs.read_text(encoding="utf-8")
    text = _replace_block(
        text, "<!-- BENCH_IOPATH_BEGIN -->", "<!-- BENCH_IOPATH_END -->", iopath
    )
    text = _replace_block(
        text,
        "<!-- BENCH_HIGHLIGHTS_BEGIN -->",
        "<!-- BENCH_HIGHLIGHTS_END -->",
        highlights,
    )
    text = _replace_block(
        text,
        "<!-- BENCH_FULL_TABLE_BEGIN -->",
        "<!-- BENCH_FULL_TABLE_END -->",
        full_table,
    )
    text = _replace_block(
        text, "<!-- BENCH_DEFICITS_BEGIN -->", "<!-- BENCH_DEFICITS_END -->", deficits
    )
    if "<!-- BENCH_HOSTS_BEGIN -->" in text:
        text = _replace_block(
            text, "<!-- BENCH_HOSTS_BEGIN -->", "<!-- BENCH_HOSTS_END -->", hosts
        )
    text = _replace_block(
        text, "<!-- BENCH_QUICK_BEGIN -->", "<!-- BENCH_QUICK_END -->", quick
    )
    if "<!-- BENCH_ML_BEGIN -->" in text:
        text = _replace_block(
            text, "<!-- BENCH_ML_BEGIN -->", "<!-- BENCH_ML_END -->", ml
        )
    args.docs.write_text(text, encoding="utf-8")
    concat_deficits.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
