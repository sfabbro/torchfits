#!/usr/bin/env python3
"""Splice rendered benchmark sections into docs/benchmarks.md."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
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


def _snapshot_row(run_id: str, csv_path: Path, deficits_path: Path) -> str:
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
    notes = ["lab bench-all"]
    if has_mmap_off:
        notes.append("`--mmap-matrix`")
    if has_gpu:
        notes.append("CUDA/MPS")
    return (
        f"| `{run_id}` | fits + fitstable (lab) | {row_count} | {deficit_count} | "
        f"{' + '.join(notes)} |\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, default=_REPO / "docs" / "benchmarks.md")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--deficits", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    iopath = _render("render_bench_iopath_table.py", "--csv", str(args.csv))
    deficits = _render("render_bench_deficits.py", "--csv", str(args.deficits))
    highlights = _render(
        "render_bench_highlights.py", "--results-dir", str(args.csv.parent)
    )
    full_table = _render(
        "render_full_benchmarks_table.py", "--results-dir", str(args.csv.parent)
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
    snapshot = _snapshot_row(args.run_id, args.csv, args.deficits)

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
    text = _replace_block(
        text, "<!-- BENCH_SNAPSHOT_BEGIN -->", "<!-- BENCH_SNAPSHOT_END -->", snapshot
    )
    args.docs.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
