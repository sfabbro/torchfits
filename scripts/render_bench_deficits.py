#!/usr/bin/env python3
"""Render torchfits_deficits.csv into a markdown section for docs/benchmarks.md."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def render_deficits(csv_path: Path, *, max_rows: int = 40) -> str:
    rows: list[dict[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    lines = [
        "## Performance deficits",
        "",
        "Cases where torchfits is **not** first in its comparison family "
        "(documented for transparency; not fixed in this release).",
        "",
    ]
    if not rows:
        lines.append("_No deficits in this run — torchfits won every comparable case._")
        lines.append("")
        return "\n".join(lines)

    lines.extend(
        [
            "| Domain | Case | torchfits | Winner | Lag ratio |",
            "|---|---|---|---:|---:|",
        ]
    )
    for row in rows[:max_rows]:
        case = row.get("case_label") or row.get("case_id") or "-"
        tf = row.get("torchfits_time_s") or "-"
        winner = f"{row.get('best_library', '-')}/{row.get('best_method', '-')}"
        lag = row.get("lag_ratio") or "-"
        lines.append(f"| {row.get('domain', '-')} | {case} | {tf} | {winner} | {lag} |")
    if len(rows) > max_rows:
        lines.append("")
        lines.append(
            f"_…and {len(rows) - max_rows} more rows in `torchfits_deficits.csv`._"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=40)
    args = parser.parse_args()
    print(render_deficits(args.csv, max_rows=args.max_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
