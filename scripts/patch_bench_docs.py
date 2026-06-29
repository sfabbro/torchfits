#!/usr/bin/env python3
"""Splice rendered benchmark sections into docs/benchmarks.md."""

from __future__ import annotations

import argparse
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
        rf"({re.escape(begin)})\n.*?\n({re.escape(end)})",
        re.DOTALL,
    )
    if not pattern.search(text):
        raise SystemExit(f"missing markers {begin!r} .. {end!r} in docs")
    return pattern.sub(rf"\1\n{body.rstrip()}\n\2", text, count=1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, default=_REPO / "docs" / "benchmarks.md")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--deficits", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    iopath = _render("render_bench_iopath_table.py", "--csv", str(args.csv))
    deficits = _render("render_bench_deficits.py", "--csv", str(args.deficits))
    if deficits.startswith("## Performance deficits"):
        deficits = deficits.split("\n", 2)[-1] if "\n\n" in deficits else ""
        if deficits.startswith("\n"):
            deficits = deficits[1:]
    snapshot = (
        f"| `{args.run_id}` | fits + fitstable (lab) | "
        f"(see CSV) | (see deficits CSV) | CI weekly bench-all |\n"
    )

    text = args.docs.read_text(encoding="utf-8")
    text = _replace_block(text, "<!-- BENCH_IOPATH_BEGIN -->", "<!-- BENCH_IOPATH_END -->", iopath)
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
