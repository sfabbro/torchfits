#!/usr/bin/env python3
"""Render the bench-all ``results.csv`` into the side-by-side I/O Transport
x Backend markdown section for ``docs/benchmarks.md``.

Mapping rules (single source of truth):

  Backend (column index):
    * ``library == "torchfits"`` -> ``"torchfits"``  (the cfitsio C engine
      is exposed via the torchfits binding, so the 4th column documents
      that without re-measuring it as a separate backend).
    * ``library == "astropy"``    -> ``"astropy"``
    * ``library == "fitsio"``     -> ``"fitsio"``

  I/O transport (row index):
    * ``disk->CPU``         streamed read to CPU buffer (no mmap). This run
                           uses ``mmap_target = "on"`` for every measured
                           row, so no measured row lands here.
    * ``disk->RAM->CPU``    mmap -> OS page cache -> CPU tensor on host
                           RAM. The only transport populated by the
                           current CPU-only run.
    * ``disk->GPU``         directly to GPU memory (``device=cuda``).
                           Reserved pending ``pixi run -e bench-gpu
                           bench-gpu``.
    * ``disk->RAM->GPU``    mmap -> CPU RAM -> explicit GPU copy.
                           Reserved pending ``pixi run -e bench-gpu
                           bench-gpu``.

Cell text rules:
  * Populated cells are the median of comparable OK rows in the bucket,
    formatted as ``\\`xx.xx ms\\` (n=N)``.
  * Throughput (MB/s) is intentionally OMITTED: aggregating it across
    heterogeneous operations + payload sizes produces physically-
    impossible median rates (e.g. ``>10^8 MB/s`` for ``fitstable``).
  * ``disk->GPU`` / ``disk->RAM->GPU`` cells always render
    ``_pending bench-gpu_``.
  * ``disk->CPU`` cells always render
    ``_no measured row (this run is mmap-on)_`` for the measured backends.
  * ``fitsio`` cells in ``disk->RAM->CPU`` render
    ``— (rows skipped under ``strict_mmap_fairness``)`` (fitsio rows are
    present in the CSV but the mmap-fairness rule excludes them).
  * ``cfitsio`` (direct) column always renders
    ``— (engine exposed under ``torchfits``)`` for measured transports.

Usage::

    python scripts/render_bench_iopath_table.py
    python scripts/render_bench_iopath_table.py --csv <path>

Reads ``benchmarks_results/<run-dir>/results.csv`` and prints the
markdown body of the section to stdout. The HTML comment + cwd-aware
``Source:`` line keep the rendered section traceable to the CSV that
produced it.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

IO_PATHS = [
    "disk\u2192CPU",
    "disk\u2192RAM\u2192CPU",
    "disk\u2192GPU",
    "disk\u2192RAM\u2192GPU",
]
BACKENDS = ["torchfits", "astropy", "fitsio"]
DOMAINS = [("fits", "FITS image I/O"), ("fitstable", "FITS table I/O")]
GPU_TRANSPORTS = ("disk\u2192GPU", "disk\u2192RAM\u2192GPU")


def to_float(s: str | None) -> float | None:
    """Best-effort float coercion; returns ``None`` for empty / unparseable."""
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def backend_of(row: dict[str, str]) -> str | None:
    """Map ``library`` -> one of the three measured backends.

    Returns ``None`` for libraries ``bench-all`` doesn't aggregate here
    so the caller can drop them.
    """
    lib = row.get("library", "")
    if lib == "torchfits":
        return "torchfits"
    if lib == "astropy":
        return "astropy"
    if lib == "fitsio":
        return "fitsio"
    return None


def _metadata_dict(row: dict[str, str]) -> dict[str, str]:
    md = row.get("metadata") or ""
    if isinstance(md, dict):
        return {str(k): str(v) for k, v in md.items()}
    if not isinstance(md, str) or not md.strip():
        return {}
    try:
        import ast

        parsed = ast.literal_eval(md.strip())
        if isinstance(parsed, dict):
            return {str(k): str(v) for k, v in parsed.items()}
    except Exception:
        pass
    try:
        import json

        parsed = json.loads(md)
        if isinstance(parsed, dict):
            return {str(k): str(v) for k, v in parsed.items()}
    except Exception:
        return {}
    return {}


def io_path_of(row: dict[str, str]) -> str | None:
    """Map an OK row onto one of the four I/O transports."""
    if row.get("status") != "OK":
        return None
    md = _metadata_dict(row)
    explicit = md.get("io_transport")
    if explicit:
        return explicit
    return "disk\u2192RAM\u2192CPU"


def fmt_measured_cell(
    samples: list[tuple[float, float | None, float | None, str]],
) -> str:
    """Format the populated cell as ``\\`xx.xx ms\\` (n=N)``.

    Throughput is intentionally omitted -- see module docstring for why.
    """
    if not samples:
        return "\u2014"
    med_t = statistics.median(t for (t, _tp, _sz, _op) in samples)
    return f"`{med_t * 1000:.2f} ms` (n={len(samples)})"


def cell_text(
    domain: str,
    transport: str,
    backend: str,
    samples: list[tuple[float, float | None, float | None, str]],
) -> str:
    """Decide the cell text from the (domain, transport, backend) bucket."""
    if transport == "disk\u2192GPU":
        return "\u2014"
    if domain == "fitstable" and transport in GPU_TRANSPORTS:
        return "\u2014"
    if transport == "disk\u2192RAM\u2192GPU":
        if samples:
            return fmt_measured_cell(samples)
        return "\u2014"
    if transport == "disk\u2192CPU":
        return "_no measured row (this run is mmap-on)_"
    if backend == "fitsio":
        return "\u2014 (rows skipped under `strict_mmap_fairness`)"
    return fmt_measured_cell(samples)


def render_table(
    domain_key: str,
    domain_label: str,
    buckets: dict[
        tuple[str, str, str],
        list[tuple[float, float | None, float | None, str]],
    ],
) -> str:
    """Render one section of the side-by-side table for a single domain."""
    header_cells = [
        "I/O transport",
        "`torchfits` (libcfitsio)",
        "`astropy`",
        "`fitsio`",
        "`cfitsio` (direct)",
    ]
    lines = [f"### {domain_label} ({domain_key})", ""]
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("|---|" + "|".join("---:" for _ in header_cells[1:]) + "|")
    for transport in IO_PATHS:
        cells: list[str] = []
        for backend in BACKENDS:
            samples = buckets.get((domain_key, transport, backend), [])
            cells.append(cell_text(domain_key, transport, backend, samples))
        # cfitsio-direct column has no GPU support.
        if transport in GPU_TRANSPORTS:
            cells.append("\u2014")
        else:
            cells.append("\u2014 (engine exposed under `torchfits`)")
        lines.append(f"| `{transport}` | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def aggregate(
    csv_path: str,
) -> tuple[
    str,
    dict[tuple[str, str, str], list[tuple[float, float | None, float | None, str]]],
]:
    """Read a bench-all ``results.csv`` and return ``(run_dir, buckets)``.

    ``run_dir`` is the parent directory of the CSV (e.g.
    ``20260626_postfix_full_zero_deficit``); used to populate the rendered
    ``Source:`` line when present.
    """
    buckets: dict[
        tuple[str, str, str],
        list[tuple[float, float | None, float | None, str]],
    ] = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            backend = backend_of(row)
            io_path = io_path_of(row)
            if backend is None or io_path is None:
                continue
            time_s = to_float(row.get("time_s"))
            if time_s is None:
                continue
            throughput = to_float(row.get("throughput"))
            size_mb = to_float(row.get("size_mb"))
            buckets[(row.get("domain", ""), io_path, backend)].append(
                (time_s, throughput, size_mb, row.get("operation", ""))
            )
    run_dir = Path(csv_path).parent.name or "(unknown)"
    return run_dir, buckets


def render_callout() -> str:
    """Render the leading blockquote that says GPU columns are reserved."""
    return "\n".join(
        [
            "> **GPU columns (`disk\u2192GPU`, `disk\u2192RAM\u2192GPU`) "
            "will be populated by**",
            "> `pixi run -e bench-gpu bench-gpu`. **Cells presently marked**",
            "> `_pending bench-gpu_` **are deliberate reservations for that run, not",
            "> missing data.** Once `benchmarks_results/gpu_<id>/results.csv` lands,",
            "> re-run `scripts/render_bench_iopath_table.py` with that path to fill",
            "> the cells. This section is regenerated from CSV by the script \u2014 do",
            "> not hand-edit.",
            "",
        ]
    )


def render_source(run_dir: str, *, has_gpu: bool) -> str:
    """Render the trailing ``Source:`` paragraph that names the CSV."""
    gpu_note = "MPS/CUDA GPU transport rows included." if has_gpu else "CPU mmap run."
    return "\n".join(
        [
            f"Source: `benchmarks_results/{run_dir}/results.csv` ({gpu_note})",
            "Cell values are median wall-clock over all comparable OK rows in the",
            "`(domain \u00d7 I/O transport \u00d7 backend)` bucket; "
            "throughput is intentionally",
            "omitted because the cell aggregates heterogeneous payloads and would",
            "produce physically-impossible rates when small and large sizes are",
            "median-mixed. See `scripts/render_bench_iopath_table.py` for the",
            "aggregation rules.",
            "",
        ]
    )


def render_notes() -> str:
    """Render the appendix that explains the table layout."""
    return "\n".join(
        [
            "### Notes on the layout",
            "",
            "- Rows are **I/O transports** (`disk\u2192CPU`, "
            "`disk\u2192RAM\u2192CPU`, `disk\u2192GPU`, "
            "`disk\u2192RAM\u2192GPU`).",
            "- Columns are **backends** (`torchfits` / `astropy` / `fitsio` "
            "/ `cfitsio-direct`).",
            "- `cfitsio` is the C engine used by `torchfits`; no standalone "
            "`cfitsio`-only",
            "  benchmark row is generated by `bench-all`, so the cell is documented as",
            '  "engine exposed under `torchfits`".',
            "- Cell `n=` counts comparable OK rows in the bucket; "
            "`\u2014` indicates the",
            "  bucket is empty (no rows match, or rows were excluded under",
            "  `strict_mmap_fairness` in the original `bench-all` summary).",
            "- Median is computed over heterogeneous operations (`read_full`,",
            "  `cutout_100x100`, `header_read`, `predicate_filter`, `projection`,",
            "  `row_slice`, etc.) and payload sizes; treat the per-cell ms as a",
            "  coarse representative number, not a precise benchmark.",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Render a bench-all `results.csv` into the I/O Transport x "
            "Backend markdown section of docs/benchmarks.md."
        )
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to the bench-all `results.csv` (e.g. benchmarks_results/<run-id>/results.csv).",
    )
    args = parser.parse_args()
    run_dir, buckets = aggregate(args.csv)
    has_gpu = any(
        io in GPU_TRANSPORTS and times for (_, io, _), times in buckets.items() if times
    )
    print(render_source(run_dir, has_gpu=has_gpu))
    for domain_key, domain_label in DOMAINS:
        if any(k[0] == domain_key for k in buckets):
            print(render_table(domain_key, domain_label, buckets))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
