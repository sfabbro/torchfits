#!/usr/bin/env python3
"""Emit full-suite jobs and run pure-C CFITSIO microbench (vendored libcfitsio).

Covers every image + table fixture/operation used by the scorecard harness, with
op→API mapping implemented in benchmarks/cfitsio_direct/bench_cfitsio_direct.c.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import fitsio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.bench_fits_io import FITSBenchmarkSuite, _hdu_for_file_type  # noqa: E402
from benchmarks.bench_fitstable_io import (  # noqa: E402
    _build_cases,
    _choose_numeric_col,
)

DIRECT_DIR = ROOT / "benchmarks" / "cfitsio_direct"
DEFAULT_BUILD = ROOT / "build" / "cfitsio_direct"


def _build_binary(build_dir: Path) -> Path:
    build_dir.mkdir(parents=True, exist_ok=True)
    cmake = os.environ.get("CMAKE", "cmake")
    subprocess.run(
        [
            cmake,
            "-S",
            str(DIRECT_DIR),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        check=True,
        cwd=ROOT,
    )
    subprocess.run(
        [cmake, "--build", str(build_dir), "--target", "bench_cfitsio_direct", "-j"],
        check=True,
        cwd=ROOT,
    )
    exe = build_dir / "bench_cfitsio_direct"
    if exe.is_file():
        return exe
    candidates = [
        c
        for c in build_dir.rglob("bench_cfitsio_direct")
        if c.is_file() and os.access(c, os.X_OK)
    ]
    if not candidates:
        raise FileNotFoundError(f"bench_cfitsio_direct not found under {build_dir}")
    return candidates[0]


def _file_type_for_name(name: str) -> str:
    lowered = name.lower()
    if "compressed" in lowered:
        return "compressed"
    if "multi_mef" in lowered:
        return "multi_mef"
    if "mef" in lowered:
        return "mef"
    if "scaled" in lowered:
        return "scaled"
    return "image"


def _py_hdu_for_name(name: str) -> int:
    return _hdu_for_file_type(_file_type_for_name(name))


def _cfitsio_hdu(py_hdu: int) -> int:
    """Astropy/fitsio/torchfits 0-based → CFITSIO fits_movabs_hdu 1-based."""
    return int(py_hdu) + 1


def _emit_image_jobs(
    files: dict[str, Path], jobs: list[list[str]], *, coords_dir: Path
) -> None:
    for name, path in sorted(files.items()):
        p = str(path)
        hdu = str(_cfitsio_hdu(_py_hdu_for_name(name)))
        jobs.append([name, "read_full", p, hdu])
        jobs.append([name, "header_read", p, hdu])

    # Scorecard cutout_100x100 cases: fixed [100:200,100:200] on named HDUs.
    for name, py_hdu in (("multi_mef_10ext", 5), ("compressed_rice_1", 1)):
        path = files.get(name)
        if path is None:
            continue
        jobs.append(
            [
                name,
                "cutout",
                str(path),
                "100",
                "100",
                str(_cfitsio_hdu(py_hdu)),
                "100",
                "100",
            ]
        )

    # Scorecard repeated cutouts: same numpy RNG seed/geometry as bench_fits_io.
    path = files.get("medium_float32_2d")
    if path is not None:
        py_hdu = _py_hdu_for_name("medium_float32_2d")
        with fitsio.FITS(str(path)) as f:
            header = f[py_hdu].read_header()
            naxis1 = int(header.get("NAXIS1", 1024))
            naxis2 = int(header.get("NAXIS2", 1024))
        cutout_size = min(100, naxis1 // 2, naxis2 // 2)
        if cutout_size < 2:
            cutout_size = 2
        coords_rng = np.random.default_rng(42)
        coords_path = coords_dir / "repeated_cutouts_coords.txt"
        lines = []
        for _ in range(50):
            x1 = int(coords_rng.integers(0, max(1, naxis1 - cutout_size)))
            y1 = int(coords_rng.integers(0, max(1, naxis2 - cutout_size)))
            lines.append(f"{x1} {y1} {cutout_size} {cutout_size}")
        coords_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        jobs.append(
            [
                f"repeated_cutouts_50x_{cutout_size}x{cutout_size}",
                "cutout_rep",
                str(path),
                str(coords_path),
                str(_cfitsio_hdu(py_hdu)),
            ]
        )

    # Rice compressed repeated cutouts (same API path MegaCam pays: fits_read_subset).
    rice = files.get("compressed_rice_1")
    if rice is not None:
        py_hdu = 1
        with fitsio.FITS(str(rice)) as f:
            header = f[py_hdu].read_header()
            naxis1 = int(header.get("ZNAXIS1") or header.get("NAXIS1") or 0)
            naxis2 = int(header.get("ZNAXIS2") or header.get("NAXIS2") or 0)
        if naxis1 >= 32 and naxis2 >= 32:
            cutout_size = min(64, naxis1 // 4, naxis2 // 4)
            coords_rng = np.random.default_rng(7)
            coords_path = coords_dir / "compressed_rice_cutout_rep_coords.txt"
            lines = []
            for _ in range(40):
                x1 = int(coords_rng.integers(0, max(1, naxis1 - cutout_size)))
                y1 = int(coords_rng.integers(0, max(1, naxis2 - cutout_size)))
                lines.append(f"{x1} {y1} {cutout_size} {cutout_size}")
            coords_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            jobs.append(
                [
                    f"compressed_rice_1_cutout_rep_40x_{cutout_size}x{cutout_size}",
                    "cutout_rep",
                    str(rice),
                    str(coords_path),
                    str(_cfitsio_hdu(py_hdu)),
                ]
            )

    # Optional real MegaCam .fz (when fetched) — first file, first image HDU.
    megacam_dir = ROOT / "benchmarks_data" / "cfht_megacam"
    if megacam_dir.is_dir():
        megacam_files = sorted(
            p
            for p in megacam_dir.iterdir()
            if p.is_file()
            and p.name.lower().endswith((".fits", ".fits.fz", ".fits.gz"))
        )
        if megacam_files:
            megacam = megacam_files[0]
            with fitsio.FITS(str(megacam)) as handle:
                hdu_idx = None
                naxis1 = naxis2 = 0
                for idx in range(len(handle)):
                    header = handle[idx].read_header()
                    z1 = int(header.get("ZNAXIS1", 0) or 0)
                    z2 = int(header.get("ZNAXIS2", 0) or 0)
                    n1 = z1 or int(header.get("NAXIS1", 0) or 0)
                    n2 = z2 or int(header.get("NAXIS2", 0) or 0)
                    if n1 >= 16 and n2 >= 16:
                        hdu_idx, naxis1, naxis2 = idx, n1, n2
                        break
            if hdu_idx is not None:
                cutout_size = min(256, naxis1 // 4, naxis2 // 4)
                coords_rng = np.random.default_rng(hdu_idx + len(megacam.name))
                coords_path = coords_dir / "megacam_cutout_rep_coords.txt"
                lines = []
                for _ in range(40):
                    x1 = int(coords_rng.integers(0, max(1, naxis1 - cutout_size)))
                    y1 = int(coords_rng.integers(0, max(1, naxis2 - cutout_size)))
                    lines.append(f"{x1} {y1} {cutout_size} {cutout_size}")
                coords_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                jobs.append(
                    [
                        f"megacam_{megacam.stem}_hdu{hdu_idx}_cutout_rep",
                        "cutout_rep",
                        str(megacam),
                        str(coords_path),
                        str(_cfitsio_hdu(hdu_idx)),
                    ]
                )

    if "multi_mef_10ext" in files:
        jobs.append(
            [
                "multi_mef_10ext",
                "random_ext",
                str(files["multi_mef_10ext"]),
                "200",
            ]
        )


def _emit_table_jobs(cases: list[dict], jobs: list[list[str]]) -> None:
    for case in cases:
        if case.get("unsupported") or case.get("compressed"):
            continue
        name = str(case["name"])
        path = str(case["path"])
        nrows = int(case["nrows"])
        columns = list(case["columns"])
        schema = case.get("schema")
        num_col = _choose_numeric_col(columns, schema)
        proj_n = min(3, len(columns))
        slice_start = 1
        slice_n = min(10_000, max(100, nrows // 10))

        jobs.append([name, "table_read", path])
        jobs.append([name, "table_header", path])
        jobs.append([name, "table_proj", path, str(proj_n)])
        jobs.append([name, "table_slice", path, str(slice_start), str(slice_n)])
        jobs.append([name, "table_scan", path, num_col])
        jobs.append([name, "table_pred", path, num_col])


def _write_jobs(path: Path, jobs: list[list[str]]) -> None:
    lines = [
        "# case_id\top\tpath\t[a\tb\tc\td\te]",
        "# see benchmarks/cfitsio_direct/bench_cfitsio_direct.c header for op→API map",
    ]
    for row in jobs:
        lines.append("\t".join(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _count_job_rows(jobs_path: Path) -> int:
    n = 0
    for line in jobs_path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        n += 1
    return n


def _run_c(exe: Path, jobs_path: Path, csv_path: Path, runs: int, warmup: int) -> None:
    cmd = [
        str(exe),
        "--jobs",
        str(jobs_path),
        "--runs",
        str(runs),
        "--warmup",
        str(warmup),
        "--csv",
        str(csv_path),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(
            f"cfitsio_direct exited rc={proc.returncode}: {proc.stderr.strip()}"
        )
    if not csv_path.is_file() or csv_path.stat().st_size < 10:
        raise RuntimeError("cfitsio_direct produced empty/missing CSV")
    n_jobs = _count_job_rows(jobs_path)
    with csv_path.open(encoding="utf-8") as f:
        n_csv = sum(1 for _ in csv.DictReader(f))
    if n_csv != n_jobs:
        raise RuntimeError(
            f"cfitsio_direct CSV rows ({n_csv}) != jobs ({n_jobs}); "
            "crash/skip would previously look like success"
        )


def _summarize(csv_path: Path, summary_path: Path) -> dict[str, int]:
    by_op: dict[str, list[float]] = {}
    n_ok = n_err = 0
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "OK" and row.get("time_s"):
                n_ok += 1
                by_op.setdefault(row["operation"], []).append(float(row["time_s"]))
            else:
                n_err += 1
    lines = [
        "# cfitsio_direct full-suite summary (median seconds geo-ish: median of case medians)",
        "operation,n_ok,median_s,p90_s",
    ]
    for op, xs in sorted(by_op.items()):
        xs = sorted(xs)
        med = xs[len(xs) // 2]
        p90 = xs[int(0.9 * (len(xs) - 1))]
        lines.append(f"{op},{len(xs)},{med:.9g},{p90:.9g}")
    lines.append(f"# totals ok={n_ok} error={n_err}")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"ok": n_ok, "error": n_err, "ops": len(by_op)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default=time.strftime("cfitsio_direct_%Y%m%d_%H%M%S"))
    ap.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument(
        "--profile",
        choices=("full", "gpu_core", "quick"),
        default="full",
        help="full=scorecard fixtures; gpu_core=smaller images; quick=smaller tables",
    )
    ap.add_argument("--images-only", action="store_true")
    ap.add_argument("--tables-only", action="store_true")
    ap.add_argument("--keep-temp", action="store_true")
    args = ap.parse_args()

    exe = _build_binary(args.build_dir)
    out_dir = ROOT / "benchmarks_results" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp = Path(tempfile.mkdtemp(prefix="torchfits_cfitsio_direct_"))
    try:
        jobs: list[list[str]] = []
        if not args.tables_only:
            fixture_profile = "gpu_core" if args.profile == "gpu_core" else "full"
            suite = FITSBenchmarkSuite(
                output_dir=tmp / "images", use_mmap=False, profile="lab"
            )
            files = suite.create_test_files(fixture_profile=fixture_profile)
            _emit_image_jobs(files, jobs, coords_dir=out_dir)

        if not args.images_only:
            tables_dir = tmp / "tables"
            tables_dir.mkdir(parents=True, exist_ok=True)
            table_cases = _build_cases(tables_dir, quick=(args.profile == "quick"))
            _emit_table_jobs(table_cases, jobs)

        jobs_path = out_dir / "jobs.tsv"
        _write_jobs(jobs_path, jobs)
        print(f"jobs={len(jobs)} written {jobs_path}", flush=True)

        csv_path = out_dir / "cfitsio_direct.csv"
        _run_c(exe, jobs_path, csv_path, args.runs, args.warmup)
        stats = _summarize(csv_path, out_dir / "cfitsio_direct_summary.csv")
        print(f"wrote {csv_path}")
        print(f"wrote {out_dir / 'cfitsio_direct_summary.csv'}")
        print((out_dir / "cfitsio_direct_summary.csv").read_text(encoding="utf-8"))

        # These are the only checks that the C benchmark actually produced usable
        # numbers, and they must not be `assert`: under `python -O` (or
        # PYTHONOPTIMIZE=1) an assert statement is removed from the bytecode
        # entirely -- compiling this function at optimize=2 reduces it to a bare
        # `return 0` -- so a run that produced three rows, or one ERROR row, would
        # exit 0 and publish `cfitsio_direct.csv` as a cross-library comparison.
        problems = []
        if stats["ok"] < 50:
            problems.append(f"expected >=50 OK rows, got {stats['ok']}")
        if stats["ops"] < 5:
            problems.append(f"expected broadly covered ops, got {stats['ops']}")
        if stats["error"] != 0:
            problems.append(f"expected zero ERROR rows, got {stats['error']}")
        if problems:
            raise SystemExit(
                "cfitsio_direct benchmark validation failed: " + "; ".join(problems)
            )
        return 0
    finally:
        if not args.keep_temp:
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            print(f"kept temp {tmp}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
