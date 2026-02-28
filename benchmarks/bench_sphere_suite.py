#!/usr/bin/env python3
"""Sphere benchmark domain orchestrator (geometry/advanced/sparse/spectral/polygon/core)."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from bench_contract import RESULT_COLUMNS, annotate_rankings, write_csv


REQUIRED_COMPARATORS = ["healpy", "hpgeom", "astropy-healpix", "healsparse"]


def _run_json_command(name: str, cmd: list[str], out_json: Path) -> tuple[bool, list[dict[str, Any]], str]:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        return False, [], f"{name} failed (code {proc.returncode}): {proc.stdout[-400:]}"
    try:
        rows = json.loads(out_json.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            return False, [], f"{name} output is not a JSON list"
        return True, rows, ""
    except Exception as exc:
        return False, [], f"{name} JSON parse failed: {exc}"


def _safe_float(v: Any) -> float | None:
    try:
        x = float(v)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return x


def _row(
    *,
    run_id: str,
    suite: str,
    case_id: str,
    case_label: str,
    operation: str,
    family: str,
    library: str,
    method: str,
    mode: str,
    status: str,
    skip_reason: str,
    comparable: bool,
    time_s: float | None,
    throughput: float | None,
    unit: str,
    n_points: int | str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "domain": "sphere",
        "suite": suite,
        "case_id": case_id,
        "case_label": case_label,
        "operation": operation,
        "family": family,
        "library": library,
        "method": method,
        "mode": mode,
        "status": status,
        "skip_reason": skip_reason,
        "comparable": comparable,
        "mmap_target": "-",
        "time_s": time_s,
        "throughput": throughput,
        "unit": unit,
        "size_mb": "",
        "n_points": n_points,
        "metadata": metadata,
    }


def run_sphere_domain(*, run_id: str, output_dir: Path, include_gpu: bool = True) -> list[dict[str, Any]]:
    raw_dir = output_dir / "_raw" / "sphere"
    raw_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    # 1) Geometry / projection operations across ecosystem libs (CPU baseline).
    print("[sphere] running geometry CPU suite...", flush=True)
    geo_json = raw_dir / "sphere_geometry_cpu.json"
    geo_cmd = [
        sys.executable,
        str(Path(__file__).with_name("bench_sphere_geometry.py")),
        "--device",
        "cpu",
        "--sample-profile",
        "mixed",
        "--nside",
        "1024",
        "--n-points",
        "200000",
        "--runs",
        "5",
        "--libraries",
        "torchfits,healpy,hpgeom,astropy-healpix,healpix,mhealpy",
        "--json-out",
        str(geo_json),
    ]
    ok, geo_rows, reason = _run_json_command("sphere_geometry_cpu", geo_cmd, geo_json)
    if not ok:
        rows.append(
            _row(
                run_id=run_id,
                suite="sphere_geometry",
                case_id="sphere_geometry::cpu",
                case_label="sphere geometry cpu",
                operation="geometry",
                family="specialized",
                library="torchfits",
                method="torchfits",
                mode="specialized",
                status="FAILED",
                skip_reason=reason,
                comparable=False,
                time_s=None,
                throughput=None,
                unit="Mpts/s",
                n_points=200000,
                metadata={},
            )
        )
    else:
        seen_ops: dict[str, set[str]] = {}
        for r in geo_rows:
            lib = str(r.get("library", "unknown"))
            op = str(r.get("operation", "unknown"))
            n_points = int(r.get("n_points", 0) or 0)
            t = _safe_float(r.get("ms"))
            time_s = t / 1000.0 if t is not None else None
            thr = _safe_float(r.get("mpts_s"))
            seen_ops.setdefault(op, set()).add(lib)
            rows.append(
                _row(
                    run_id=run_id,
                    suite="sphere_geometry",
                    case_id=f"geometry::{op}::cpu",
                    case_label=f"geometry {op} cpu",
                    operation=op,
                    family="specialized",
                    library=lib,
                    method=lib,
                    mode="specialized",
                    status="OK" if time_s is not None else "FAILED",
                    skip_reason="",
                    comparable=time_s is not None,
                    time_s=time_s,
                    throughput=thr,
                    unit="Mpts/s",
                    n_points=n_points,
                    metadata={
                        "device": r.get("device"),
                        "sample_profile": r.get("sample_profile"),
                        "mismatches": r.get("mismatches"),
                        "max_dra_deg": r.get("max_dra_deg"),
                        "max_ddec_deg": r.get("max_ddec_deg"),
                    },
                )
            )

        # Explicitly mark required ecosystem comparators if not present.
        for op, libs in seen_ops.items():
            for comp in REQUIRED_COMPARATORS:
                if comp in libs:
                    continue
                rows.append(
                    _row(
                        run_id=run_id,
                        suite="sphere_geometry",
                        case_id=f"geometry::{op}::cpu::{comp}",
                        case_label=f"geometry {op} cpu",
                        operation=op,
                        family="specialized",
                        library=comp,
                        method=comp,
                        mode="specialized",
                        status="SKIPPED",
                        skip_reason="comparator_unavailable",
                        comparable=False,
                        time_s=None,
                        throughput=None,
                        unit="Mpts/s",
                        n_points=200000,
                        metadata={"required_comparator": True},
                    )
                )

    # Optional GPU subsection.
    if include_gpu and (torch.cuda.is_available() or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())):
        gpu_dev = "cuda" if torch.cuda.is_available() else "mps"
        print(f"[sphere] running geometry {gpu_dev} suite...", flush=True)
        geo_gpu_json = raw_dir / f"sphere_geometry_{gpu_dev}.json"
        geo_gpu_cmd = [
            sys.executable,
            str(Path(__file__).with_name("bench_sphere_geometry.py")),
            "--device",
            gpu_dev,
            "--sample-profile",
            "mixed",
            "--nside",
            "1024",
            "--n-points",
            "200000",
            "--runs",
            "3",
            "--libraries",
            "torchfits",
            "--json-out",
            str(geo_gpu_json),
        ]
        ok_gpu, gpu_rows, reason_gpu = _run_json_command(
            f"sphere_geometry_{gpu_dev}", geo_gpu_cmd, geo_gpu_json
        )
        if ok_gpu:
            for r in gpu_rows:
                op = str(r.get("operation", "unknown"))
                t = _safe_float(r.get("ms"))
                time_s = t / 1000.0 if t is not None else None
                rows.append(
                    _row(
                        run_id=run_id,
                        suite="sphere_geometry",
                        case_id=f"geometry::{op}::{gpu_dev}",
                        case_label=f"geometry {op} {gpu_dev}",
                        operation=op,
                        family="specialized",
                        library="torchfits",
                        method="torchfits",
                        mode="specialized",
                        status="OK" if time_s is not None else "FAILED",
                        skip_reason="",
                        comparable=False,
                        time_s=time_s,
                        throughput=_safe_float(r.get("mpts_s")),
                        unit="Mpts/s",
                        n_points=int(r.get("n_points", 0) or 0),
                        metadata={"device": gpu_dev, "gpu_section": True},
                    )
                )
        else:
            rows.append(
                _row(
                    run_id=run_id,
                    suite="sphere_geometry",
                    case_id=f"geometry::gpu::{gpu_dev}",
                    case_label=f"geometry {gpu_dev}",
                    operation="geometry_gpu",
                    family="specialized",
                    library="torchfits",
                    method="torchfits",
                    mode="specialized",
                    status="SKIPPED",
                    skip_reason=reason_gpu,
                    comparable=False,
                    time_s=None,
                    throughput=None,
                    unit="Mpts/s",
                    n_points=200000,
                    metadata={"gpu_section": True},
                )
            )

    # 2) Advanced HEALPix operations.
    print("[sphere] running advanced HEALPix suite...", flush=True)
    adv_json = raw_dir / "healpix_advanced.json"
    adv_cmd = [
        sys.executable,
        str(Path(__file__).with_name("bench_healpix_advanced.py")),
        "--device",
        "cpu",
        "--nside",
        "1024",
        "--n-points",
        "200000",
        "--runs",
        "5",
        "--json-out",
        str(adv_json),
    ]
    ok_adv, adv_rows, reason_adv = _run_json_command("healpix_advanced", adv_cmd, adv_json)
    if ok_adv:
        for r in adv_rows:
            op = str(r.get("operation", "unknown"))
            tf_t = _safe_float(r.get("torch_ms"))
            hp_t = _safe_float(r.get("healpy_ms"))
            rows.append(
                _row(
                    run_id=run_id,
                    suite="healpix_advanced",
                    case_id=f"advanced::{op}",
                    case_label=f"advanced {op}",
                    operation=op,
                    family="specialized",
                    library="torchfits",
                    method="torchfits",
                    mode="specialized",
                    status="OK" if tf_t is not None else "FAILED",
                    skip_reason="",
                    comparable=tf_t is not None,
                    time_s=(tf_t / 1000.0) if tf_t is not None else None,
                    throughput=_safe_float(r.get("torch_mpts_s")),
                    unit="Mpts/s",
                    n_points=int(r.get("n_points", 0) or 0),
                    metadata={"mismatches": r.get("mismatches")},
                )
            )
            rows.append(
                _row(
                    run_id=run_id,
                    suite="healpix_advanced",
                    case_id=f"advanced::{op}",
                    case_label=f"advanced {op}",
                    operation=op,
                    family="specialized",
                    library="healpy",
                    method="healpy",
                    mode="specialized",
                    status="OK" if hp_t is not None else "FAILED",
                    skip_reason="",
                    comparable=hp_t is not None,
                    time_s=(hp_t / 1000.0) if hp_t is not None else None,
                    throughput=_safe_float(r.get("healpy_mpts_s")),
                    unit="Mpts/s",
                    n_points=int(r.get("n_points", 0) or 0),
                    metadata={"mismatches": r.get("mismatches")},
                )
            )
    else:
        rows.append(
            _row(
                run_id=run_id,
                suite="healpix_advanced",
                case_id="advanced::all",
                case_label="advanced ops",
                operation="advanced_ops",
                family="specialized",
                library="torchfits",
                method="torchfits",
                mode="specialized",
                status="FAILED",
                skip_reason=reason_adv,
                comparable=False,
                time_s=None,
                throughput=None,
                unit="Mpts/s",
                n_points=200000,
                metadata={},
            )
        )

    # 3) Sparse operations.
    print("[sphere] running sparse HEALPix suite...", flush=True)
    sparse_json = raw_dir / "sphere_sparse.json"
    sparse_cmd = [
        sys.executable,
        str(Path(__file__).with_name("bench_sphere_sparse.py")),
        "--nside",
        "512",
        "--coverage-frac",
        "0.1",
        "--n-queries",
        "200000",
        "--runs",
        "5",
        "--json-out",
        str(sparse_json),
    ]
    ok_sparse, sparse_rows, reason_sparse = _run_json_command("sphere_sparse", sparse_cmd, sparse_json)
    if ok_sparse:
        for r in sparse_rows:
            op = str(r.get("operation", "unknown"))
            n_queries = int(r.get("n_queries", 0) or 0)
            qps = _safe_float(r.get("qps"))
            if qps is not None and qps > 0 and n_queries > 0:
                t = n_queries / qps
                rows.append(
                    _row(
                        run_id=run_id,
                        suite="sphere_sparse",
                        case_id=f"sparse::{op}",
                        case_label=f"sparse {op}",
                        operation=op,
                        family="specialized",
                        library="torchfits",
                        method="torchfits",
                        mode="specialized",
                        status="OK",
                        skip_reason="",
                        comparable=False,
                        time_s=t,
                        throughput=qps,
                        unit="rows/s",
                        n_points=n_queries,
                        metadata={"coverage_frac": r.get("coverage_frac")},
                    )
                )
            elif op == "ud_grade_sparse_vs_dense":
                sparse_s = _safe_float(r.get("sparse_s"))
                dense_s = _safe_float(r.get("dense_s"))
                rows.append(
                    _row(
                        run_id=run_id,
                        suite="sphere_sparse",
                        case_id="sparse::ud_grade",
                        case_label="sparse ud_grade",
                        operation="ud_grade",
                        family="specialized",
                        library="torchfits",
                        method="torchfits",
                        mode="specialized",
                        status="OK" if sparse_s is not None else "FAILED",
                        skip_reason="",
                        comparable=bool(sparse_s is not None and dense_s is not None),
                        time_s=sparse_s,
                        throughput=(1.0 / sparse_s) if sparse_s else None,
                        unit="ops/s",
                        n_points="",
                        metadata={"ratio_sparse_vs_dense": r.get("ratio_sparse_vs_dense")},
                    )
                )
                rows.append(
                    _row(
                        run_id=run_id,
                        suite="sphere_sparse",
                        case_id="sparse::ud_grade",
                        case_label="sparse ud_grade",
                        operation="ud_grade",
                        family="specialized",
                        library="dense_cpu_baseline",
                        method="dense_cpu_baseline",
                        mode="specialized",
                        status="OK" if dense_s is not None else "FAILED",
                        skip_reason="",
                        comparable=bool(sparse_s is not None and dense_s is not None),
                        time_s=dense_s,
                        throughput=(1.0 / dense_s) if dense_s else None,
                        unit="ops/s",
                        n_points="",
                        metadata={"ratio_sparse_vs_dense": r.get("ratio_sparse_vs_dense")},
                    )
                )
    else:
        rows.append(
            _row(
                run_id=run_id,
                suite="sphere_sparse",
                case_id="sparse::all",
                case_label="sparse all",
                operation="sparse",
                family="specialized",
                library="torchfits",
                method="torchfits",
                mode="specialized",
                status="FAILED",
                skip_reason=reason_sparse,
                comparable=False,
                time_s=None,
                throughput=None,
                unit="rows/s",
                n_points=200000,
                metadata={},
            )
        )

    # 4) Spectral operations.
    print("[sphere] running spectral sphere suite...", flush=True)
    spec_json = raw_dir / "sphere_spectral.json"
    spec_cmd = [
        sys.executable,
        str(Path(__file__).with_name("bench_sphere_spectral.py")),
        "--nside",
        "32",
        "--lmax",
        "32",
        "--runs",
        "5",
        "--json-out",
        str(spec_json),
    ]
    ok_spec, spec_rows, reason_spec = _run_json_command("sphere_spectral", spec_cmd, spec_json)
    if ok_spec:
        for r in spec_rows:
            backend = str(r.get("backend", "unknown"))
            lib = "torchfits" if backend.startswith("torch") else backend
            op = str(r.get("op", "unknown"))
            t = _safe_float(r.get("time_s"))
            mops = _safe_float(r.get("mops_s"))
            rows.append(
                _row(
                    run_id=run_id,
                    suite="sphere_spectral",
                    case_id=f"spectral::{op}",
                    case_label=f"spectral {op}",
                    operation=op,
                    family="specialized",
                    library=lib,
                    method=backend,
                    mode="specialized",
                    status="OK" if t is not None else "FAILED",
                    skip_reason="",
                    comparable=t is not None,
                    time_s=t,
                    throughput=mops,
                    unit="Mops/s",
                    n_points=int(r.get("n", 0) or 0),
                    metadata={"nside": r.get("nside"), "lmax": r.get("lmax")},
                )
            )
    else:
        rows.append(
            _row(
                run_id=run_id,
                suite="sphere_spectral",
                case_id="spectral::all",
                case_label="spectral all",
                operation="spectral",
                family="specialized",
                library="torchfits",
                method="torchfits",
                mode="specialized",
                status="FAILED",
                skip_reason=reason_spec,
                comparable=False,
                time_s=None,
                throughput=None,
                unit="Mops/s",
                n_points="",
                metadata={},
            )
        )

    # 5) Polygon operations.
    print("[sphere] running polygon suite...", flush=True)
    poly_json = raw_dir / "sphere_polygons.json"
    poly_cmd = [
        sys.executable,
        str(Path(__file__).with_name("bench_sphere_polygons.py")),
        "--nside",
        "512",
        "--n-points",
        "200000",
        "--runs",
        "5",
        "--json-out",
        str(poly_json),
    ]
    ok_poly, poly_rows, reason_poly = _run_json_command("sphere_polygons", poly_cmd, poly_json)
    if ok_poly:
        for r in poly_rows:
            op = str(r.get("operation", "unknown"))
            lib = str(r.get("library", "torchfits"))
            n_points = int(r.get("n_points", 0) or 0)
            t = None
            thr = None
            unit = "ops/s"
            if _safe_float(r.get("points_s")):
                p = float(r["points_s"])
                thr = p
                unit = "points/s"
                if n_points > 0 and p > 0:
                    t = n_points / p
            elif _safe_float(r.get("queries_s")):
                q = float(r["queries_s"])
                thr = q
                unit = "queries/s"
                if q > 0:
                    t = 1.0 / q
            elif _safe_float(r.get("calls_s")):
                c = float(r["calls_s"])
                thr = c
                unit = "calls/s"
                if c > 0:
                    t = 1.0 / c

            rows.append(
                _row(
                    run_id=run_id,
                    suite="sphere_polygons",
                    case_id=f"polygon::{op}",
                    case_label=f"polygon {op}",
                    operation=op,
                    family="specialized",
                    library=lib,
                    method=lib,
                    mode="specialized",
                    status="OK" if t is not None else "SKIPPED",
                    skip_reason="no_timing_field" if t is None else "",
                    comparable=t is not None,
                    time_s=t,
                    throughput=thr,
                    unit=unit,
                    n_points=n_points,
                    metadata={
                        "mismatches_vs_spherical_geometry": r.get(
                            "mismatches_vs_spherical_geometry"
                        ),
                        "relative_error_vs_torchfits": r.get(
                            "relative_error_vs_torchfits"
                        ),
                    },
                )
            )
    else:
        rows.append(
            _row(
                run_id=run_id,
                suite="sphere_polygons",
                case_id="polygon::all",
                case_label="polygon all",
                operation="polygon",
                family="specialized",
                library="torchfits",
                method="torchfits",
                mode="specialized",
                status="FAILED",
                skip_reason=reason_poly,
                comparable=False,
                time_s=None,
                throughput=None,
                unit="ops/s",
                n_points=200000,
                metadata={},
            )
        )

    # 6) Sphere core operations.
    print("[sphere] running sphere core suite...", flush=True)
    core_json = raw_dir / "sphere_core.json"
    core_cmd = [
        sys.executable,
        str(Path(__file__).with_name("bench_sphere_core.py")),
        "--device",
        "cpu",
        "--nside",
        "256",
        "--n-points",
        "100000",
        "--n-bands",
        "8",
        "--runs",
        "5",
        "--json-out",
        str(core_json),
    ]
    ok_core, core_rows, reason_core = _run_json_command("sphere_core", core_cmd, core_json)
    if ok_core:
        for r in core_rows:
            op = str(r.get("operation", "unknown"))
            n_points = int(r.get("n_points", 0) or 0)
            t = None
            thr = None
            unit = "ops/s"
            if _safe_float(r.get("m_samples_s")):
                ms = float(r["m_samples_s"])
                thr = ms
                unit = "Msamples/s"
                n_bands = int(r.get("n_bands", 1) or 1)
                denom = ms * 1e6
                if denom > 0 and n_points > 0 and n_bands > 0:
                    t = (n_points * n_bands) / denom
            elif _safe_float(r.get("m_pairwise_s")):
                mp = float(r["m_pairwise_s"])
                thr = mp
                unit = "Mpairs/s"
            elif _safe_float(r.get("queries_s")):
                qs = float(r["queries_s"])
                thr = qs
                unit = "queries/s"
                if qs > 0:
                    t = 1.0 / qs

            rows.append(
                _row(
                    run_id=run_id,
                    suite="sphere_core",
                    case_id=f"core::{op}",
                    case_label=f"core {op}",
                    operation=op,
                    family="specialized",
                    library="torchfits",
                    method="torchfits",
                    mode="specialized",
                    status="OK" if (t is not None or thr is not None) else "SKIPPED",
                    skip_reason="no_timing_field" if (t is None and thr is None) else "",
                    comparable=False,
                    time_s=t,
                    throughput=thr,
                    unit=unit,
                    n_points=n_points,
                    metadata={"device": r.get("device")},
                )
            )
    else:
        rows.append(
            _row(
                run_id=run_id,
                suite="sphere_core",
                case_id="core::all",
                case_label="core all",
                operation="core",
                family="specialized",
                library="torchfits",
                method="torchfits",
                mode="specialized",
                status="FAILED",
                skip_reason=reason_core,
                comparable=False,
                time_s=None,
                throughput=None,
                unit="ops/s",
                n_points="",
                metadata={},
            )
        )

    annotate_rankings(rows)
    print(f"[sphere] normalized rows={len(rows)}", flush=True)
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks_results"))
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--no-gpu", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_id = args.run_id.strip() or time.strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_id

    rows = run_sphere_domain(run_id=run_id, output_dir=run_dir, include_gpu=not args.no_gpu)
    out_csv = run_dir / "sphere_results.csv"
    write_csv(out_csv, rows, RESULT_COLUMNS)
    print(f"[sphere] wrote {len(rows)} rows to {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
