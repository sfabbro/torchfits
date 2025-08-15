#!/usr/bin/env python3
"""
Plot summary figures from the benchmark JSONL produced by compare_readers.py.

Generates:
- Per-scenario bar plots (impl+api vs mean ms) for each scenario+key combo
- Trend plots:
  * image_full: size vs mean ms (best per impl)
  * cutouts_random: cut_hw vs mean ms (best per impl)
  * sky_cutouts: radius_arcsec vs mean ms (best per impl)
  * table_cols: rows vs mean ms (best per impl)

Usage:
  python benchmarks/plot_full_sweep.py artifacts/benchmarks/full_sweep.jsonl --outdir artifacts/benchmarks/plots
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Sequence, Mapping, cast

# Import matplotlib if available; fall back to a typed Any placeholder.
try:  # pragma: no cover - import presence depends on env
    import matplotlib.pyplot as plt  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    plt = cast(Any, None)


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def _group_key(rec: Dict[str, Any]) -> str:
    parts = [rec.get("scenario", "")]  # type: ignore
    for dim in ("size", "rows", "cutouts", "cut_hw", "hdus", "files", "radius_arcsec"):
        if dim in rec:
            parts.append(f"{dim}={rec[dim]}")
    return " | ".join(parts)


def _best_per_impl(items: List[Dict[str, Any]]) -> Dict[str, Tuple[float, Dict[str, Any]]]:
    best: Dict[str, Tuple[float, Dict[str, Any]]] = {}
    for r in items:
        impl = str(r.get("impl"))
        ms = float(r.get("mean_ms", 1e30))
        cur = best.get(impl)
        if cur is None or ms < cur[0]:
            best[impl] = (ms, r)
    return best


def _ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def plot_bars_per_group(rows: List[Dict[str, Any]], outdir: str) -> List[str]:
    if plt is None:
        return []
    groups = defaultdict(list)
    for r in rows:
        if r.get("scenario") == "meta":
            continue
        groups[_group_key(r)].append(r)

    _ensure_dir(outdir)
    paths: List[str] = []
    for gk, items in groups.items():
        xs: List[str] = []
        ys: List[float] = []
        for r in items:
            xs.append(f"{r.get('impl')}-{r.get('api')}")
            ys.append(float(r.get("mean_ms", 0)))
        if not xs:
            continue
        order = sorted(range(len(xs)), key=lambda i: ys[i])
        xs = [xs[i] for i in order]
        ys = [ys[i] for i in order]
        p = cast(Any, plt)
        p.figure(figsize=(8, 3 + max(1, len(xs) * 0.15)))
        p.barh(range(len(xs)), ys, color="#3b82f6")
        p.yticks(range(len(xs)), xs)
        p.xlabel("mean ms")
        p.title(gk)
        p.tight_layout()
        fname = os.path.join(outdir, gk.replace(" ", "_").replace("/", "-") + ".png")
        p.savefig(fname)
        p.close()
        paths.append(fname)
    return paths


def plot_trends(rows: List[Dict[str, Any]], outdir: str) -> List[str]:
    if plt is None:
        return []
    _ensure_dir(outdir)
    out: List[str] = []

    def lineplot(title: str, xs: Sequence[float | int], series: Mapping[str, Sequence[float | int]], xlabel: str) -> str:
        p = cast(Any, plt)
        p.figure(figsize=(7, 4))
        xs_f = [float(x) for x in xs]
        for label, ys in series.items():
            ys_f = [float(y) for y in ys]
            p.plot(xs_f, ys_f, marker="o", label=label)
        p.title(title)
        p.xlabel(xlabel)
        p.ylabel("mean ms (best per impl)")
        p.legend()
        p.grid(True, alpha=0.3)
        p.tight_layout()
        fname = os.path.join(outdir, title.lower().replace(" ", "_") + ".png")
        p.savefig(fname)
        p.close()
        return fname

    # image_full: size trend
    img = [r for r in rows if r.get("scenario") == "image_full"]
    if img:
        sizes = sorted({int(r.get("size", 0)) for r in img})
        impls = sorted({str(r.get("impl")) for r in img})
        series_img: Dict[str, List[float]] = {impl: [] for impl in impls}
        for sz in sizes:
            items = [r for r in img if int(r.get("size", 0)) == sz]
            best = _best_per_impl(items)
            for impl in impls:
                series_img[impl].append(best.get(impl, (float("nan"), {}))[0])
        out.append(lineplot("image_full trend", sizes, cast(Mapping[str, Sequence[float | int]], series_img), "size (pixels)"))

    # cutouts_random: cut_hw trend
    co = [r for r in rows if r.get("scenario") == "cutouts_random"]
    if co:
        cut_hws = sorted({int(r.get("cut_hw", 0)) for r in co})
        impls = sorted({str(r.get("impl")) for r in co})
        series_co: Dict[str, List[float]] = {impl: [] for impl in impls}
        for hw in cut_hws:
            items = [r for r in co if int(r.get("cut_hw", 0)) == hw]
            best = _best_per_impl(items)
            for impl in impls:
                series_co[impl].append(best.get(impl, (float("nan"), {}))[0])
        out.append(lineplot("cutouts_random trend", cut_hws, cast(Mapping[str, Sequence[float | int]], series_co), "cutout half-width (pixels)"))

    # sky_cutouts: radius trend
    sky = [r for r in rows if r.get("scenario") == "sky_cutouts"]
    if sky:
        radii = sorted({float(r.get("radius_arcsec", 0.0)) for r in sky})
        impls = sorted({str(r.get("impl")) for r in sky})
        series_sky: Dict[str, List[float]] = {impl: [] for impl in impls}
        for rad in radii:
            items = [r for r in sky if abs(float(r.get("radius_arcsec", 0.0)) - rad) < 1e-9]
            best = _best_per_impl(items)
            for impl in impls:
                series_sky[impl].append(best.get(impl, (float("nan"), {}))[0])
        out.append(lineplot("sky_cutouts trend", [int(r) for r in radii], cast(Mapping[str, Sequence[float | int]], series_sky), "radius (arcsec)"))

    # table_cols: rows trend (if multiple rows present)
    tcols = [r for r in rows if r.get("scenario") == "table_cols"]
    if tcols:
        nrows = sorted({int(r.get("rows", 0)) for r in tcols})
        if len(nrows) > 1:
            impls = sorted({str(r.get("impl")) for r in tcols})
            series_tbl: Dict[str, List[float]] = {impl: [] for impl in impls}
            for rn in nrows:
                items = [r for r in tcols if int(r.get("rows", 0)) == rn]
                best = _best_per_impl(items)
                for impl in impls:
                    series_tbl[impl].append(best.get(impl, (float("nan"), {}))[0])
            out.append(lineplot("table_cols trend", nrows, cast(Mapping[str, Sequence[float | int]], series_tbl), "rows"))

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=str, help="Path to full_sweep JSONL")
    ap.add_argument("--outdir", type=str, default="artifacts/benchmarks/plots", help="Directory to write plots")
    args = ap.parse_args()

    rows = _load_jsonl(args.jsonl)
    if not rows:
        print("No rows loaded from", args.jsonl)
        return

    if plt is None:
        print("matplotlib missing; cannot plot")
        return

    os.makedirs(args.outdir, exist_ok=True)
    bar_paths = plot_bars_per_group(rows, args.outdir)
    trend_paths = plot_trends(rows, args.outdir)
    print("Wrote", len(bar_paths) + len(trend_paths), "plots to", args.outdir)


if __name__ == "__main__":
    main()
