#!/usr/bin/env python3
"""
Quick analyzer for JSONL benchmark results produced by compare_readers.py.

Usage:
  python benchmarks/analyze_jsonl.py artifacts/benchmarks/full_sweep.jsonl [--plots-dir artifacts/benchmarks/plots]

Prints per-scenario summaries and torchfits vs fitsio speedup ratios when both exist.
Optionally, saves simple bar plots per scenario if matplotlib is available.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict


def _sanitize_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)[:200]


def maybe_plot_group(plots_dir: str, group_key: str, items: list[dict]):
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return None

    xs = []
    ys = []
    for r in items:
        if r.get("scenario") == "meta":
            continue
        label = f"{r.get('impl')}-{r.get('api')}"
        xs.append(label)
        ys.append(float(r.get("mean_ms", 0)))
    if not xs:
        return None
    pairs = sorted(zip(xs, ys), key=lambda p: p[1])
    xs_sorted, ys_sorted = zip(*pairs)
    os.makedirs(plots_dir, exist_ok=True)
    plt.figure(figsize=(8, 3 + max(1, len(xs_sorted) * 0.15)))
    plt.barh(range(len(xs_sorted)), ys_sorted, color="#3b82f6")
    plt.yticks(range(len(xs_sorted)), xs_sorted)
    plt.xlabel("mean ms")
    plt.title(group_key)
    out = os.path.join(plots_dir, _sanitize_filename(group_key) + ".png")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=str, help="Path to JSONL file")
    ap.add_argument("--plots-dir", type=str, default=None, help="Directory to save plots; requires matplotlib")
    args = ap.parse_args()

    rows = []
    with open(args.path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass

    # meta summary (if available)
    metas = [r for r in rows if r.get("scenario") == "meta"]
    if metas:
        m = metas[-1]
        print("== meta ==")
        print("python:", m.get("python"))
        print("platform:", m.get("platform"))
        print("versions:", m.get("versions"))

    # group by scenario + key dimensions
    def key(rec):
        k = [rec.get("scenario")]
        for dim in ("size", "rows", "cutouts", "cut_hw", "hdus", "files", "radius_arcsec"):
            if dim in rec:
                k.append(f"{dim}={rec[dim]}")
        return " | ".join(k)

    groups = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)

    for gk, items in sorted(groups.items()):
        if gk.startswith("meta"):
            continue
        print(f"\n== {gk} ==")
        tfs = [r for r in items if r.get("impl") == "torchfits"]
        fsi = [r for r in items if r.get("impl") == "fitsio"]
        apy = [r for r in items if r.get("impl") == "astropy"]
        t_best = min(tfs, key=lambda r: r.get("mean_ms", float("inf")), default=None)
        f_best = min(fsi, key=lambda r: r.get("mean_ms", float("inf")), default=None)
        a_best = min(apy, key=lambda r: r.get("mean_ms", float("inf")), default=None)

        def fmt(r):
            if not r:
                return "n/a"
            return f"{r['impl']} {r['api']} {r['mean_ms']:.2f}ms"

        print("best torchfits:", fmt(t_best))
        if f_best:
            print("best fitsio:  ", fmt(f_best))
        if a_best:
            print("best astropy: ", fmt(a_best))

        if t_best and f_best:
            ratio = f_best["mean_ms"] / t_best["mean_ms"] if t_best["mean_ms"] > 0 else float("inf")
            print(f"speedup vs fitsio: {ratio:.2f}x (>1.00x means torchfits faster)")

        if args.plots_dir:
            out = maybe_plot_group(args.plots_dir, gk, items)
            if out:
                print("plot:", out)


if __name__ == "__main__":
    main()
