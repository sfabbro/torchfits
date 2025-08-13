#!/usr/bin/env python3
"""
Update README.md with key benchmark metrics between markers.

Markers in README:
<!-- perf:begin -->
... auto content ...
<!-- perf:end -->

Usage:
  python scripts/update_readme_perf.py artifacts/benchmarks/full_sweep.jsonl README.md
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict


def load_rows(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def group_key(rec: dict) -> str:
    k = [rec.get("scenario")]
    for dim in ("size", "rows", "cutouts", "cut_hw", "hdus", "files", "radius_arcsec"):
        if dim in rec:
            k.append(f"{dim}={rec[dim]}")
    return " ".join([str(x) for x in k if x])


def best(items: list[dict], impl: str) -> dict | None:
    cands = [r for r in items if r.get("impl") == impl]
    if not cands:
        return None
    return min(cands, key=lambda r: r.get("mean_ms", 1e30))


def summarize(jsonl: str) -> str:
    rows = load_rows(jsonl)
    groups = defaultdict(list)
    for r in rows:
        if r.get("scenario") == "meta":
            continue
        groups[group_key(r)].append(r)

    # Pick key scenarios
    def pick(predicate):
        for k, items in groups.items():
            if predicate(k):
                tb = best(items, "torchfits")
                fb = best(items, "fitsio")
                ab = best(items, "astropy")
                yield k, tb, fb, ab

    bullets: list[str] = []

    # Image full reads
    image_rows = list(pick(lambda k: k.startswith("image_full") or k.startswith("Image full read") or "scenario=image_full" in k))
    if image_rows:
        rlines = []
        for k, tb, fb, ab in image_rows:
            size = re.search(r"size=(\d+)", k)
            sz = size.group(1) if size else "?"
            if tb:
                rlines.append(f"{sz}: {tb['mean_ms']:.2f} ms")
        if rlines:
            bullets.append("- Image full read (torchfits): " + ", ".join(rlines))

    # Random cutouts 32x32
    cut32 = [x for x in groups.items() if "Random cutouts" in x[0] and ("cut_hw=(32,32)" in x[0] or "10 x 32x32" in x[0])]
    if cut32:
        _, items = cut32[0]
        tb = [r for r in items if r.get("impl") == "torchfits" and ("full->slice" in r.get("api","") or "full→slice" in r.get("api",""))]
        if not tb:
            tb = [r for r in items if r.get("impl") == "torchfits"]
        if tb:
            v = min(tb, key=lambda r: r["mean_ms"])  # type: ignore
            bullets.append(f"- Cutouts 10×32×32 (torchfits best): {v['mean_ms']:.2f} ms")

    # Sky cutouts ~30"
    sky = [x for x in groups.items() if "Sky-position cutouts" in x[0] and ("radius_arcsec=30" in x[0] or "radius arcsec=30.0" in x[0])]
    if sky:
        _, items = sky[0]
        tb = [r for r in items if r.get("impl") == "torchfits" and "multi" in r.get("api","")]
        if tb:
            v = min(tb, key=lambda r: r["mean_ms"])  # type: ignore
            bullets.append(f"- Sky cutouts 30\" (torchfits batched): {v['mean_ms']:.2f} ms")

    # Table subset (4 cols, 200k)
    tsub = [x for x in groups.items() if "Table column subset" in x[0] and "cols=4" in x[0]]
    if tsub:
        _, items = tsub[0]
        tf = [r for r in items if r.get("impl") == "torchfits" and "tensor" in r.get("api","")]
        if tf:
            v = min(tf, key=lambda r: r["mean_ms"])  # type: ignore
            bullets.append(f"- Table 4 cols × 200k (torchfits tensor): {v['mean_ms']:.2f} ms")

    # Frameworks (full table 200k)
    tfw = [x for x in groups.items() if "Table frameworks" in x[0]]
    if tfw:
        _, items = tfw[0]
        tf = [r for r in items if r.get("impl") == "torchfits" and r.get("format") in ("table","dataframe")]
        if tf:
            v = min(tf, key=lambda r: r["mean_ms"])  # type: ignore
            bullets.append(f"- Full table 200k (torchfits): {v['mean_ms']:.2f} ms")

    content = ["<!-- perf:begin -->", "", "Performance highlights (auto-updated):", ""]
    content += bullets or ["- No data found"]
    content += ["", "See artifacts/benchmarks/plots for charts.", "", "<!-- perf:end -->"]
    return "\n".join(content) + "\n"


def update_readme(readme_path: str, block: str) -> None:
    with open(readme_path, "r") as f:
        txt = f.read()
    if "<!-- perf:begin -->" in txt and "<!-- perf:end -->" in txt:
        new = re.sub(r"<!-- perf:begin -->(.|\n)*?<!-- perf:end -->", block, txt, flags=re.MULTILINE)
    else:
        # Insert near top after Features or Quickstart
        if "## Features" in txt:
            new = txt.replace("## Features", block + "\n## Features", 1)
        else:
            new = block + txt
    with open(readme_path, "w") as f:
        f.write(new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", help="Path to full_sweep.jsonl")
    ap.add_argument("readme", help="README.md path")
    args = ap.parse_args()
    block = summarize(args.jsonl)
    update_readme(args.readme, block)


if __name__ == "__main__":
    main()
