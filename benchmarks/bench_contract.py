#!/usr/bin/env python3
"""Shared benchmark contract helpers for 4-domain benchmark orchestration."""

from __future__ import annotations

import csv
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

RESULT_COLUMNS = [
    "run_id",
    "domain",
    "suite",
    "case_id",
    "case_label",
    "operation",
    "family",
    "library",
    "method",
    "mode",
    "status",
    "skip_reason",
    "comparable",
    "mmap_target",
    "time_s",
    "throughput",
    "unit",
    "size_mb",
    "n_points",
    "metadata",
    "best_in_family",
    "rank_in_family",
    "lag_ratio",
    "pct_behind",
]

DEFICIT_COLUMNS = [
    "run_id",
    "domain",
    "family",
    "case_id",
    "case_label",
    "operation",
    "mmap_target",
    "torchfits_method",
    "torchfits_time_s",
    "best_library",
    "best_method",
    "best_time_s",
    "lag_ratio",
    "pct_behind",
]


def make_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except Exception:
        return None
    if v != v:  # NaN
        return None
    return v


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            out: dict[str, Any] = {}
            for key in columns:
                value = row.get(key)
                if isinstance(value, (dict, list, tuple)):
                    out[key] = str(value)
                else:
                    out[key] = value
            writer.writerow(out)


def annotate_rankings(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        row["best_in_family"] = False
        row["rank_in_family"] = ""
        row["lag_ratio"] = ""
        row["pct_behind"] = ""
        groups[
            (str(row.get("domain")), str(row.get("case_id")), str(row.get("family")))
        ].append(row)

    for _key, grp_rows in groups.items():
        comparable_rows = []
        for row in grp_rows:
            if not bool(row.get("comparable", False)):
                continue
            if str(row.get("status")) != "OK":
                continue
            t = _to_float(row.get("time_s"))
            if t is None or t <= 0:
                continue
            comparable_rows.append((row, t))

        if not comparable_rows:
            continue

        comparable_rows.sort(key=lambda x: x[1])
        best_t = comparable_rows[0][1]
        for i, (row, t) in enumerate(comparable_rows, start=1):
            lag = t / best_t if best_t > 0 else None
            row["best_in_family"] = i == 1
            row["rank_in_family"] = i
            row["lag_ratio"] = lag
            row["pct_behind"] = ((lag - 1.0) * 100.0) if lag is not None else ""


def compute_deficits(rows: list[dict[str, Any]], run_id: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (str(row.get("domain")), str(row.get("case_id")), str(row.get("family")))
        ].append(row)

    preferred_method_by_family = {
        "smart": "torchfits",
        "specialized": "torchfits_specialized",
        "numpy": "torchfits_numpy",
    }

    deficits: list[dict[str, Any]] = []
    for (domain, _case_id, family), grp_rows in groups.items():
        comparable_rows = []
        for row in grp_rows:
            if not bool(row.get("comparable", False)):
                continue
            if str(row.get("status")) != "OK":
                continue
            t = _to_float(row.get("time_s"))
            if t is None or t <= 0:
                continue
            comparable_rows.append((row, t))

        if len(comparable_rows) < 2:
            continue

        comparable_rows.sort(key=lambda x: x[1])
        best_row, best_t = comparable_rows[0]

        tf_candidates = [
            r for (r, _t) in comparable_rows if str(r.get("library")) == "torchfits"
        ]
        if not tf_candidates:
            continue

        preferred = preferred_method_by_family.get(family)
        torch_row = None
        if preferred is not None:
            for r in tf_candidates:
                if str(r.get("method")) == preferred:
                    torch_row = r
                    break
        if torch_row is None:
            for r in tf_candidates:
                if str(r.get("method", "")).startswith("torchfits"):
                    torch_row = r
                    break
        if torch_row is None:
            torch_row = tf_candidates[0]

        tf_rank = torch_row.get("rank_in_family")
        tf_time = _to_float(torch_row.get("time_s"))
        if tf_rank in ("", None) or tf_time is None:
            continue
        try:
            tf_rank_int = int(tf_rank)
        except Exception:
            continue
        if tf_rank_int <= 1:
            continue

        lag_ratio = tf_time / best_t if best_t > 0 else None
        deficits.append(
            {
                "run_id": run_id,
                "domain": domain,
                "family": family,
                "case_id": torch_row.get("case_id"),
                "case_label": torch_row.get("case_label"),
                "operation": torch_row.get("operation"),
                "mmap_target": torch_row.get("mmap_target"),
                "torchfits_method": torch_row.get("method"),
                "torchfits_time_s": tf_time,
                "best_library": best_row.get("library"),
                "best_method": best_row.get("method"),
                "best_time_s": best_t,
                "lag_ratio": lag_ratio,
                "pct_behind": ((lag_ratio - 1.0) * 100.0) if lag_ratio else "",
            }
        )

    deficits.sort(
        key=lambda r: (
            str(r.get("domain")),
            str(r.get("family")),
            -float(r.get("pct_behind") or 0.0),
            str(r.get("case_id")),
        )
    )
    return deficits


def _fmt_float(value: Any, digits: int = 4) -> str:
    v = _to_float(value)
    if v is None:
        return "-"
    return f"{v:.{digits}f}"


def write_summary(
    path: Path,
    *,
    run_id: str,
    scopes: list[str],
    rows: list[dict[str, Any]],
    deficits: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    by_domain: dict[str, int] = defaultdict(int)
    skipped_by_domain: dict[str, int] = defaultdict(int)
    for row in rows:
        domain = str(row.get("domain"))
        by_domain[domain] += 1
        if str(row.get("status")) == "SKIPPED":
            skipped_by_domain[domain] += 1

    by_domain_family: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for d in deficits:
        by_domain_family[(str(d.get("domain")), str(d.get("family")))].append(d)

    with path.open("w", encoding="utf-8") as f:
        f.write("# Benchmark Summary\n\n")
        f.write(f"- Run ID: `{run_id}`\n")
        f.write(f"- Scopes: `{', '.join(scopes)}`\n")
        f.write(f"- Total normalized rows: `{len(rows)}`\n")
        f.write(f"- TorchFits deficit rows: `{len(deficits)}`\n\n")

        f.write("## Domain Coverage\n\n")
        f.write("| Domain | Rows | Skipped |\n")
        f.write("|---|---:|---:|\n")
        for domain in sorted(by_domain.keys()):
            f.write(
                f"| {domain} | {by_domain[domain]} | {skipped_by_domain.get(domain, 0)} |\n"
            )
        f.write("\n")

        f.write("## TorchFits Deficits (Not First)\n\n")
        if not deficits:
            f.write("No comparable cases where TorchFits is behind.\n\n")
        else:
            for (domain, family), items in sorted(by_domain_family.items()):
                f.write(f"### {domain.upper()} - {family}\n\n")
                f.write(
                    "| Case | Operation | TorchFits | Best | Lag (x) | Behind (%) | mmap |\n"
                )
                f.write("|---|---|---:|---:|---:|---:|---|\n")
                for row in items:
                    tf_time = _fmt_float(row.get("torchfits_time_s"), 6)
                    best_time = _fmt_float(row.get("best_time_s"), 6)
                    lag = _fmt_float(row.get("lag_ratio"), 3)
                    pct = _fmt_float(row.get("pct_behind"), 2)
                    mmap = row.get("mmap_target") or "-"
                    case_label = row.get("case_label") or row.get("case_id")
                    f.write(
                        f"| {case_label} | {row.get('operation')} | {tf_time} | {best_time} | {lag} | {pct} | {mmap} |\n"
                    )
                f.write("\n")

        f.write("## Notes\n\n")
        f.write(
            "- Strict mmap fairness is enforced in comparable sets. Rows with unmatched mmap controls are marked `SKIPPED`.\n"
        )
        f.write(
            "- Rankings are family-specific and never mix smart vs specialized method families.\n"
        )
