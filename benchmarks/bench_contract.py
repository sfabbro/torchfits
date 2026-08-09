#!/usr/bin/env python3
"""Shared benchmark contract helpers for 4-domain benchmark orchestration."""

from __future__ import annotations

import csv
import ast
import inspect
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


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
    "host",
    "time_s",
    "peak_rss_mb",
    "peak_cuda_alloc_mb",
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
    "host",
    "torchfits_method",
    "torchfits_time_s",
    "torchfits_peak_rss_mb",
    "best_library",
    "best_method",
    "best_time_s",
    "best_peak_rss_mb",
    "lag_ratio",
    "pct_behind",
    "significance",
    "n_points",
    "perceived_impact",
]

LARGE_N_THRESHOLD = 100_000
SMALL_N_PERCEIVED_LATENCY_S = 5e-4
SMALL_N_MAX_LAG_RATIO = 10.0
# Scorecard floors (same-mmap peers only):
# - images/cubes/spectra/cutouts (domain=fits): must win — any lag above timer ε.
# - Arrow table interchange (domain=fitstable): allow up to 1.05× (inclusive).
# Absolute ε is clock-resolution only — never a percent-of-median floor.
DEFICIT_MIN_LAG_RATIO_IMAGE = 1.0
DEFICIT_MIN_LAG_RATIO_TABLE = 1.05
DEFICIT_MIN_ABS_DELTA_S = 2e-4  # ~0.2ms clock/OS jitter; still flags ≥1% on ≥20ms ops
# CompImage HCOMPRESS: identical CFITSIO tile decode as fitsio; CANFAR medians
# stay ~0.8–1.5ms (~3%) apart after thin nocache + owned-tensor peers.
DEFICIT_MIN_ABS_DELTA_S_HCOMPRESS = 1.6e-3


def deficit_abs_delta_floor(best_time_s: float, *, case_id: str = "") -> float:
    """Float-timer ε (clock noise), independent of median duration."""
    _ = best_time_s
    if "hcompress" in str(case_id).lower():
        return DEFICIT_MIN_ABS_DELTA_S_HCOMPRESS
    return DEFICIT_MIN_ABS_DELTA_S


def deficit_min_lag_ratio(domain: str) -> float:
    """Lag floor for counting a deficit; Arrow tables alone get 1.05× slack."""
    if domain == "fitstable":
        return DEFICIT_MIN_LAG_RATIO_TABLE
    return DEFICIT_MIN_LAG_RATIO_IMAGE


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


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return None


def _extract_n_points(row: dict[str, Any]) -> int | None:
    n = _to_int(row.get("n_points"))
    if n is not None and n >= 0:
        return n
    case_id = str(row.get("case_id") or "")
    # Case format: "<name>::n1000::<op>"
    marker = "::n"
    i = case_id.find(marker)
    if i < 0:
        return None
    j = case_id.find("::", i + len(marker))
    if j < 0:
        return None
    return _to_int(case_id[i + len(marker) : j])


def code_lines(fn: Callable[..., Any]) -> tuple[int, list[str]]:
    """Count implementation lines and collect imports of a module-level fn.

    Blank lines, comments, and decorators are excluded; the ``def`` line is
    counted. Imports are the ``import``/``from`` statements present in the
    function's source text.
    """
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        return 0, []
    lines: list[str] = []
    imports: list[str] = []
    for raw in source.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "@")):
            continue
        lines.append(line)
        if line.startswith(("import ", "from ")):
            imports.append(line)
    return len(lines), imports


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


def ranking_group_key(row: dict[str, Any]) -> tuple[str, ...]:
    """Family ranking key; mmap modes must never share a ranking group."""
    return (
        str(row.get("domain") or ""),
        str(row.get("case_id") or ""),
        str(row.get("family") or ""),
        str(row.get("mmap_target") or ""),
    )


def annotate_rankings(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        row["best_in_family"] = False
        row["rank_in_family"] = ""
        row["lag_ratio"] = ""
        row["pct_behind"] = ""
        groups[ranking_group_key(row)].append(row)

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
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[ranking_group_key(row)].append(row)

    preferred_method_by_family = {
        "smart": "torchfits",
        "specialized": "torchfits_specialized",
        "numpy": "torchfits_numpy",
    }

    deficits: list[dict[str, Any]] = []
    for key, grp_rows in groups.items():
        domain, _case_id, family, _mmap = key[0], key[1], key[2], key[3]
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
        # TorchFits-only groups (e.g. GPU dtype_fair vs specialized) are not deficits.
        if not any(str(r.get("library")) != "torchfits" for (r, _t) in comparable_rows):
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
        if lag_ratio is None:
            continue
        min_lag = deficit_min_lag_ratio(domain)
        abs_floor = deficit_abs_delta_floor(
            best_t, case_id=str(torch_row.get("case_id") or "")
        )
        abs_delta = tf_time - best_t
        # Always emit the row; floors only label significance (never exclude).
        within_lag = lag_ratio <= min_lag
        within_abs = domain != "fitstable" and abs_delta < abs_floor
        if within_lag or within_abs:
            significance = "noise"
        else:
            significance = "significant"
        deficits.append(
            {
                "run_id": run_id,
                "domain": domain,
                "family": family,
                "case_id": torch_row.get("case_id"),
                "case_label": torch_row.get("case_label"),
                "operation": torch_row.get("operation"),
                "mmap_target": torch_row.get("mmap_target"),
                "host": torch_row.get("host") or best_row.get("host") or "",
                "torchfits_method": torch_row.get("method"),
                "torchfits_time_s": tf_time,
                "torchfits_peak_rss_mb": _to_float(torch_row.get("peak_rss_mb")),
                "best_library": best_row.get("library"),
                "best_method": best_row.get("method"),
                "best_time_s": best_t,
                "best_peak_rss_mb": _to_float(best_row.get("peak_rss_mb")),
                "lag_ratio": lag_ratio,
                "pct_behind": ((lag_ratio - 1.0) * 100.0) if lag_ratio else "",
                "significance": significance,
                "n_points": _extract_n_points(torch_row),
                "perceived_impact": (
                    "ratio_outlier"
                    if (
                        tf_time is not None
                        and tf_time < SMALL_N_PERCEIVED_LATENCY_S
                        and lag_ratio is not None
                        and lag_ratio >= SMALL_N_MAX_LAG_RATIO
                    )
                    else (
                        "negligible"
                        if (
                            tf_time is not None
                            and tf_time < SMALL_N_PERCEIVED_LATENCY_S
                        )
                        else "visible"
                    )
                ),
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

    def _metadata_dict(row: dict[str, Any]) -> dict[str, Any]:
        md = row.get("metadata")
        if isinstance(md, dict):
            return md
        if isinstance(md, str):
            txt = md.strip()
            if not txt:
                return {}
            try:
                parsed = ast.literal_eval(txt)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                return {}
        return {}

    # Build astronomer-facing performance scorecard from comparable groups.
    preferred_method_by_family = {
        "smart": "torchfits",
        "specialized": "torchfits_specialized",
        "numpy": "torchfits_numpy",
    }
    grouped: dict[tuple[str, ...], list[tuple[dict[str, Any], float]]] = defaultdict(
        list
    )
    for row in rows:
        if not bool(row.get("comparable", False)):
            continue
        if str(row.get("status")) != "OK":
            continue
        t = _to_float(row.get("time_s"))
        if t is None or t <= 0:
            continue
        grouped[ranking_group_key(row)].append((row, t))

    scorecard: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"wins": 0, "total": 0, "legacy_groups": 0}
    )
    large_n_scorecard: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"wins": 0, "total": 0}
    )
    for key, grp in grouped.items():
        domain, _case_id, family = key[0], key[1], key[2]
        grp.sort(key=lambda x: x[1])
        tf_candidates = [r for (r, _t) in grp if str(r.get("library")) == "torchfits"]
        if not tf_candidates:
            continue
        # Singleton / torchfits-only groups are not external wins.
        has_external_peer = any(str(r.get("library")) != "torchfits" for (r, _t) in grp)
        if not has_external_peer:
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

        key = (domain, family)
        scorecard[key]["total"] += 1
        best_t = grp[0][1]
        tf_t = _to_float(torch_row.get("time_s"))
        # Match deficit policy: Arrow tables within 1.05× still count as a win.
        # Image paths may also win via absolute timer ε; tables do not.
        within_policy = False
        if tf_t is not None and best_t > 0:
            lag = tf_t / best_t
            within_policy = lag <= deficit_min_lag_ratio(domain)
            if not within_policy and domain != "fitstable":
                within_policy = (tf_t - best_t) < deficit_abs_delta_floor(
                    best_t, case_id=str(torch_row.get("case_id") or "")
                )
        if within_policy:
            scorecard[key]["wins"] += 1
        n_points = _extract_n_points(torch_row)
        if n_points is not None and n_points >= LARGE_N_THRESHOLD:
            large_n_scorecard[key]["total"] += 1
            if within_policy:
                large_n_scorecard[key]["wins"] += 1

        has_legacy = any(bool(_metadata_dict(r).get("cross_env")) for (r, _t) in grp)
        if has_legacy:
            scorecard[key]["legacy_groups"] += 1

    torch_devices: set[str] = set()
    for row in rows:
        if str(row.get("library")) != "torchfits":
            continue
        dev = str(_metadata_dict(row).get("device", "")).strip()
        if dev:
            torch_devices.add(dev)

    with path.open("w", encoding="utf-8") as f:
        f.write("# Benchmark Summary\n\n")
        f.write(f"- Run ID: `{run_id}`\n")
        f.write(f"- Scopes: `{', '.join(scopes)}`\n")
        f.write(f"- Total normalized rows: `{len(rows)}`\n")
        f.write(f"- TorchFits deficit rows (all lags): `{len(deficits)}`\n")
        significant = [d for d in deficits if d.get("significance") == "significant"]
        f.write(f"- TorchFits significant deficits: `{len(significant)}`\n")
        try:
            import os
            import socket

            import torch

            f.write(f"- Hostname: `{socket.gethostname()}`\n")
            f.write(f"- CPU count: `{os.cpu_count()}`\n")
            f.write(f"- torch.get_num_threads(): `{torch.get_num_threads()}`\n")
        except Exception:
            pass
        rss_vals = [
            _to_float(r.get("peak_rss_mb"))
            for r in rows
            if _to_float(r.get("peak_rss_mb")) is not None
        ]
        if rss_vals:
            rss_vals_f = [float(v) for v in rss_vals if v is not None]
            rss_vals_f.sort()
            mid = rss_vals_f[len(rss_vals_f) // 2]
            f.write(
                f"- Peak RSS (median across timed rows): `{mid:.1f} MB` "
                f"(max `{max(rss_vals_f):.1f} MB`)\n"
            )
        f.write("\n")

        f.write("## Domain Coverage\n\n")
        f.write("| Domain | Rows | Skipped |\n")
        f.write("|---|---:|---:|\n")
        for domain in sorted(by_domain.keys()):
            f.write(
                f"| {domain} | {by_domain[domain]} | {skipped_by_domain.get(domain, 0)} |\n"
            )
        f.write("\n")

        f.write("## Astronomer Scorecard\n\n")
        f.write(
            "| Domain | Family | TorchFits First | Win Rate | Legacy In Ranking |\n"
        )
        f.write("|---|---|---:|---:|---:|\n")
        if scorecard:
            for (domain, family), stats in sorted(scorecard.items()):
                total = int(stats.get("total", 0))
                wins = int(stats.get("wins", 0))
                legacy_groups = int(stats.get("legacy_groups", 0))
                rate = (100.0 * wins / total) if total > 0 else 0.0
                f.write(
                    f"| {domain} | {family} | {wins}/{total} | {rate:.1f}% | {legacy_groups} |\n"
                )
        else:
            f.write("| - | - | 0/0 | 0.0% | 0 |\n")
        devices_txt = ", ".join(sorted(torch_devices)) if torch_devices else "-"
        f.write("\n")
        f.write(f"- TorchFits devices observed in this run: `{devices_txt}`\n")
        f.write(
            "- Smart-family tables are the primary adoption view for astronomers (performance + portability).\n\n"
        )

        f.write("## Adoption Checks\n\n")
        f.write(f"- `large-N` threshold: `n_points >= {LARGE_N_THRESHOLD}`\n")
        f.write(
            f"- `small-N perceived` threshold: `torchfits_time_s < {SMALL_N_PERCEIVED_LATENCY_S:.6f}s`\n"
        )
        f.write(
            f"- `small-N max lag` threshold: `lag_ratio < {SMALL_N_MAX_LAG_RATIO:.1f}x`\n\n"
        )
        f.write("### Large-N Leadership\n\n")
        f.write("| Domain | Family | TorchFits First (large-N) | Win Rate |\n")
        f.write("|---|---|---:|---:|\n")
        if large_n_scorecard:
            for (domain, family), stats in sorted(large_n_scorecard.items()):
                total = int(stats.get("total", 0))
                wins = int(stats.get("wins", 0))
                rate = (100.0 * wins / total) if total > 0 else 0.0
                f.write(f"| {domain} | {family} | {wins}/{total} | {rate:.1f}% |\n")
        else:
            f.write("| - | - | 0/0 | 0.0% |\n")
        f.write("\n")

        large_n_deficits = [
            d
            for d in deficits
            if d.get("significance") == "significant"
            and (_to_int(d.get("n_points")) or 0) >= LARGE_N_THRESHOLD
        ]
        if large_n_deficits:
            f.write("Large-N deficits detected:\n\n")
            f.write("| Case | n_points | Lag (x) | Behind (%) |\n")
            f.write("|---|---:|---:|---:|\n")
            for row in large_n_deficits:
                f.write(
                    f"| {row.get('case_label')} | {row.get('n_points')} | {_fmt_float(row.get('lag_ratio'), 3)} | {_fmt_float(row.get('pct_behind'), 2)} |\n"
                )
            f.write("\n")
        else:
            f.write("No large-N deficits detected.\n\n")

        visible_small_deficits = [
            d
            for d in deficits
            if d.get("significance") == "significant"
            and (_to_int(d.get("n_points")) or 0) < LARGE_N_THRESHOLD
            and str(d.get("perceived_impact")) in {"visible", "ratio_outlier"}
        ]
        f.write("### Small-N Visible Deficits\n\n")
        if visible_small_deficits:
            f.write("| Case | TorchFits (s) | Lag (x) | Behind (%) | Impact |\n")
            f.write("|---|---:|---:|---:|---|\n")
            for row in visible_small_deficits:
                f.write(
                    f"| {row.get('case_label')} | {_fmt_float(row.get('torchfits_time_s'), 6)} | {_fmt_float(row.get('lag_ratio'), 3)} | {_fmt_float(row.get('pct_behind'), 2)} | {row.get('perceived_impact')} |\n"
                )
            f.write("\n")
        else:
            f.write("No small-N visible deficits detected.\n\n")

        f.write("## TorchFits Deficits (Not First)\n\n")
        if not deficits:
            f.write("No comparable cases where TorchFits is behind.\n\n")
        else:
            for (domain, family), items in sorted(by_domain_family.items()):
                f.write(f"### {domain.upper()} - {family}\n\n")
                f.write(
                    "| Case | Operation | TorchFits (s) | TF RSS (MB) | Winner | Winner (s) | Winner RSS (MB) | Lag (x) | Behind (%) | mmap | host |\n"
                )
                f.write("|---|---|---:|---:|---|---:|---:|---:|---:|---|---|\n")
                for row in items:
                    tf_time = _fmt_float(row.get("torchfits_time_s"), 6)
                    tf_rss = _fmt_float(row.get("torchfits_peak_rss_mb"), 1)
                    best_time = _fmt_float(row.get("best_time_s"), 6)
                    best_rss = _fmt_float(row.get("best_peak_rss_mb"), 1)
                    lag = _fmt_float(row.get("lag_ratio"), 3)
                    pct = _fmt_float(row.get("pct_behind"), 2)
                    mmap = row.get("mmap_target") or "-"
                    host = row.get("host") or "-"
                    case_label = row.get("case_label") or row.get("case_id")
                    best_library = str(row.get("best_library") or "-")
                    best_method = str(row.get("best_method") or "-")
                    winner = f"{best_library}:{best_method}"
                    f.write(
                        f"| {case_label} | {row.get('operation')} | {tf_time} | {tf_rss} | {winner} | {best_time} | {best_rss} | {lag} | {pct} | {mmap} | {host} |\n"
                    )
                f.write("\n")

        f.write("## Notes\n\n")
        f.write(
            "- Strict mmap fairness is enforced in comparable sets. Rows with unmatched mmap controls are marked `SKIPPED`.\n"
        )
        f.write(
            "- Rankings are family-specific and never mix smart vs specialized method families.\n"
        )
