#!/usr/bin/env python3
"""Replay upstream spherical-geometry polygon fixtures against torchfits.sphere."""

from __future__ import annotations

import argparse
import codecs
import importlib.metadata as importlib_metadata
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from spherical_geometry import polygon as sg_polygon
from spherical_geometry.tests.helpers import ROOT_DIR as SG_ROOT_DIR

from torchfits.sphere.geom import SphericalPolygon, spherical_polygon_contains
from torchfits.wcs.healpix import (
    ang2pix as hp_ang2pix,
    get_all_neighbours as hp_get_all_neighbours,
    max_pixrad as hp_max_pixrad,
    nest_children as hp_nest_children,
    nside2pixarea as hp_nside2pixarea,
    pix2ang as hp_pix2ang,
)


PAIR_CASES = [
    {
        "name": "test_intersects_poly_simple",
        "lon1": [-10.0, 10.0, 10.0, -10.0, -10.0],
        "lat1": [30.0, 30.0, 0.0, 0.0, 30.0],
        "lon2": [-5.0, 15.0, 15.0, -5.0, -5.0],
        "lat2": [20.0, 20.0, -10.0, -10.0, 20.0],
        "expected_intersects": True,
    },
    {
        "name": "test_intersects_poly_fully_contained",
        "lon1": [-10.0, 10.0, 10.0, -10.0, -10.0],
        "lat1": [30.0, 30.0, 0.0, 0.0, 30.0],
        "lon2": [-5.0, 5.0, 5.0, -5.0, -5.0],
        "lat2": [20.0, 20.0, 10.0, 10.0, 20.0],
        "expected_intersects": True,
    },
    {
        "name": "test_hard_intersects_poly",
        "lon1": [-10.0, 10.0, 10.0, -10.0, -10.0],
        "lat1": [30.0, 30.0, 0.0, 0.0, 30.0],
        "lon2": [-20.0, 20.0, 20.0, -20.0, -20.0],
        "lat2": [20.0, 20.0, 10.0, 10.0, 20.0],
        "expected_intersects": True,
    },
    {
        "name": "test_not_intersects_poly",
        "lon1": [-10.0, 10.0, 10.0, -10.0, -10.0],
        "lat1": [30.0, 30.0, 5.0, 5.0, 30.0],
        "lon2": [-20.0, 20.0, 20.0, -20.0, -20.0],
        "lat2": [-20.0, -20.0, -10.0, -10.0, -20.0],
        "expected_intersects": False,
    },
]


@dataclass
class ContainsRow:
    case: str
    points: int
    mismatches: int
    mismatches_nonboundary: int
    area_rel_err: float
    torchfits_points_s: float
    comparator_points_s: float
    ratio_vs_comparator: float


@dataclass
class PairRow:
    case: str
    expected_intersects: bool
    comparator_intersects: bool
    torchfits_intersects: bool
    mismatch: int


@dataclass
class DifficultRow:
    case: str
    comparator_area_sr: float | None
    torchfits_pixel_area_sr: float | None
    comparator_nonempty: bool | None
    torchfits_nonempty: bool | None
    mismatch_nonempty: int
    healpix_area_by_nside: dict[str, float] | None
    healpix_eval_pixels_by_nside: dict[str, int] | None
    healpix_area_rel_err_by_nside: dict[str, float] | None
    healpix_convergence_rel: float | None


def _distribution_provenance(dist_name: str) -> dict[str, Any] | None:
    try:
        dist = importlib_metadata.distribution(dist_name)
    except importlib_metadata.PackageNotFoundError:
        return None
    meta: dict[str, Any] = {
        "name": dist.metadata.get("Name", dist_name),
        "version": dist.version,
        "installer": (dist.read_text("INSTALLER") or "unknown").strip().lower() or "unknown",
        "source": "site-packages",
        "release_like": True,
        "reason": "installed from index/conda/wheel/sdist",
    }
    if meta["installer"] == "conda":
        meta["reason"] = "conda package"
        return meta
    raw_direct = dist.read_text("direct_url.json")
    if raw_direct:
        try:
            direct = json.loads(raw_direct)
        except json.JSONDecodeError:
            direct = {}
        url = str(direct.get("url", "")).strip()
        if url:
            meta["source"] = url
        if direct.get("vcs_info") is not None:
            meta["release_like"] = False
            meta["reason"] = "VCS install"
        elif direct.get("dir_info") is not None and direct.get("archive_info") is None:
            meta["release_like"] = False
            meta["reason"] = "local path/editable install"
    return meta


def _time_many(fn, runs: int) -> float:
    fn()
    vals: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        vals.append(time.perf_counter() - t0)
    return float(np.median(vals))


def _strip_closed(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if lon.size >= 4 and np.isclose(lon[0], lon[-1]) and np.isclose(lat[0], lat[-1]):
        return lon[:-1], lat[:-1]
    return lon, lat


def _sg_contains_scalar(poly: Any, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    if hasattr(poly, "contains_lonlat"):
        return np.asarray(
            [bool(poly.contains_lonlat(float(lo), float(la), degrees=True)) for lo, la in zip(lon, lat, strict=False)],
            dtype=bool,
        )
    if hasattr(poly, "contains_radec"):
        return np.asarray(
            [bool(poly.contains_radec(float(lo), float(la), degrees=True)) for lo, la in zip(lon, lat, strict=False)],
            dtype=bool,
        )
    raise RuntimeError("unsupported comparator contains API")


def _sample_points_for_polygon(lon: np.ndarray, lat: np.ndarray, n_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    lon_open, lat_open = _strip_closed(lon, lat)
    n_local = max(n_points // 2, lon_open.size)
    n_global = n_points - n_local

    lon_min = float(np.min(lon_open))
    lon_max = float(np.max(lon_open))
    lat_min = float(np.min(lat_open))
    lat_max = float(np.max(lat_open))
    lon_pad = max(2.0, 0.25 * (lon_max - lon_min))
    lat_pad = max(2.0, 0.25 * (lat_max - lat_min))

    lon_local = rng.uniform(lon_min - lon_pad, lon_max + lon_pad, size=n_local)
    lat_local = rng.uniform(lat_min - lat_pad, lat_max + lat_pad, size=n_local)
    lon_global = rng.uniform(0.0, 360.0, size=n_global)
    lat_global = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, size=n_global)))

    qlon = np.concatenate([lon_open, lon_local, lon_global])[:n_points]
    qlat = np.concatenate([lat_open, lat_local, lat_global])[:n_points]
    return qlon.astype(np.float64), qlat.astype(np.float64)


def _xy_to_lonlat(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    lon = np.degrees(np.arctan2(y, x))
    lon = np.remainder(lon, 360.0)
    lat = np.degrees(np.arcsin(np.clip(z / np.sqrt(x * x + y * y + z * z), -1.0, 1.0)))
    return lon.astype(np.float64), lat.astype(np.float64)


def _to_array(line: bytes) -> np.ndarray:
    x = np.frombuffer(codecs.decode(line.strip(), "hex_codec"), dtype="<f8")
    return x.reshape((len(x) // 3, 3))


def _load_difficult_pairs(max_pairs: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    path = Path(SG_ROOT_DIR) / "difficult_intersections.txt"
    if not path.exists():
        return []
    lines = path.read_bytes().splitlines()
    out: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for i in range(0, len(lines), 4):
        if len(out) >= max_pairs:
            break
        if i + 3 >= len(lines):
            break
        a_points = _to_array(lines[i])
        a_inside = _to_array(lines[i + 1])
        b_points = _to_array(lines[i + 2])
        b_inside = _to_array(lines[i + 3])
        out.append((a_points, a_inside, b_points, b_inside))
    return out


def _build_sg_poly_lonlat(lon: np.ndarray, lat: np.ndarray):
    if hasattr(sg_polygon.SphericalPolygon, "from_lonlat"):
        return sg_polygon.SphericalPolygon.from_lonlat(lon, lat, degrees=True)
    return sg_polygon.SphericalPolygon.from_radec(lon, lat, degrees=True)


def _parse_nsides(raw: str) -> list[int]:
    vals: list[int] = []
    for part in [p.strip() for p in raw.split(",") if p.strip()]:
        val = int(part)
        if val <= 0 or (val & (val - 1)) != 0:
            raise ValueError(f"invalid nside {val}: must be positive power-of-two")
        vals.append(val)
    if not vals:
        raise ValueError("difficult-area-nsides must include at least one nside")
    for prev, cur in zip(vals[:-1], vals[1:], strict=False):
        if cur <= prev:
            raise ValueError("difficult-area-nsides must be strictly increasing")
        ratio = cur // prev
        if ratio * prev != cur or (ratio & (ratio - 1)) != 0:
            raise ValueError("each difficult-area nside must be a power-of-two multiple of the previous nside")
    return vals


def _levels_between(parent_nside: int, child_nside: int) -> int:
    ratio = child_nside // parent_nside
    if ratio * parent_nside != child_nside or ratio <= 0 or (ratio & (ratio - 1)) != 0:
        raise ValueError(f"invalid nside transition {parent_nside} -> {child_nside}")
    return int(ratio.bit_length() - 1)


def _xyz_to_lonlat_t(xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    lon = torch.remainder(torch.rad2deg(torch.atan2(y, x)), 360.0)
    lat = torch.rad2deg(torch.asin(torch.clamp(z / torch.linalg.norm(xyz, dim=-1).clamp_min(1e-15), -1.0, 1.0)))
    return lon.to(torch.float64), lat.to(torch.float64)


def _lonlat_to_xyz_t(lon_deg: torch.Tensor, lat_deg: torch.Tensor) -> torch.Tensor:
    lon = torch.deg2rad(lon_deg.to(torch.float64))
    lat = torch.deg2rad(lat_deg.to(torch.float64))
    c = torch.cos(lat)
    return torch.stack((c * torch.cos(lon), c * torch.sin(lon), torch.sin(lat)), dim=-1)


def _polygon_edge_seed_points(poly: SphericalPolygon, step_deg: float) -> tuple[torch.Tensor, torch.Tensor]:
    lon = poly.lon_deg.to(torch.float64).reshape(-1)
    lat = poly.lat_deg.to(torch.float64).reshape(-1)
    verts = _lonlat_to_xyz_t(lon, lat)
    verts_next = torch.roll(verts, shifts=-1, dims=0)
    pts: list[torch.Tensor] = []
    for i in range(verts.shape[0]):
        a = verts[i]
        b = verts_next[i]
        ang = float(torch.acos(torch.clamp(torch.dot(a, b), -1.0, 1.0)).item())
        nseg = max(1, int(math.ceil(math.degrees(ang) / max(step_deg, 1e-3))))
        t = torch.linspace(0.0, 1.0, steps=nseg + 1, dtype=torch.float64)
        if ang < 1e-12:
            edge_pts = a.reshape(1, 3).expand(nseg + 1, 3).clone()
        else:
            s = math.sin(ang)
            w0 = torch.sin((1.0 - t) * ang) / s
            w1 = torch.sin(t * ang) / s
            edge_pts = (w0.unsqueeze(1) * a.unsqueeze(0)) + (w1.unsqueeze(1) * b.unsqueeze(0))
            edge_pts = edge_pts / torch.linalg.norm(edge_pts, dim=-1, keepdim=True).clamp_min(1e-15)
        pts.append(edge_pts[:-1])
    if not pts:
        return lon, lat
    all_pts = torch.cat(pts, dim=0)
    return _xyz_to_lonlat_t(all_pts)


def _expand_with_neighbours(nside: int, pix: torch.Tensor, hops: int) -> torch.Tensor:
    out = torch.unique(torch.as_tensor(pix, dtype=torch.int64).reshape(-1))
    if out.numel() == 0 or hops <= 0:
        return out
    frontier = out
    for _ in range(hops):
        neigh = hp_get_all_neighbours(nside, frontier, nest=True, lonlat=False)
        neigh = torch.as_tensor(neigh, dtype=torch.int64).reshape(-1)
        neigh = neigh[neigh >= 0]
        if neigh.numel() == 0:
            break
        frontier = torch.unique(neigh)
        out = torch.unique(torch.cat((out, frontier)))
    return out


def _boundary_mask_from_neighbours(
    nside: int,
    pix: torch.Tensor,
    inside_mask: torch.Tensor,
    poly: SphericalPolygon,
) -> torch.Tensor:
    pix_i64 = torch.as_tensor(pix, dtype=torch.int64).reshape(-1)
    if pix_i64.numel() == 0:
        return torch.zeros(0, dtype=torch.bool)
    inside = torch.as_tensor(inside_mask, dtype=torch.bool).reshape(-1)
    neigh = hp_get_all_neighbours(nside, pix_i64, nest=True, lonlat=False)
    neigh = torch.as_tensor(neigh, dtype=torch.int64)
    if neigh.ndim == 1:
        neigh = neigh.reshape(8, 1)
    valid = neigh >= 0
    if not bool(valid.any()):
        return torch.zeros_like(inside)

    neigh_ids = torch.unique(neigh[valid])
    lon_n, lat_n = hp_pix2ang(nside, neigh_ids, nest=True, lonlat=True)
    neigh_inside = poly.contains(lon_n, lat_n, inclusive=True, atol_deg=1e-8).to(torch.bool).reshape(-1)

    sorted_ids, order = torch.sort(neigh_ids)
    sorted_inside = neigh_inside[order]

    flat = neigh.reshape(-1)
    flat_valid = flat >= 0
    flat_inside = torch.zeros_like(flat, dtype=torch.bool)
    if bool(flat_valid.any()):
        idx = torch.searchsorted(sorted_ids, flat[flat_valid])
        flat_inside[flat_valid] = sorted_inside[idx]
    neigh_inside_m = flat_inside.reshape(neigh.shape)
    return ((neigh_inside_m != inside.reshape(1, -1)) & valid).any(dim=0)


def _estimate_intersection_area_multires(
    poly_a: SphericalPolygon,
    poly_b: SphericalPolygon,
    seed_lon_deg: torch.Tensor,
    seed_lat_deg: torch.Tensor,
    nsides: list[int],
    sg_area_sr: float,
    sg_area_eps: float,
) -> tuple[float, dict[str, float], dict[str, int], dict[str, float], float | None]:
    area_by_nside: dict[str, float] = {}
    eval_pixels_by_nside: dict[str, int] = {}
    area_rel_err_by_nside: dict[str, float] = {}

    base_nside = nsides[0]
    candidate = torch.arange(12 * base_nside * base_nside, dtype=torch.int64)
    parents = candidate

    for i, nside in enumerate(nsides):
        if i > 0:
            levels = _levels_between(nsides[i - 1], nside)
            candidate = hp_nest_children(parents, levels=levels).reshape(-1)

        edge_step_deg = max(0.05, 0.75 * hp_max_pixrad(nside, degrees=True))
        edge_a_lon, edge_a_lat = _polygon_edge_seed_points(poly_a, step_deg=edge_step_deg)
        edge_b_lon, edge_b_lat = _polygon_edge_seed_points(poly_b, step_deg=edge_step_deg)
        seed_lon_all = torch.cat((seed_lon_deg, edge_a_lon, edge_b_lon))
        seed_lat_all = torch.cat((seed_lat_deg, edge_a_lat, edge_b_lat))
        seed_pix = hp_ang2pix(nside, seed_lon_all, seed_lat_all, nest=True, lonlat=True).reshape(-1).to(torch.int64)
        seed_hops = 2 if hp_max_pixrad(nside, degrees=True) > 5.0 else 1
        seed_support = _expand_with_neighbours(nside, seed_pix, hops=seed_hops)
        candidate = torch.unique(torch.cat((candidate, seed_support)))

        lon, lat = hp_pix2ang(nside, candidate, nest=True, lonlat=True)
        in_a = poly_a.contains(lon, lat, inclusive=True, atol_deg=1e-8).to(torch.bool).reshape(-1)
        in_b = poly_b.contains(lon, lat, inclusive=True, atol_deg=1e-8).to(torch.bool).reshape(-1)
        overlap = in_a & in_b

        area = float(int(overlap.sum().item()) * hp_nside2pixarea(nside, degrees=False))
        key = str(nside)
        area_by_nside[key] = area
        eval_pixels_by_nside[key] = int(candidate.numel())
        area_rel_err_by_nside[key] = abs(area - sg_area_sr) / max(abs(sg_area_sr), sg_area_eps)

        if i < len(nsides) - 1:
            boundary_a = _boundary_mask_from_neighbours(nside, candidate, in_a, poly_a)
            boundary_b = _boundary_mask_from_neighbours(nside, candidate, in_b, poly_b)
            parents = candidate[overlap | boundary_a | boundary_b | in_a | in_b]
            parents = torch.unique(torch.cat((parents, seed_support)))
            if parents.numel() == 0:
                parents = seed_support

    final_area = area_by_nside[str(nsides[-1])]
    convergence_rel = None
    if len(nsides) >= 2:
        prev = area_by_nside[str(nsides[-2])]
        convergence_rel = abs(final_area - prev) / max(abs(final_area), sg_area_eps)
    return final_area, area_by_nside, eval_pixels_by_nside, area_rel_err_by_nside, convergence_rel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-points", type=int, default=20_000)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--intersection-nside", type=int, default=512, help="nside for simple pair pixelized intersection parity")
    parser.add_argument("--max-difficult-pairs", type=int, default=8)
    parser.add_argument("--sg-area-nonempty-eps", type=float, default=1e-10)
    parser.add_argument("--max-contains-mismatches-nonboundary", type=int, default=0)
    parser.add_argument("--max-pair-intersection-mismatches", type=int, default=0)
    parser.add_argument("--max-difficult-nonempty-mismatches", type=int, default=0)
    parser.add_argument("--difficult-area-nsides", type=str, default="128,256,512,1024,2048,4096,8192,16384")
    parser.add_argument("--max-difficult-area-rel-error-final", type=float, default=0.12)
    parser.add_argument("--max-difficult-area-convergence-rel", type=float, default=0.10)
    parser.add_argument("--max-area-rel-error", type=float, default=1e-10)
    parser.add_argument("--allow-nonrelease-distributions", action="store_true")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("bench_results/upstream_replay_spherical_geometry_polygons.json"),
    )
    args = parser.parse_args()

    try:
        difficult_area_nsides = _parse_nsides(args.difficult_area_nsides)
    except ValueError as exc:
        print(f"Invalid --difficult-area-nsides: {exc}")
        return 2

    provenance = _distribution_provenance("spherical_geometry")
    if provenance is None:
        print("Comparator distribution metadata not found for spherical_geometry")
        return 2
    print(
        f"Comparator spherical_geometry=={provenance['version']} "
        f"({provenance['reason']}; source={provenance['source']})"
    )
    if not args.allow_nonrelease_distributions and not provenance["release_like"]:
        print("Non-release comparator distribution detected; refusing run without --allow-nonrelease-distributions")
        return 2

    contains_rows: list[ContainsRow] = []
    pair_rows: list[PairRow] = []
    difficult_rows: list[DifficultRow] = []

    failures: list[str] = []

    for i, case in enumerate(PAIR_CASES):
        lon1 = np.asarray(case["lon1"], dtype=np.float64)
        lat1 = np.asarray(case["lat1"], dtype=np.float64)
        lon2 = np.asarray(case["lon2"], dtype=np.float64)
        lat2 = np.asarray(case["lat2"], dtype=np.float64)

        sg1 = _build_sg_poly_lonlat(lon1, lat1)
        sg2 = _build_sg_poly_lonlat(lon2, lat2)

        lon1_open, lat1_open = _strip_closed(lon1, lat1)
        lon2_open, lat2_open = _strip_closed(lon2, lat2)
        tf1 = SphericalPolygon(torch.from_numpy(lon1_open), torch.from_numpy(lat1_open))
        tf2 = SphericalPolygon(torch.from_numpy(lon2_open), torch.from_numpy(lat2_open))

        qlon, qlat = _sample_points_for_polygon(lon1_open, lat1_open, args.n_points, seed=args.seed + i)
        qlon_t = torch.from_numpy(qlon)
        qlat_t = torch.from_numpy(qlat)

        t_tf = _time_many(lambda: spherical_polygon_contains(qlon_t, qlat_t, tf1.lon_deg, tf1.lat_deg, inclusive=False), runs=args.runs)
        t_sg = _time_many(lambda: _sg_contains_scalar(sg1, qlon, qlat), runs=max(1, args.runs // 2))

        tf_contains = spherical_polygon_contains(qlon_t, qlat_t, tf1.lon_deg, tf1.lat_deg, inclusive=False).cpu().numpy()
        tf_contains_inclusive = spherical_polygon_contains(
            qlon_t,
            qlat_t,
            tf1.lon_deg,
            tf1.lat_deg,
            inclusive=True,
            atol_deg=1e-5,
        ).cpu().numpy()
        sg_contains = _sg_contains_scalar(sg1, qlon, qlat)

        mismatches = int(np.sum(tf_contains != sg_contains))
        boundary_like = tf_contains_inclusive != tf_contains
        mismatches_nonboundary = int(np.sum((tf_contains != sg_contains) & (~boundary_like)))

        area_tf = float(tf1.area(degrees=False))
        area_sg = float(sg1.area())
        area_rel_err = abs(area_tf - area_sg) / max(abs(area_sg), 1e-15)

        contains_rows.append(
            ContainsRow(
                case=case["name"],
                points=int(qlon.size),
                mismatches=mismatches,
                mismatches_nonboundary=mismatches_nonboundary,
                area_rel_err=area_rel_err,
                torchfits_points_s=qlon.size / t_tf,
                comparator_points_s=qlon.size / t_sg,
                ratio_vs_comparator=t_sg / t_tf,
            )
        )

        sg_intersects = bool(sg1.intersects_poly(sg2))
        tf_intersection = tf1.pixelize(args.intersection_nside, nest=False).intersection(
            tf2.pixelize(args.intersection_nside, nest=False)
        )
        tf_intersects = bool(tf_intersection.pixels.numel() > 0)
        mismatch = int(tf_intersects != sg_intersects or sg_intersects != bool(case["expected_intersects"]))
        pair_rows.append(
            PairRow(
                case=case["name"],
                expected_intersects=bool(case["expected_intersects"]),
                comparator_intersects=sg_intersects,
                torchfits_intersects=tf_intersects,
                mismatch=mismatch,
            )
        )

        if mismatches_nonboundary > args.max_contains_mismatches_nonboundary:
            failures.append(
                f"{case['name']}: non-boundary contains mismatches {mismatches_nonboundary} > {args.max_contains_mismatches_nonboundary}"
            )
        if area_rel_err > args.max_area_rel_error:
            failures.append(
                f"{case['name']}: area rel err {area_rel_err:.3e} > {args.max_area_rel_error:.3e}"
            )
        if mismatch > 0:
            failures.append(f"{case['name']}: intersection mismatch (expected={case['expected_intersects']}, sg={sg_intersects}, tf={tf_intersects})")

    difficult = _load_difficult_pairs(args.max_difficult_pairs)
    for i, (a_points, a_inside, b_points, b_inside) in enumerate(difficult):
        sg_a = sg_polygon.SphericalPolygon(a_points, a_inside)
        sg_b = sg_polygon.SphericalPolygon(b_points, b_inside)
        sg_area = float(sg_a.intersection(sg_b).area())

        lon_a, lat_a = _xy_to_lonlat(a_points)
        lon_b, lat_b = _xy_to_lonlat(b_points)
        inside_a_lon, inside_a_lat = _xy_to_lonlat(a_inside[:1])
        inside_b_lon, inside_b_lat = _xy_to_lonlat(b_inside[:1])
        lon_a, lat_a = _strip_closed(lon_a, lat_a)
        lon_b, lat_b = _strip_closed(lon_b, lat_b)
        tf_a = SphericalPolygon(
            torch.from_numpy(lon_a),
            torch.from_numpy(lat_a),
            inside_lon_deg=float(inside_a_lon[0]),
            inside_lat_deg=float(inside_a_lat[0]),
        )
        tf_b = SphericalPolygon(
            torch.from_numpy(lon_b),
            torch.from_numpy(lat_b),
            inside_lon_deg=float(inside_b_lon[0]),
            inside_lat_deg=float(inside_b_lat[0]),
        )

        tf_nonempty = bool(tf_a.intersects(tf_b))
        seed_lon_np = np.concatenate([lon_a, lon_b, inside_a_lon, inside_b_lon]).astype(np.float64)
        seed_lat_np = np.concatenate([lat_a, lat_b, inside_a_lat, inside_b_lat]).astype(np.float64)
        (
            tf_area,
            area_by_nside,
            eval_pixels_by_nside,
            area_rel_err_by_nside,
            convergence_rel,
        ) = _estimate_intersection_area_multires(
            tf_a,
            tf_b,
            torch.from_numpy(seed_lon_np),
            torch.from_numpy(seed_lat_np),
            difficult_area_nsides,
            sg_area,
            sg_area_eps=args.sg_area_nonempty_eps,
        )

        sg_nonempty = bool(sg_area > args.sg_area_nonempty_eps)
        mismatch_nonempty = int(sg_nonempty != tf_nonempty)

        difficult_rows.append(
            DifficultRow(
                case=f"difficult_intersections[{i}]",
                comparator_area_sr=sg_area,
                torchfits_pixel_area_sr=tf_area,
                comparator_nonempty=sg_nonempty,
                torchfits_nonempty=tf_nonempty,
                mismatch_nonempty=mismatch_nonempty,
                healpix_area_by_nside=area_by_nside,
                healpix_eval_pixels_by_nside=eval_pixels_by_nside,
                healpix_area_rel_err_by_nside=area_rel_err_by_nside,
                healpix_convergence_rel=convergence_rel,
            )
        )

        if mismatch_nonempty > args.max_difficult_nonempty_mismatches:
            failures.append(
                f"difficult[{i}]: nonempty mismatch sg={sg_nonempty} tf={tf_nonempty}"
            )
        if sg_nonempty:
            final_rel_err = area_rel_err_by_nside[str(difficult_area_nsides[-1])]
            if final_rel_err > args.max_difficult_area_rel_error_final:
                failures.append(
                    f"difficult[{i}]: final area rel err {final_rel_err:.3e} > {args.max_difficult_area_rel_error_final:.3e}"
                )
            if convergence_rel is not None and convergence_rel > args.max_difficult_area_convergence_rel:
                failures.append(
                    f"difficult[{i}]: area convergence rel {convergence_rel:.3e} > {args.max_difficult_area_convergence_rel:.3e}"
                )

    print("Contains rows:")
    for r in contains_rows:
        print(
            f"  {r.case:38s} mismatches={r.mismatches:4d} nonboundary={r.mismatches_nonboundary:4d} "
            f"area_rel_err={r.area_rel_err:.3e} ratio={r.ratio_vs_comparator:.2f}x"
        )

    print("\nPair rows:")
    pair_mismatch_total = int(sum(r.mismatch for r in pair_rows))
    for r in pair_rows:
        print(
            f"  {r.case:38s} expected={r.expected_intersects} sg={r.comparator_intersects} "
            f"tf={r.torchfits_intersects} mismatch={r.mismatch}"
        )
    if pair_mismatch_total > args.max_pair_intersection_mismatches:
        failures.append(
            f"pair intersection mismatches total {pair_mismatch_total} > {args.max_pair_intersection_mismatches}"
        )

    if difficult_rows:
        print("\nDifficult rows:")
        difficult_mismatch_total = int(sum(r.mismatch_nonempty for r in difficult_rows))
        for r in difficult_rows:
            assert r.comparator_area_sr is not None
            eval_last = None
            area_rel_last = None
            if r.healpix_eval_pixels_by_nside:
                eval_last = r.healpix_eval_pixels_by_nside.get(str(difficult_area_nsides[-1]))
            if r.healpix_area_rel_err_by_nside:
                area_rel_last = r.healpix_area_rel_err_by_nside.get(str(difficult_area_nsides[-1]))
            print(
                f"  {r.case:38s} sg_nonempty={r.comparator_nonempty} tf_nonempty={r.torchfits_nonempty} "
                f"sg_area={r.comparator_area_sr:.3e} tf_area={r.torchfits_pixel_area_sr if r.torchfits_pixel_area_sr is not None else float('nan'):.3e} "
                f"rel_err@{difficult_area_nsides[-1]}={area_rel_last if area_rel_last is not None else float('nan'):.3e} "
                f"conv_rel={r.healpix_convergence_rel if r.healpix_convergence_rel is not None else float('nan'):.3e} "
                f"eval={eval_last if eval_last is not None else -1} mismatch={r.mismatch_nonempty}"
            )
        if difficult_mismatch_total > args.max_difficult_nonempty_mismatches:
            failures.append(
                f"difficult nonempty mismatches total {difficult_mismatch_total} > {args.max_difficult_nonempty_mismatches}"
            )

    payload = {
        "comparator_provenance": provenance,
        "settings": {
            "n_points": args.n_points,
            "runs": args.runs,
            "intersection_nside": args.intersection_nside,
            "max_difficult_pairs": args.max_difficult_pairs,
            "sg_area_nonempty_eps": args.sg_area_nonempty_eps,
            "difficult_area_nsides": difficult_area_nsides,
            "max_difficult_area_rel_error_final": args.max_difficult_area_rel_error_final,
            "max_difficult_area_convergence_rel": args.max_difficult_area_convergence_rel,
        },
        "contains_rows": [asdict(r) for r in contains_rows],
        "pair_rows": [asdict(r) for r in pair_rows],
        "difficult_rows": [asdict(r) for r in difficult_rows],
        "failures": failures,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nJSON: {args.json_out}")

    if failures:
        print("\nFailures:")
        for f in failures:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
