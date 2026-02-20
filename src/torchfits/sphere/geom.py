"""Spherical geometry primitives and polygon utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from ..wcs import healpix as _healpix
from .core import great_circle_distance, lonlat_to_unit_xyz


def _normalize_polygon_vertices(
    lon_deg: Tensor | list[float],
    lat_deg: Tensor | list[float],
) -> tuple[Tensor, Tensor]:
    lon_t = torch.as_tensor(lon_deg, dtype=torch.float64)
    lat_t = torch.as_tensor(lat_deg, dtype=torch.float64)
    if lon_t.ndim != 1 or lat_t.ndim != 1:
        raise ValueError("polygon vertices must be 1D sequences")
    if lon_t.shape[0] != lat_t.shape[0]:
        raise ValueError("lon_deg and lat_deg must have the same number of vertices")
    if lon_t.shape[0] < 3:
        raise ValueError("polygon must contain at least three vertices")

    if lon_t.shape[0] >= 4:
        # Accept closed polygons and normalize to open [N] form.
        if bool(torch.isclose(lon_t[0], lon_t[-1], atol=1e-12, rtol=0.0)) and bool(
            torch.isclose(lat_t[0], lat_t[-1], atol=1e-12, rtol=0.0)
        ):
            lon_t = lon_t[:-1]
            lat_t = lat_t[:-1]
    if lon_t.shape[0] < 3:
        raise ValueError("polygon must contain at least three distinct vertices")
    return lon_t, lat_t


def _signed_area_triangle(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    det = (a * torch.cross(b, c, dim=-1)).sum(dim=-1)
    den = 1.0 + (a * b).sum(dim=-1) + (b * c).sum(dim=-1) + (c * a).sum(dim=-1)
    return 2.0 * torch.atan2(det, den.clamp(min=1e-15))


def _points_on_edges(points: Tensor, vertices: Tensor, atol_rad: float) -> Tensor:
    v0 = vertices
    v1 = torch.roll(vertices, shifts=-1, dims=0)
    d_p_v0 = torch.acos(torch.clamp(torch.einsum("mj,nj->mn", points, v0), -1.0, 1.0))
    d_p_v1 = torch.acos(torch.clamp(torch.einsum("mj,nj->mn", points, v1), -1.0, 1.0))
    d_v0_v1 = torch.acos(torch.clamp((v0 * v1).sum(dim=-1), -1.0, 1.0)).unsqueeze(0)
    return torch.abs((d_p_v0 + d_p_v1) - d_v0_v1) <= atol_rad


def _arc_contains_point(a: Tensor, b: Tensor, p: Tensor, tol_rad: float) -> Tensor:
    """
    Check whether point(s) `p` lie on the minor great-circle arc from `a` to `b`.

    a,b: [3]
    p: [..., 3]
    """
    d_ab = torch.acos(torch.clamp((a * b).sum(dim=-1), -1.0, 1.0))
    d_ap = torch.acos(torch.clamp((p * a).sum(dim=-1), -1.0, 1.0))
    d_pb = torch.acos(torch.clamp((p * b).sum(dim=-1), -1.0, 1.0))
    return torch.abs((d_ap + d_pb) - d_ab) <= tol_rad


def _great_circle_arcs_intersect_many(
    a0: Tensor,
    a1: Tensor,
    b0: Tensor,
    b1: Tensor,
    *,
    tol_rad: float = 1e-10,
) -> Tensor:
    """
    Vectorized arc-arc intersection for one arc (a0,a1) vs many arcs (b0,b1).

    Returns boolean tensor of shape [M].
    """
    n1 = torch.cross(a0, a1, dim=-1)
    n1_norm = torch.linalg.norm(n1)
    if float(n1_norm.item()) < 1e-15:
        return torch.zeros(b0.shape[0], dtype=torch.bool, device=b0.device)

    n2 = torch.cross(b0, b1, dim=-1)
    n2_norm = torch.linalg.norm(n2, dim=-1)
    valid = n2_norm > 1e-15
    if not bool(valid.any()):
        return torch.zeros(b0.shape[0], dtype=torch.bool, device=b0.device)

    p = torch.cross(n1.unsqueeze(0).expand_as(n2), n2, dim=-1)
    p_norm = torch.linalg.norm(p, dim=-1)
    valid = valid & (p_norm > 1e-15)
    if not bool(valid.any()):
        return torch.zeros(b0.shape[0], dtype=torch.bool, device=b0.device)

    i1 = p / p_norm.clamp_min(1e-15).unsqueeze(-1)
    i2 = -i1

    on_a_i1 = _arc_contains_point(a0, a1, i1, tol_rad)
    on_a_i2 = _arc_contains_point(a0, a1, i2, tol_rad)
    on_b_i1 = _arc_contains_point(b0, b1, i1, tol_rad)
    on_b_i2 = _arc_contains_point(b0, b1, i2, tol_rad)

    out = valid & ((on_a_i1 & on_b_i1) | (on_a_i2 & on_b_i2))
    return out


def _great_circle_arcs_intersect_matrix(
    a0: Tensor,
    a1: Tensor,
    b0: Tensor,
    b1: Tensor,
    *,
    tol_rad: float = 1e-10,
) -> Tensor:
    """
    Pairwise arc intersection matrix.

    a0,a1: [E1,3], b0,b1: [E2,3]
    returns: [E1,E2] bool
    """
    n1 = torch.cross(a0, a1, dim=-1)  # [E1,3]
    n2 = torch.cross(b0, b1, dim=-1)  # [E2,3]
    n1_norm = torch.linalg.norm(n1, dim=-1)
    n2_norm = torch.linalg.norm(n2, dim=-1)
    valid = (n1_norm > 1e-15).unsqueeze(1) & (n2_norm > 1e-15).unsqueeze(0)
    if not bool(valid.any()):
        return torch.zeros((a0.shape[0], b0.shape[0]), dtype=torch.bool, device=a0.device)

    p = torch.cross(n1[:, None, :], n2[None, :, :], dim=-1)  # [E1,E2,3]
    p_norm = torch.linalg.norm(p, dim=-1)
    valid = valid & (p_norm > 1e-15)
    if not bool(valid.any()):
        return torch.zeros((a0.shape[0], b0.shape[0]), dtype=torch.bool, device=a0.device)

    i1 = p / p_norm.clamp_min(1e-15).unsqueeze(-1)
    i2 = -i1

    a0e = a0[:, None, :]
    a1e = a1[:, None, :]
    b0e = b0[None, :, :]
    b1e = b1[None, :, :]

    on_a_i1 = _arc_contains_point(a0e, a1e, i1, tol_rad)
    on_a_i2 = _arc_contains_point(a0e, a1e, i2, tol_rad)
    on_b_i1 = _arc_contains_point(b0e, b1e, i1, tol_rad)
    on_b_i2 = _arc_contains_point(b0e, b1e, i2, tol_rad)

    return valid & ((on_a_i1 & on_b_i1) | (on_a_i2 & on_b_i2))


def _planar_point_in_polygon(
    points_xy: Tensor,
    polygon_xy: Tensor,
    *,
    eps: float = 1e-15,
) -> Tensor:
    """
    Even-odd rule for a simple planar polygon.

    points_xy: [M, 2]
    polygon_xy: [N, 2]
    returns: [M] bool
    """
    x = points_xy[:, 0].unsqueeze(1)
    y = points_xy[:, 1].unsqueeze(1)

    xi = polygon_xy[:, 0].unsqueeze(0)
    yi = polygon_xy[:, 1].unsqueeze(0)
    xj = torch.roll(xi, shifts=-1, dims=1)
    yj = torch.roll(yi, shifts=-1, dims=1)

    y_between = (yi > y) != (yj > y)
    dy = yj - yi
    dy = torch.where(dy.abs() < eps, torch.where(dy >= 0.0, torch.full_like(dy, eps), torch.full_like(dy, -eps)), dy)
    x_inter = (xj - xi) * (y - yi) / dy + xi
    crossings = y_between & (x < x_inter)
    return (crossings.to(torch.int64).sum(dim=1) % 2) == 1


def _initial_bearing_rad(
    lon1_deg: Tensor | float,
    lat1_deg: Tensor | float,
    lon2_deg: Tensor | float,
    lat2_deg: Tensor | float,
) -> Tensor:
    lon1 = torch.deg2rad(torch.as_tensor(lon1_deg))
    lat1 = torch.deg2rad(torch.as_tensor(lat1_deg))
    lon2 = torch.deg2rad(torch.as_tensor(lon2_deg))
    lat2 = torch.deg2rad(torch.as_tensor(lat2_deg))
    lon1, lat1, lon2, lat2 = torch.broadcast_tensors(lon1, lat1, lon2, lat2)

    dlon = lon2 - lon1
    y = torch.sin(dlon) * torch.cos(lat2)
    x = torch.cos(lat1) * torch.sin(lat2) - torch.sin(lat1) * torch.cos(lat2) * torch.cos(dlon)
    return torch.atan2(y, x)


def _destination_lonlat_deg(
    lon_deg: Tensor | float,
    lat_deg: Tensor | float,
    bearing_rad: Tensor | float,
    distance_rad: Tensor | float,
) -> tuple[Tensor, Tensor]:
    """Great-circle destination from start point + initial bearing + angular distance."""
    lon1 = torch.deg2rad(torch.as_tensor(lon_deg))
    lat1 = torch.deg2rad(torch.as_tensor(lat_deg))
    bearing = torch.as_tensor(bearing_rad)
    dist = torch.as_tensor(distance_rad)
    lon1, lat1, bearing, dist = torch.broadcast_tensors(lon1, lat1, bearing, dist)

    sin_lat2 = torch.sin(lat1) * torch.cos(dist) + torch.cos(lat1) * torch.sin(dist) * torch.cos(bearing)
    lat2 = torch.asin(torch.clamp(sin_lat2, -1.0, 1.0))

    y = torch.sin(bearing) * torch.sin(dist) * torch.cos(lat1)
    x = torch.cos(dist) - (torch.sin(lat1) * torch.sin(lat2))
    lon2 = lon1 + torch.atan2(y, x)
    lon2 = torch.remainder(lon2, 2.0 * math.pi)
    return torch.rad2deg(lon2), torch.rad2deg(lat2)


def _make_tangent_basis(center: Tensor) -> tuple[Tensor, Tensor]:
    ref = torch.tensor([0.0, 0.0, 1.0], dtype=center.dtype, device=center.device)
    if float(torch.abs((ref * center).sum()).item()) > 0.9:
        ref = torch.tensor([1.0, 0.0, 0.0], dtype=center.dtype, device=center.device)
    e1 = torch.cross(ref, center, dim=-1)
    e1 = e1 / torch.linalg.norm(e1).clamp_min(1e-15)
    e2 = torch.cross(center, e1, dim=-1)
    return e1, e2


def _gnomonic_project(vertices_xyz: Tensor, center: Tensor) -> np.ndarray:
    e1, e2 = _make_tangent_basis(center)
    den = (vertices_xyz * center.unsqueeze(0)).sum(dim=-1)
    if bool(torch.any(den <= 1e-8)):
        raise ValueError("polygon cannot be projected gnomonically (vertices must lie in one open hemisphere)")
    x = (vertices_xyz * e1.unsqueeze(0)).sum(dim=-1) / den
    y = (vertices_xyz * e2.unsqueeze(0)).sum(dim=-1) / den
    return torch.stack((x, y), dim=-1).cpu().numpy()


def _signed_area_2d(xy: np.ndarray) -> float:
    x = xy[:, 0]
    y = xy[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _is_point_in_triangle_2d(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray, eps: float = 1e-15) -> bool:
    v0 = c - a
    v1 = b - a
    v2 = p - a
    den = (v0[0] * v1[1]) - (v1[0] * v0[1])
    if abs(den) < eps:
        return False
    u = (v2[0] * v1[1] - v1[0] * v2[1]) / den
    v = (v0[0] * v2[1] - v2[0] * v0[1]) / den
    return (u >= -eps) and (v >= -eps) and (u + v <= 1.0 + eps)


def _triangulate_simple_polygon_2d(xy: np.ndarray) -> list[tuple[int, int, int]]:
    n = xy.shape[0]
    if n < 3:
        raise ValueError("need at least three vertices")
    if n == 3:
        return [(0, 1, 2)]

    orientation = 1.0 if _signed_area_2d(xy) >= 0.0 else -1.0
    indices = list(range(n))
    triangles: list[tuple[int, int, int]] = []
    guard = 0
    max_iters = n * n

    while len(indices) > 3 and guard < max_iters:
        guard += 1
        ear_found = False
        m = len(indices)
        for k in range(m):
            i_prev = indices[(k - 1) % m]
            i_curr = indices[k]
            i_next = indices[(k + 1) % m]
            a = xy[i_prev]
            b = xy[i_curr]
            c = xy[i_next]

            cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
            if cross * orientation <= 1e-15:
                continue

            has_inner = False
            for j in indices:
                if j in (i_prev, i_curr, i_next):
                    continue
                if _is_point_in_triangle_2d(xy[j], a, b, c):
                    has_inner = True
                    break
            if has_inner:
                continue

            triangles.append((i_prev, i_curr, i_next))
            del indices[k]
            ear_found = True
            break

        if not ear_found:
            break

    if len(indices) == 3:
        triangles.append((indices[0], indices[1], indices[2]))
    if len(indices) != 3:
        raise ValueError("failed to triangulate polygon; ensure polygon is simple and non-self-intersecting")
    if not triangles:
        raise ValueError("failed to triangulate polygon; ensure polygon is simple and non-self-intersecting")
    return triangles


def spherical_triangle_area(
    lon1_deg: Tensor | float,
    lat1_deg: Tensor | float,
    lon2_deg: Tensor | float,
    lat2_deg: Tensor | float,
    lon3_deg: Tensor | float,
    lat3_deg: Tensor | float,
    *,
    degrees: bool = False,
) -> Tensor:
    """
    Spherical triangle area on the unit sphere.

    Returns steradians by default, or square degrees when `degrees=True`.
    """
    a = lonlat_to_unit_xyz(lon1_deg, lat1_deg)
    b = lonlat_to_unit_xyz(lon2_deg, lat2_deg)
    c = lonlat_to_unit_xyz(lon3_deg, lat3_deg)
    area_sr = torch.abs(_signed_area_triangle(a, b, c))
    if degrees:
        return area_sr * (180.0 / math.pi) ** 2
    return area_sr


def spherical_polygon_signed_area(
    lon_deg: Tensor | list[float],
    lat_deg: Tensor | list[float],
    *,
    degrees: bool = False,
) -> Tensor:
    """
    Signed area of a simple spherical polygon on the unit sphere.

    Contract:
    - vertices must define a simple, non-self-intersecting polygon
    - polygon is expected to lie in a region smaller than a full sphere
    - sign follows vertex winding orientation
    """
    lon_t, lat_t = _normalize_polygon_vertices(lon_deg, lat_deg)
    v = lonlat_to_unit_xyz(lon_t, lat_t).to(dtype=torch.float64)

    a = v[0].expand(v.shape[0] - 2, 3)
    b = v[1:-1]
    c = v[2:]
    area = _signed_area_triangle(a, b, c).sum()
    if degrees:
        return area * (180.0 / math.pi) ** 2
    return area


def spherical_polygon_area(
    lon_deg: Tensor | list[float],
    lat_deg: Tensor | list[float],
    *,
    degrees: bool = False,
    oriented: bool = False,
) -> Tensor:
    """
    Area of a simple spherical polygon.

    By default returns the non-oriented enclosed area in [0, 2*pi] steradians.
    Set `oriented=True` to preserve signed winding orientation.
    """
    signed = spherical_polygon_signed_area(lon_deg, lat_deg, degrees=False)
    if oriented:
        area = signed
    else:
        area = torch.abs(signed)
        full = 4.0 * math.pi
        if float(area.item()) > 2.0 * math.pi:
            area = torch.tensor(full, dtype=area.dtype, device=area.device) - area
    if degrees:
        return area * (180.0 / math.pi) ** 2
    return area


def spherical_polygon_contains(
    lon_deg: Tensor | float,
    lat_deg: Tensor | float,
    poly_lon_deg: Tensor | list[float],
    poly_lat_deg: Tensor | list[float],
    *,
    inclusive: bool = True,
    atol_deg: float = 1e-10,
    inside_lon_deg: Tensor | float | None = None,
    inside_lat_deg: Tensor | float | None = None,
) -> Tensor:
    """
    Point-in-polygon for simple (possibly non-convex) spherical polygons.

    Uses tangent-plane winding number around each query point.
    """
    lon_poly, lat_poly = _normalize_polygon_vertices(poly_lon_deg, poly_lat_deg)
    vertices = lonlat_to_unit_xyz(lon_poly, lat_poly).to(dtype=torch.float64)

    lon_t = torch.as_tensor(lon_deg, dtype=torch.float64)
    lat_t = torch.as_tensor(lat_deg, dtype=torch.float64)
    lon_t, lat_t = torch.broadcast_tensors(lon_t, lat_t)
    shape = lon_t.shape
    points = lonlat_to_unit_xyz(lon_t, lat_t).reshape(-1, 3).to(dtype=torch.float64)

    # Fast/stable path: if polygon is in one hemisphere, gnomonic projection maps
    # great-circle edges to straight segments exactly.
    center = vertices.mean(dim=0)
    center = center / torch.linalg.norm(center).clamp_min(1e-15)
    vden = (vertices * center.unsqueeze(0)).sum(dim=-1)
    use_gnomonic = bool(torch.all(vden > 1e-10))

    if use_gnomonic:
        e1, e2 = _make_tangent_basis(center)
        vx = (vertices * e1.unsqueeze(0)).sum(dim=-1) / vden
        vy = (vertices * e2.unsqueeze(0)).sum(dim=-1) / vden
        poly_xy = torch.stack((vx, vy), dim=-1)

        pden = (points * center.unsqueeze(0)).sum(dim=-1)
        valid = pden > 1e-12
        inside = torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
        if valid.any():
            p = points[valid]
            px = (p * e1.unsqueeze(0)).sum(dim=-1) / pden[valid]
            py = (p * e2.unsqueeze(0)).sum(dim=-1) / pden[valid]
            inside_valid = _planar_point_in_polygon(torch.stack((px, py), dim=-1), poly_xy)
            inside[valid] = inside_valid
    else:
        # Fallback path for polygons not entirely in one hemisphere.
        pv = torch.einsum("mj,nj->mn", points, vertices)
        proj = vertices.unsqueeze(0) - points.unsqueeze(1) * pv.unsqueeze(-1)
        proj_norm = torch.linalg.norm(proj, dim=-1)
        proj = proj / proj_norm.clamp_min(1e-14).unsqueeze(-1)
        proj_next = torch.roll(proj, shifts=-1, dims=1)
        cross = torch.cross(proj, proj_next, dim=-1)
        num = (points.unsqueeze(1) * cross).sum(dim=-1)
        den = torch.clamp((proj * proj_next).sum(dim=-1), -1.0, 1.0)
        winding = torch.atan2(num, den).sum(dim=-1)
        inside = torch.abs(winding) > math.pi

    if inclusive:
        atol_rad = math.radians(atol_deg)
        pv = torch.einsum("mj,nj->mn", points, vertices)
        proj = vertices.unsqueeze(0) - points.unsqueeze(1) * pv.unsqueeze(-1)
        proj_norm = torch.linalg.norm(proj, dim=-1)
        near_vertex = proj_norm < 1e-14
        on_edge = _points_on_edges(points, vertices, atol_rad).any(dim=-1)
        inside = inside | near_vertex.any(dim=-1) | on_edge

    if inside_lon_deg is not None and inside_lat_deg is not None:
        ref = spherical_polygon_contains(
            torch.as_tensor(inside_lon_deg, dtype=torch.float64),
            torch.as_tensor(inside_lat_deg, dtype=torch.float64),
            lon_poly,
            lat_poly,
            inclusive=True,
            atol_deg=atol_deg,
        )
        if not bool(torch.as_tensor(ref).item()):
            inside = ~inside

    return inside.reshape(shape)


def convex_polygon_contains(
    lon_deg: Tensor | float,
    lat_deg: Tensor | float,
    poly_lon_deg: Tensor | list[float],
    poly_lat_deg: Tensor | list[float],
    *,
    atol: float = 1e-12,
) -> Tensor:
    """Fast half-space point-in-convex-spherical-polygon predicate."""
    v_poly = lonlat_to_unit_xyz(torch.as_tensor(poly_lon_deg), torch.as_tensor(poly_lat_deg))
    if v_poly.ndim != 2 or v_poly.shape[0] < 3:
        raise ValueError("polygon must contain at least three vertices")

    v_next = torch.roll(v_poly, shifts=-1, dims=0)
    normals = torch.cross(v_poly, v_next, dim=-1)

    centroid = v_poly.sum(dim=0)
    centroid = centroid / torch.linalg.norm(centroid).clamp_min(1e-15)
    orient = torch.sign((normals * centroid).sum(dim=-1))
    orient = torch.where(orient == 0, torch.ones_like(orient), orient)
    normals = normals * orient.unsqueeze(-1)

    points = lonlat_to_unit_xyz(lon_deg, lat_deg)
    dots = torch.einsum("...j,kj->...k", points, normals)
    return torch.all(dots >= -atol, dim=-1)


def query_polygon_general(
    nside: int,
    lon_deg: Tensor | list[float],
    lat_deg: Tensor | list[float],
    *,
    nest: bool = False,
) -> Tensor:
    """
    Query HEALPix pixels inside a simple spherical polygon (convex or non-convex).

    Non-convex polygons are triangulated via gnomonic-plane ear clipping.
    """
    lon_t, lat_t = _normalize_polygon_vertices(lon_deg, lat_deg)
    vertices_xyz = lonlat_to_unit_xyz(lon_t, lat_t).to(dtype=torch.float64)

    if vertices_xyz.shape[0] == 3:
        return _healpix.query_polygon(nside, vertices_xyz, nest=nest)

    center = vertices_xyz.mean(dim=0)
    center = center / torch.linalg.norm(center).clamp_min(1e-15)
    xy = _gnomonic_project(vertices_xyz, center)
    triangles = _triangulate_simple_polygon_2d(xy)

    chunks: list[Tensor] = []
    for i, j, k in triangles:
        verts = vertices_xyz[[i, j, k]]
        chunks.append(_healpix.query_polygon(nside, verts, nest=nest))
    if not chunks:
        return torch.empty(0, dtype=torch.int64)
    return torch.unique(torch.cat(chunks)).to(torch.int64)


def spherical_polygons_intersect(
    lon1_deg: Tensor | list[float],
    lat1_deg: Tensor | list[float],
    lon2_deg: Tensor | list[float],
    lat2_deg: Tensor | list[float],
    *,
    inside1_lon_deg: Tensor | float | None = None,
    inside1_lat_deg: Tensor | float | None = None,
    inside2_lon_deg: Tensor | float | None = None,
    inside2_lat_deg: Tensor | float | None = None,
    atol_deg: float = 1e-8,
) -> bool:
    """
    Polygon intersection predicate for simple spherical polygons.

    Returns True if interiors intersect or boundaries cross/touch.
    """
    lon1, lat1 = _normalize_polygon_vertices(lon1_deg, lat1_deg)
    lon2, lat2 = _normalize_polygon_vertices(lon2_deg, lat2_deg)
    v1 = lonlat_to_unit_xyz(lon1, lat1).to(dtype=torch.float64)
    v2 = lonlat_to_unit_xyz(lon2, lat2).to(dtype=torch.float64)
    tol_rad = math.radians(atol_deg)

    v1_next = torch.roll(v1, shifts=-1, dims=0)
    v2_next = torch.roll(v2, shifts=-1, dims=0)

    edge_hits = _great_circle_arcs_intersect_matrix(v1, v1_next, v2, v2_next, tol_rad=tol_rad)
    if bool(edge_hits.any()):
        return True

    # If no edge crossings, containment of one representative vertex is sufficient.
    rep1_lon = inside1_lon_deg if inside1_lon_deg is not None else lon1[0]
    rep1_lat = inside1_lat_deg if inside1_lat_deg is not None else lat1[0]
    rep2_lon = inside2_lon_deg if inside2_lon_deg is not None else lon2[0]
    rep2_lat = inside2_lat_deg if inside2_lat_deg is not None else lat2[0]

    c12 = spherical_polygon_contains(
        rep1_lon,
        rep1_lat,
        lon2,
        lat2,
        inclusive=True,
        atol_deg=atol_deg,
        inside_lon_deg=inside2_lon_deg,
        inside_lat_deg=inside2_lat_deg,
    )
    if bool(torch.as_tensor(c12).item()):
        return True

    c21 = spherical_polygon_contains(
        rep2_lon,
        rep2_lat,
        lon1,
        lat1,
        inclusive=True,
        atol_deg=atol_deg,
        inside_lon_deg=inside1_lon_deg,
        inside_lat_deg=inside1_lat_deg,
    )
    if bool(torch.as_tensor(c21).item()):
        return True

    return False


@dataclass(frozen=True)
class PixelizedRegion:
    """Pixelized spherical region with set-like boolean operations."""

    nside: int
    nest: bool
    pixels: Tensor

    def __post_init__(self) -> None:
        pix = torch.as_tensor(self.pixels, dtype=torch.int64).reshape(-1)
        pix = torch.unique(pix)
        object.__setattr__(self, "pixels", pix)

    def _check_compatible(self, other: "PixelizedRegion") -> None:
        if self.nside != other.nside or self.nest != other.nest:
            raise ValueError("regions must have same nside and nest ordering")

    def union(self, other: "PixelizedRegion") -> "PixelizedRegion":
        self._check_compatible(other)
        return PixelizedRegion(self.nside, self.nest, torch.cat((self.pixels, other.pixels)))

    def intersection(self, other: "PixelizedRegion") -> "PixelizedRegion":
        self._check_compatible(other)
        return PixelizedRegion(self.nside, self.nest, self.pixels[torch.isin(self.pixels, other.pixels)])

    def difference(self, other: "PixelizedRegion") -> "PixelizedRegion":
        self._check_compatible(other)
        return PixelizedRegion(self.nside, self.nest, self.pixels[~torch.isin(self.pixels, other.pixels)])

    def area(self, *, degrees: bool = False) -> float:
        return float(self.pixels.numel()) * _healpix.nside2pixarea(self.nside, degrees=degrees)


@dataclass(frozen=True)
class RegionAreaEstimate:
    """Controlled-error area estimate summary over a multi-resolution nside ladder."""

    area_sr: float
    area_deg2: float
    nside: int
    pixels: int
    convergence_rel: float | None
    area_by_nside_sr: dict[str, float]
    pixels_by_nside: dict[str, int]


def _parse_nside_ladder(nsides: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    ladder = tuple(int(v) for v in nsides)
    if len(ladder) == 0:
        raise ValueError("nsides ladder must contain at least one nside")
    for i, n in enumerate(ladder):
        if n <= 0 or (n & (n - 1)) != 0:
            raise ValueError(f"invalid nside {n}: must be a positive power of two")
        if i > 0 and n <= ladder[i - 1]:
            raise ValueError("nsides ladder must be strictly increasing")
    return ladder


def _estimate_area_from_query(
    query_fn,
    *,
    nsides: tuple[int, ...] | list[int],
    nest: bool,
) -> RegionAreaEstimate:
    ladder = _parse_nside_ladder(nsides)
    area_by_nside_sr: dict[str, float] = {}
    pixels_by_nside: dict[str, int] = {}
    prev_area = None

    for nside in ladder:
        pix = torch.as_tensor(query_fn(nside, nest=nest), dtype=torch.int64).reshape(-1)
        pix = torch.unique(pix)
        count = int(pix.numel())
        area_sr = float(count) * _healpix.nside2pixarea(nside, degrees=False)
        area_by_nside_sr[str(nside)] = area_sr
        pixels_by_nside[str(nside)] = count
        prev_area = area_sr

    assert prev_area is not None
    convergence_rel = None
    if len(ladder) >= 2:
        a_prev = area_by_nside_sr[str(ladder[-2])]
        convergence_rel = abs(prev_area - a_prev) / max(abs(prev_area), 1e-15)
    return RegionAreaEstimate(
        area_sr=prev_area,
        area_deg2=prev_area * (180.0 / math.pi) ** 2,
        nside=ladder[-1],
        pixels=pixels_by_nside[str(ladder[-1])],
        convergence_rel=convergence_rel,
        area_by_nside_sr=area_by_nside_sr,
        pixels_by_nside=pixels_by_nside,
    )


@dataclass(frozen=True)
class SphericalPolygon:
    """Simple spherical polygon with signed-area and containment contracts."""

    lon_deg: Tensor
    lat_deg: Tensor
    inside_lon_deg: float | None = None
    inside_lat_deg: float | None = None

    def __post_init__(self) -> None:
        lon_t, lat_t = _normalize_polygon_vertices(self.lon_deg, self.lat_deg)
        object.__setattr__(self, "lon_deg", lon_t)
        object.__setattr__(self, "lat_deg", lat_t)
        if (self.inside_lon_deg is None) ^ (self.inside_lat_deg is None):
            raise ValueError("inside_lon_deg and inside_lat_deg must both be set or both be None")

    def signed_area(self, *, degrees: bool = False) -> Tensor:
        return spherical_polygon_signed_area(self.lon_deg, self.lat_deg, degrees=degrees)

    def area(self, *, degrees: bool = False, oriented: bool = False) -> Tensor:
        return spherical_polygon_area(self.lon_deg, self.lat_deg, degrees=degrees, oriented=oriented)

    def contains(self, lon_deg: Tensor | float, lat_deg: Tensor | float, *, inclusive: bool = True, atol_deg: float = 1e-10) -> Tensor:
        return spherical_polygon_contains(
            lon_deg,
            lat_deg,
            self.lon_deg,
            self.lat_deg,
            inclusive=inclusive,
            atol_deg=atol_deg,
            inside_lon_deg=self.inside_lon_deg,
            inside_lat_deg=self.inside_lat_deg,
        )

    def query_pixels(self, nside: int, *, nest: bool = False) -> Tensor:
        return query_polygon_general(nside, self.lon_deg, self.lat_deg, nest=nest)

    def pixelize(self, nside: int, *, nest: bool = False) -> PixelizedRegion:
        return PixelizedRegion(nside=nside, nest=nest, pixels=self.query_pixels(nside, nest=nest))

    def area_estimate(
        self,
        *,
        nsides: tuple[int, ...] | list[int] = (128, 256, 512, 1024),
        nest: bool = False,
    ) -> RegionAreaEstimate:
        return _estimate_area_from_query(self.query_pixels, nsides=nsides, nest=nest)

    def intersects(self, other: "SphericalPolygon", *, atol_deg: float = 1e-8) -> bool:
        return spherical_polygons_intersect(
            self.lon_deg,
            self.lat_deg,
            other.lon_deg,
            other.lat_deg,
            inside1_lon_deg=self.inside_lon_deg,
            inside1_lat_deg=self.inside_lat_deg,
            inside2_lon_deg=other.inside_lon_deg,
            inside2_lat_deg=other.inside_lat_deg,
            atol_deg=atol_deg,
        )

    def union(self, other: "RegionOperand") -> "SphericalBooleanRegion":
        return SphericalBooleanRegion(op="union", left=self, right=other)

    def intersection(self, other: "RegionOperand") -> "SphericalBooleanRegion":
        return SphericalBooleanRegion(op="intersection", left=self, right=other)

    def difference(self, other: "RegionOperand") -> "SphericalBooleanRegion":
        return SphericalBooleanRegion(op="difference", left=self, right=other)

    def to_exact(self, *, cap_steps: int = 256) -> "ExactSphericalRegion":
        return to_exact_region(self, cap_steps=cap_steps)

    def exact_union(self, other: "RegionOperand", *, cap_steps: int = 256) -> "ExactSphericalRegion":
        return self.to_exact(cap_steps=cap_steps).union(other, cap_steps=cap_steps)

    def exact_intersection(self, other: "RegionOperand", *, cap_steps: int = 256) -> "ExactSphericalRegion":
        return self.to_exact(cap_steps=cap_steps).intersection(other, cap_steps=cap_steps)

    def exact_difference(self, other: "RegionOperand", *, cap_steps: int = 256) -> "ExactSphericalRegion":
        return self.to_exact(cap_steps=cap_steps).difference(other, cap_steps=cap_steps)


@dataclass(frozen=True)
class SphericalMultiPolygon:
    """Collection of spherical polygons with boolean-style predicates."""

    polygons: tuple[SphericalPolygon, ...]

    def __post_init__(self) -> None:
        polys = tuple(self.polygons)
        if len(polys) == 0:
            raise ValueError("SphericalMultiPolygon requires at least one polygon")
        object.__setattr__(self, "polygons", polys)

    def area(self, *, degrees: bool = False, oriented: bool = False) -> Tensor:
        vals = [p.area(degrees=degrees, oriented=oriented) for p in self.polygons]
        out = vals[0]
        for v in vals[1:]:
            out = out + v
        return out

    def contains(self, lon_deg: Tensor | float, lat_deg: Tensor | float, *, inclusive: bool = True, atol_deg: float = 1e-10) -> Tensor:
        out = self.polygons[0].contains(lon_deg, lat_deg, inclusive=inclusive, atol_deg=atol_deg)
        for p in self.polygons[1:]:
            out = out | p.contains(lon_deg, lat_deg, inclusive=inclusive, atol_deg=atol_deg)
        return out

    def intersects(self, other: "SphericalPolygon | SphericalMultiPolygon", *, atol_deg: float = 1e-8) -> bool:
        others = other.polygons if isinstance(other, SphericalMultiPolygon) else (other,)
        for p in self.polygons:
            for q in others:
                if p.intersects(q, atol_deg=atol_deg):
                    return True
        return False

    def query_pixels(self, nside: int, *, nest: bool = False) -> Tensor:
        chunks = [p.query_pixels(nside, nest=nest) for p in self.polygons]
        if not chunks:
            return torch.empty(0, dtype=torch.int64)
        return torch.unique(torch.cat(chunks))

    def pixelize(self, nside: int, *, nest: bool = False) -> PixelizedRegion:
        return PixelizedRegion(nside=nside, nest=nest, pixels=self.query_pixels(nside, nest=nest))

    def area_estimate(
        self,
        *,
        nsides: tuple[int, ...] | list[int] = (128, 256, 512, 1024),
        nest: bool = False,
    ) -> RegionAreaEstimate:
        return _estimate_area_from_query(self.query_pixels, nsides=nsides, nest=nest)

    def union(self, other: "RegionOperand") -> "SphericalBooleanRegion":
        return SphericalBooleanRegion(op="union", left=self, right=other)

    def intersection(self, other: "RegionOperand") -> "SphericalBooleanRegion":
        return SphericalBooleanRegion(op="intersection", left=self, right=other)

    def difference(self, other: "RegionOperand") -> "SphericalBooleanRegion":
        return SphericalBooleanRegion(op="difference", left=self, right=other)

    def to_exact(self, *, cap_steps: int = 256) -> "ExactSphericalRegion":
        return to_exact_region(self, cap_steps=cap_steps)

    def exact_union(self, other: "RegionOperand", *, cap_steps: int = 256) -> "ExactSphericalRegion":
        return self.to_exact(cap_steps=cap_steps).union(other, cap_steps=cap_steps)

    def exact_intersection(self, other: "RegionOperand", *, cap_steps: int = 256) -> "ExactSphericalRegion":
        return self.to_exact(cap_steps=cap_steps).intersection(other, cap_steps=cap_steps)

    def exact_difference(self, other: "RegionOperand", *, cap_steps: int = 256) -> "ExactSphericalRegion":
        return self.to_exact(cap_steps=cap_steps).difference(other, cap_steps=cap_steps)


@dataclass(frozen=True)
class SphericalCap:
    """Spherical cap centered at (lon, lat) with angular radius in degrees."""

    lon_deg: float
    lat_deg: float
    radius_deg: float

    def contains(self, lon_deg: Tensor | float, lat_deg: Tensor | float, *, inclusive: bool = True) -> Tensor:
        dist = great_circle_distance(self.lon_deg, self.lat_deg, lon_deg, lat_deg, degrees=True)
        if inclusive:
            return dist <= self.radius_deg
        return dist < self.radius_deg

    def query_pixels(self, nside: int, *, nest: bool = False) -> Tensor:
        return _healpix.query_circle(
            nside,
            self.lon_deg,
            self.lat_deg,
            self.radius_deg,
            degrees=True,
            nest=nest,
        )

    def pixelize(self, nside: int, *, nest: bool = False) -> PixelizedRegion:
        return PixelizedRegion(nside=nside, nest=nest, pixels=self.query_pixels(nside, nest=nest))

    def area_estimate(
        self,
        *,
        nsides: tuple[int, ...] | list[int] = (128, 256, 512, 1024),
        nest: bool = False,
    ) -> RegionAreaEstimate:
        return _estimate_area_from_query(self.query_pixels, nsides=nsides, nest=nest)

    def union(self, other: "RegionOperand") -> "SphericalBooleanRegion":
        return SphericalBooleanRegion(op="union", left=self, right=other)

    def intersection(self, other: "RegionOperand") -> "SphericalBooleanRegion":
        return SphericalBooleanRegion(op="intersection", left=self, right=other)

    def difference(self, other: "RegionOperand") -> "SphericalBooleanRegion":
        return SphericalBooleanRegion(op="difference", left=self, right=other)

    def to_exact(self, *, cap_steps: int = 256) -> "ExactSphericalRegion":
        return to_exact_region(self, cap_steps=cap_steps)

    def exact_union(self, other: "RegionOperand", *, cap_steps: int = 256) -> "ExactSphericalRegion":
        return self.to_exact(cap_steps=cap_steps).union(other, cap_steps=cap_steps)

    def exact_intersection(self, other: "RegionOperand", *, cap_steps: int = 256) -> "ExactSphericalRegion":
        return self.to_exact(cap_steps=cap_steps).intersection(other, cap_steps=cap_steps)

    def exact_difference(self, other: "RegionOperand", *, cap_steps: int = 256) -> "ExactSphericalRegion":
        return self.to_exact(cap_steps=cap_steps).difference(other, cap_steps=cap_steps)


RegionOperand = object


def _spherical_geometry_available() -> bool:
    try:
        from spherical_geometry.polygon import SphericalPolygon as _  # noqa: F401
    except Exception:
        return False
    return True


def _require_spherical_geometry():
    try:
        from spherical_geometry.polygon import SphericalPolygon as _SGPolygon
    except Exception as exc:  # pragma: no cover - optional backend
        raise RuntimeError("spherical-geometry is required for exact region boolean operations") from exc
    return _SGPolygon


def _sg_from_lonlat(lon_deg: Tensor, lat_deg: Tensor):
    sg_poly = _require_spherical_geometry()
    lon_np = lon_deg.detach().cpu().numpy().astype(np.float64)
    lat_np = lat_deg.detach().cpu().numpy().astype(np.float64)
    if hasattr(sg_poly, "from_lonlat"):
        return sg_poly.from_lonlat(lon_np, lat_np, degrees=True)
    return sg_poly.from_radec(lon_np, lat_np, degrees=True)


def _strip_closed_np(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if lon.size >= 4 and np.isclose(lon[0], lon[-1]) and np.isclose(lat[0], lat[-1]):
        return lon[:-1], lat[:-1]
    return lon, lat


def _region_to_sg(region: RegionOperand, *, cap_steps: int = 256):
    if isinstance(region, NativeExactSphericalRegion):
        return _region_to_sg(region._region, cap_steps=cap_steps)
    if isinstance(region, ExactSphericalRegion):
        return region._sg_poly
    if isinstance(region, SphericalPolygon):
        return _sg_from_lonlat(region.lon_deg, region.lat_deg)
    if isinstance(region, SphericalMultiPolygon):
        sg_poly = _require_spherical_geometry()
        polys = [_sg_from_lonlat(p.lon_deg, p.lat_deg) for p in region.polygons]
        return sg_poly.multi_union(polys)
    if isinstance(region, SphericalCap):
        sg_poly = _require_spherical_geometry()
        return sg_poly.from_cone(region.lon_deg, region.lat_deg, region.radius_deg, degrees=True, steps=cap_steps)
    if isinstance(region, SphericalBooleanRegion):
        return region.to_exact(cap_steps=cap_steps)._sg_poly
    raise TypeError(f"unsupported region type for exact conversion: {type(region)!r}")


def _region_contains(
    region: RegionOperand,
    lon_deg: Tensor | float,
    lat_deg: Tensor | float,
    *,
    inclusive: bool = True,
    atol_deg: float = 1e-10,
) -> Tensor:
    if isinstance(region, SphericalCap):
        return region.contains(lon_deg, lat_deg, inclusive=inclusive)
    if isinstance(region, SphericalBooleanRegion):
        return region.contains(lon_deg, lat_deg, inclusive=inclusive, atol_deg=atol_deg)
    return region.contains(lon_deg, lat_deg, inclusive=inclusive, atol_deg=atol_deg)


def _region_pixelize(region: RegionOperand, nside: int, nest: bool) -> PixelizedRegion:
    return region.pixelize(nside=nside, nest=nest)


@dataclass(frozen=True)
class SphericalBooleanRegion:
    """Boolean composition of spherical regions with controlled-error HEALPix evaluation."""

    op: Literal["union", "intersection", "difference"]
    left: RegionOperand
    right: RegionOperand

    def __post_init__(self) -> None:
        if self.op not in {"union", "intersection", "difference"}:
            raise ValueError("op must be one of {'union', 'intersection', 'difference'}")

    def contains(
        self,
        lon_deg: Tensor | float,
        lat_deg: Tensor | float,
        *,
        inclusive: bool = True,
        atol_deg: float = 1e-10,
    ) -> Tensor:
        left = _region_contains(self.left, lon_deg, lat_deg, inclusive=inclusive, atol_deg=atol_deg)
        right = _region_contains(self.right, lon_deg, lat_deg, inclusive=inclusive, atol_deg=atol_deg)
        if self.op == "union":
            return left | right
        if self.op == "intersection":
            return left & right
        return left & (~right)

    def query_pixels(self, nside: int, *, nest: bool = False) -> Tensor:
        left = _region_pixelize(self.left, nside, nest)
        right = _region_pixelize(self.right, nside, nest)
        if self.op == "union":
            return left.union(right).pixels
        if self.op == "intersection":
            return left.intersection(right).pixels
        return left.difference(right).pixels

    def pixelize(self, nside: int, *, nest: bool = False) -> PixelizedRegion:
        return PixelizedRegion(nside=nside, nest=nest, pixels=self.query_pixels(nside, nest=nest))

    def area_estimate(
        self,
        *,
        nsides: tuple[int, ...] | list[int] = (128, 256, 512, 1024),
        nest: bool = False,
    ) -> RegionAreaEstimate:
        return _estimate_area_from_query(self.query_pixels, nsides=nsides, nest=nest)

    def union(self, other: RegionOperand) -> "SphericalBooleanRegion":
        return SphericalBooleanRegion(op="union", left=self, right=other)

    def intersection(self, other: RegionOperand) -> "SphericalBooleanRegion":
        return SphericalBooleanRegion(op="intersection", left=self, right=other)

    def difference(self, other: RegionOperand) -> "SphericalBooleanRegion":
        return SphericalBooleanRegion(op="difference", left=self, right=other)

    def to_exact(self, *, cap_steps: int = 256) -> "ExactSphericalRegion":
        left = _region_to_sg(self.left, cap_steps=cap_steps)
        right = _region_to_sg(self.right, cap_steps=cap_steps)
        if self.op == "union":
            out = left.union(right)
        elif self.op == "intersection":
            out = left.intersection(right)
        else:
            out = left.intersection(right.invert_polygon())
        return ExactSphericalRegion(out)

    def exact_area(self, *, degrees: bool = False, cap_steps: int = 256) -> float:
        return self.to_exact(cap_steps=cap_steps).area(degrees=degrees)


@dataclass(frozen=True)
class NativeExactSphericalRegion:
    """
    Torch-native exact-region scaffold with controlled-error boolean semantics.

    This backend is dependency-free and deterministic, and serves as the native
    fallback when `spherical-geometry` is unavailable.
    """

    _region: RegionOperand
    _nsides: tuple[int, ...] = (256, 512, 1024, 2048)

    @property
    def is_empty(self) -> bool:
        probe_nside = int(self._nsides[0]) if len(self._nsides) > 0 else 256
        return int(self.query_pixels(probe_nside, nest=False).numel()) == 0

    def area(self, *, degrees: bool = False) -> float:
        est = _estimate_area_from_query(self.query_pixels, nsides=self._nsides, nest=False)
        return float(est.area_deg2 if degrees else est.area_sr)

    def intersects(self, other: RegionOperand, *, cap_steps: int = 256) -> bool:
        del cap_steps
        nside = int(self._nsides[-1]) if len(self._nsides) > 0 else 1024
        a = self.query_pixels(nside, nest=False)
        if a.numel() == 0:
            return False
        other_native = to_exact_region(other, backend="native", cap_steps=256, nsides=self._nsides)
        b = other_native.query_pixels(nside, nest=False)
        if b.numel() == 0:
            return False
        return bool(torch.isin(a, b).any().item())

    def contains(self, lon_deg: Tensor | float, lat_deg: Tensor | float) -> Tensor:
        return _region_contains(self._region, lon_deg, lat_deg, inclusive=True, atol_deg=1e-10)

    def query_pixels(self, nside: int, *, nest: bool = False) -> Tensor:
        return _region_pixelize(self._region, nside=nside, nest=nest).pixels

    def pixelize(self, nside: int, *, nest: bool = False) -> PixelizedRegion:
        return PixelizedRegion(nside=nside, nest=nest, pixels=self.query_pixels(nside, nest=nest))

    def union(self, other: RegionOperand, *, cap_steps: int = 256) -> "NativeExactSphericalRegion":
        del cap_steps
        return NativeExactSphericalRegion(
            SphericalBooleanRegion(op="union", left=self._region, right=other),
            _nsides=self._nsides,
        )

    def intersection(self, other: RegionOperand, *, cap_steps: int = 256) -> "NativeExactSphericalRegion":
        del cap_steps
        return NativeExactSphericalRegion(
            SphericalBooleanRegion(op="intersection", left=self._region, right=other),
            _nsides=self._nsides,
        )

    def difference(self, other: RegionOperand, *, cap_steps: int = 256) -> "NativeExactSphericalRegion":
        del cap_steps
        return NativeExactSphericalRegion(
            SphericalBooleanRegion(op="difference", left=self._region, right=other),
            _nsides=self._nsides,
        )

    def to_multipolygon(self) -> SphericalMultiPolygon:
        if isinstance(self._region, SphericalPolygon):
            return SphericalMultiPolygon((self._region,))
        if isinstance(self._region, SphericalMultiPolygon):
            return self._region
        raise ValueError("native exact region cannot be converted to multipolygon for this operand type")


@dataclass(frozen=True)
class ExactSphericalRegion:
    """Exact spherical region wrapper backed by optional `spherical-geometry`."""

    _sg_poly: object

    @property
    def is_empty(self) -> bool:
        return len(list(self._sg_poly)) == 0

    def area(self, *, degrees: bool = False) -> float:
        area_sr = float(self._sg_poly.area())
        if degrees:
            return area_sr * (180.0 / math.pi) ** 2
        return area_sr

    def intersects(self, other: RegionOperand, *, cap_steps: int = 256) -> bool:
        other_sg = _region_to_sg(other, cap_steps=cap_steps)
        return bool(self._sg_poly.intersects_poly(other_sg))

    def contains(self, lon_deg: Tensor | float, lat_deg: Tensor | float) -> Tensor:
        lon_t = torch.as_tensor(lon_deg, dtype=torch.float64)
        lat_t = torch.as_tensor(lat_deg, dtype=torch.float64)
        lon_t, lat_t = torch.broadcast_tensors(lon_t, lat_t)
        shape = lon_t.shape
        lon_f = lon_t.reshape(-1).detach().cpu().numpy()
        lat_f = lat_t.reshape(-1).detach().cpu().numpy()
        out = np.asarray(
            [bool(self._sg_poly.contains_lonlat(float(lo), float(la), degrees=True)) for lo, la in zip(lon_f, lat_f, strict=False)],
            dtype=bool,
        )
        return torch.from_numpy(out).reshape(shape)

    def query_pixels(self, nside: int, *, nest: bool = False) -> Tensor:
        # Candidate pixels from per-component polygon queries, then exact center filtering.
        chunks: list[Tensor] = []
        for lon, lat in self._sg_poly.to_lonlat():
            lon_np = np.asarray(lon, dtype=np.float64)
            lat_np = np.asarray(lat, dtype=np.float64)
            lon_np, lat_np = _strip_closed_np(lon_np, lat_np)
            if lon_np.size < 3:
                continue
            pix = query_polygon_general(
                nside,
                torch.from_numpy(lon_np),
                torch.from_numpy(lat_np),
                nest=nest,
            )
            if pix.numel() > 0:
                chunks.append(pix)
        if not chunks:
            return torch.empty(0, dtype=torch.int64)
        candidate = torch.unique(torch.cat(chunks))
        lon_c, lat_c = _healpix.pix2ang(nside, candidate, nest=nest, lonlat=True)
        keep = self.contains(lon_c, lat_c)
        return candidate[keep]

    def pixelize(self, nside: int, *, nest: bool = False) -> PixelizedRegion:
        return PixelizedRegion(nside=nside, nest=nest, pixels=self.query_pixels(nside, nest=nest))

    def union(self, other: RegionOperand, *, cap_steps: int = 256) -> "ExactSphericalRegion":
        other_sg = _region_to_sg(other, cap_steps=cap_steps)
        return ExactSphericalRegion(self._sg_poly.union(other_sg))

    def intersection(self, other: RegionOperand, *, cap_steps: int = 256) -> "ExactSphericalRegion":
        other_sg = _region_to_sg(other, cap_steps=cap_steps)
        return ExactSphericalRegion(self._sg_poly.intersection(other_sg))

    def difference(self, other: RegionOperand, *, cap_steps: int = 256) -> "ExactSphericalRegion":
        other_sg = _region_to_sg(other, cap_steps=cap_steps)
        return ExactSphericalRegion(self._sg_poly.intersection(other_sg.invert_polygon()))

    def to_multipolygon(self) -> SphericalMultiPolygon:
        polys: list[SphericalPolygon] = []
        for lon, lat in self._sg_poly.to_lonlat():
            lon_np = np.asarray(lon, dtype=np.float64)
            lat_np = np.asarray(lat, dtype=np.float64)
            lon_np, lat_np = _strip_closed_np(lon_np, lat_np)
            if lon_np.size >= 3:
                polys.append(SphericalPolygon(torch.from_numpy(lon_np), torch.from_numpy(lat_np)))
        if not polys:
            raise ValueError("exact region is empty; cannot convert to SphericalMultiPolygon")
        return SphericalMultiPolygon(tuple(polys))


def to_exact_region(
    region: RegionOperand,
    *,
    cap_steps: int = 256,
    backend: Literal["auto", "spherical-geometry", "native"] = "auto",
    nsides: tuple[int, ...] | list[int] = (256, 512, 1024, 2048),
) -> ExactSphericalRegion | NativeExactSphericalRegion:
    """
    Convert a torchfits spherical region into an exact-region backend.

    `backend='auto'` prefers `spherical-geometry` when available and otherwise
    falls back to the torch-native controlled-error backend.
    """
    if backend not in {"auto", "spherical-geometry", "native"}:
        raise ValueError("backend must be one of {'auto', 'spherical-geometry', 'native'}")
    if backend == "native":
        return NativeExactSphericalRegion(region, tuple(int(x) for x in nsides))
    if backend == "spherical-geometry":
        return ExactSphericalRegion(_region_to_sg(region, cap_steps=cap_steps))
    if _spherical_geometry_available():
        return ExactSphericalRegion(_region_to_sg(region, cap_steps=cap_steps))
    return NativeExactSphericalRegion(region, tuple(int(x) for x in nsides))


def query_ellipse(
    nside: int,
    lon_deg: float | Tensor,
    lat_deg: float | Tensor,
    semi_major_deg: float,
    semi_minor_deg: float,
    *,
    pa_deg: float = 0.0,
    nest: bool = False,
    inclusive: bool = False,
) -> Tensor:
    """
    Query pixels inside a sky ellipse.

    The ellipse is defined in geodesic polar coordinates around the center.
    `pa_deg` is position angle East of North.
    """
    if semi_major_deg <= 0.0 or semi_minor_deg <= 0.0:
        raise ValueError("semi_major_deg and semi_minor_deg must be positive")
    if semi_minor_deg > semi_major_deg:
        raise ValueError("semi_minor_deg must be <= semi_major_deg")

    # Spherical ellipse in focal form:
    # d(P, F1) + d(P, F2) <= 2a, where a is the semi-major axis.
    major = float(math.radians(semi_major_deg))
    minor = float(math.radians(semi_minor_deg))
    cos_ratio = math.cos(major) / max(math.cos(minor), 1e-15)
    cos_ratio = min(1.0, max(-1.0, cos_ratio))
    focal_sep = math.acos(cos_ratio)
    pa = math.radians(pa_deg)

    lon_f1, lat_f1 = _destination_lonlat_deg(lon_deg, lat_deg, pa, focal_sep)
    lon_f2, lat_f2 = _destination_lonlat_deg(lon_deg, lat_deg, pa + math.pi, focal_sep)

    pixrad = _healpix.max_pixrad(nside, degrees=False) if inclusive else 0.0
    candidates = _healpix.query_circle(
        nside,
        lon_deg,
        lat_deg,
        semi_major_deg + math.degrees(pixrad),
        degrees=True,
        nest=nest,
        inclusive=False,
    )
    if candidates.numel() == 0:
        return candidates

    lon_p, lat_p = _healpix.pix2ang(nside, candidates, nest=nest, lonlat=True)
    d1 = great_circle_distance(lon_f1, lat_f1, lon_p, lat_p, degrees=False)
    d2 = great_circle_distance(lon_f2, lat_f2, lon_p, lat_p, degrees=False)

    # For inclusive overlap, any point in the pixel can lie up to pixrad from center.
    thresh = (2.0 * major) + (2.0 * pixrad) + 1e-12
    keep = (d1 + d2) <= thresh
    return candidates[keep]


__all__ = [
    "ExactSphericalRegion",
    "NativeExactSphericalRegion",
    "PixelizedRegion",
    "RegionAreaEstimate",
    "SphericalBooleanRegion",
    "SphericalCap",
    "SphericalMultiPolygon",
    "SphericalPolygon",
    "convex_polygon_contains",
    "query_ellipse",
    "query_polygon_general",
    "spherical_polygon_area",
    "spherical_polygon_contains",
    "spherical_polygons_intersect",
    "spherical_polygon_signed_area",
    "spherical_triangle_area",
    "to_exact_region",
]
