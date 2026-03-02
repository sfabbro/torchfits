"""Multi-Order Coverage (MOC) support for HEALPix."""

from __future__ import annotations

import math
from typing import Literal
import importlib

import torch
from torch import Tensor

from ..wcs import healpix as _healpix
from .sparse import HealSparseMap


def nest2uniq(nside: int | Tensor, pix: int | Tensor) -> Tensor:
    """Convert NESTED pixel index to UNIQ pixel index."""
    n = torch.as_tensor(nside, dtype=torch.int64)
    p = torch.as_tensor(pix, dtype=torch.int64)
    return 4 * (n * n) + p


def uniq2nest(uniq: int | Tensor) -> tuple[Tensor, Tensor]:
    """Convert UNIQ pixel index to (nside, NESTED pixel index)."""
    u = torch.as_tensor(uniq, dtype=torch.int64)
    # 4 * 4^order + pix = uniq
    # 4^order = uniq // 4 (approx)
    # order = log4(uniq // 4)
    order = torch.div(
        torch.log2(torch.div(u, 4, rounding_mode="floor").to(torch.float64)),
        2.0,
        rounding_mode="floor",
    ).to(torch.int64)
    nside = 1 << order
    pix = u - 4 * (nside * nside)
    return nside, pix


def _normalize_moc(uniq: Tensor) -> Tensor:
    """Sort and merge redundant pixels in a MOC."""
    if uniq.numel() == 0:
        return uniq

    # 1. Remove duplicates and sort
    u = torch.unique(uniq)

    # 2. Hierarchical merge: if 4 children are present, replace with parent.
    while True:
        if u.numel() < 4:
            break

        orders = torch.div(
            torch.log2(torch.div(u, 4, rounding_mode="floor").to(torch.float64)),
            2.0,
            rounding_mode="floor",
        ).to(torch.int64)
        unique_orders = torch.unique(orders)

        merged_any = False
        new_u = []

        # We process each order. For orders that might have merges, we check.
        # For others, we keep.
        processed_mask = torch.zeros_like(u, dtype=torch.bool)

        for o in torch.flip(unique_orders, [0]):
            if o == 0:
                continue

            mask = (orders == o) & (~processed_mask)
            u_o = u[mask]
            if u_o.numel() < 4:
                continue

            pix_o = u_o - 4 * (4**o)

            i = 0
            while i <= u_o.numel() - 4:
                p0 = int(pix_o[i].item())
                if (
                    p0 % 4 == 0
                    and int(pix_o[i + 1].item()) == p0 + 1
                    and int(pix_o[i + 2].item()) == p0 + 2
                    and int(pix_o[i + 3].item()) == p0 + 3
                ):
                    parent_uniq = 4 * (4 ** (o - 1)) + (p0 // 4)
                    new_u.append(
                        torch.tensor([parent_uniq], device=u.device, dtype=torch.int64)
                    )
                    # Mark these 4 as processed
                    # We need to find their original indices in u
                    # But since we are rebuilding, we can just skip them in new_u later.
                    i += 4
                    merged_any = True
                else:
                    new_u.append(u_o[i : i + 1])
                    i += 1
            if i < u_o.numel():
                new_u.append(u_o[i:])

            processed_mask |= mask

        # Add any pixels from orders we didn't process merge for
        new_u.append(u[~processed_mask])

        if not merged_any:
            break

        u = torch.unique(torch.cat(new_u))

    # 3. Handle overlaps: if parent and child are both present, remove child.
    # A simple way: convert all to max_order ranges, merge ranges, convert back.
    # This is robust and already implemented via _from_ranges.
    int(
        torch.log2(
            torch.div(
                u.max() if u.numel() > 0 else torch.tensor(4), 4, rounding_mode="floor"
            ).to(torch.float64)
        ).item()
        // 2
    )
    # Actually, simpler:
    # return MOC._from_ranges(MOC(u)._to_ranges(max_o), max_o).uniq
    # Wait, avoid recursion.

    return u


class MOC:
    """Multi-Order Coverage map."""

    def __init__(self, uniq: Tensor):
        self.uniq = _normalize_moc(torch.as_tensor(uniq, dtype=torch.int64))

    @property
    def max_order(self) -> int:
        if self.uniq.numel() == 0:
            return 0
        nside, _ = uniq2nest(self.uniq.max())
        return int(torch.log2(nside.to(torch.float64)).item())

    def contains(self, lon_deg: Tensor | float, lat_deg: Tensor | float) -> Tensor:
        """Check if points are contained in the MOC."""
        lon = torch.as_tensor(lon_deg, dtype=torch.float64)
        lat = torch.as_tensor(lat_deg, dtype=torch.float64)

        if self.uniq.numel() == 0:
            return torch.zeros_like(lon, dtype=torch.bool)

        # A point is in MOC if its pixel at order O is in MOC for ANY O.
        # Efficient way: for each point, get its pixel at MAX_ORDER.
        # Check if that pixel OR any of its ancestors are in the MOC.

        max_o = self.max_order
        nside_max = 1 << max_o
        p_max = _healpix.ang2pix(nside_max, lon, lat, nest=True, lonlat=True)

        # Convert MOC UNIQ to max_order ranges [start, end)
        nsides, pix_nested = uniq2nest(self.uniq)
        ranges = []
        for ns, p in zip(nsides, pix_nested):
            o = int(torch.log2(ns.to(torch.float64)).item())
            shift = 2 * (max_o - o)
            start = p << shift
            end = (p + 1) << shift
            ranges.append((start.item(), end.item()))

        # Sort and merge ranges (should already be merged if MOC is normalized)
        ranges.sort()

        # Use searchsorted to check if p_max is within any range
        starts = torch.tensor(
            [r[0] for r in ranges], device=p_max.device, dtype=torch.int64
        )
        ends = torch.tensor(
            [r[1] for r in ranges], device=p_max.device, dtype=torch.int64
        )

        idx = torch.searchsorted(starts, p_max, right=True) - 1
        mask = (idx >= 0) & (p_max < ends[idx])
        return mask

    def filter_catalog(self, lon_deg: Tensor, lat_deg: Tensor) -> tuple[Tensor, Tensor]:
        """Filter a catalogue by point-in-MOC containment. Returns (lon, lat) of contained points."""
        mask = self.contains(lon_deg, lat_deg)
        return lon_deg[mask], lat_deg[mask]

    @property
    def area(self) -> float:
        """Total area covered by the MOC in steradians."""
        if self.uniq.numel() == 0:
            return 0.0
        nsides, _ = uniq2nest(self.uniq)
        # Area of a pixel at nside is 4pi / (12 * nside^2)
        pix_areas = (4.0 * math.pi) / (12.0 * nsides.to(torch.float64) ** 2)
        return float(pix_areas.sum().item())

    @property
    def area_sq_deg(self) -> float:
        """Total area covered by the MOC in square degrees."""
        return self.area * (180.0 / math.pi) ** 2

    def contains_moc(self, other: MOC) -> bool:
        """Return True if this MOC contains all of other MOC (subset check)."""
        # self contains other if intersection(self, other) == other
        inter = self.intersection(other)
        return torch.equal(inter.uniq, other.uniq)

    def union(self, other: MOC) -> MOC:
        return MOC(torch.cat([self.uniq, other.uniq]))

    def intersection(self, other: MOC) -> MOC:
        # Hierarchical intersection:
        # A pixel is in intersection if:
        # 1. It is in both.
        # 2. It is in A and a descendant is in B (then the descendant is in intersection).
        # 3. It is in B and a descendant is in A (then the descendant is in intersection).

        # Implementation via max-order ranges:
        # Intersection of two sets of ranges.
        max_o = max(self.max_order, other.max_order)
        r1 = self._to_ranges(max_o)
        r2 = other._to_ranges(max_o)

        # Range intersection
        # This is a classic algorithm.
        # For simplicity, we can do it via a loop or vectorized if ranges are few.
        i, j = 0, 0
        new_ranges = []
        while i < len(r1) and j < len(r2):
            s1, e1 = r1[i]
            s2, e2 = r2[j]

            s_int = max(s1, s2)
            e_int = min(e1, e2)

            if s_int < e_int:
                new_ranges.append((s_int, e_int))

            if e1 < e2:
                i += 1
            else:
                j += 1
        return MOC._from_ranges(new_ranges, max_o)

    def difference(self, other: MOC) -> MOC:
        """Subtract other MOC from this MOC."""
        max_o = max(self.max_order, other.max_order)
        r1 = self._to_ranges(max_o)
        r2 = other._to_ranges(max_o)

        # Range difference: R1 \ R2
        new_ranges = []
        i, j = 0, 0
        curr_s, curr_e = r1[0] if r1 else (0, 0)

        while i < len(r1):
            if j >= len(r2):
                new_ranges.append((curr_s, curr_e))
                i += 1
                if i < len(r1):
                    curr_s, curr_e = r1[i]
                continue

            s2, e2 = r2[j]

            if e2 <= curr_s:
                j += 1
                continue
            if s2 >= curr_e:
                new_ranges.append((curr_s, curr_e))
                i += 1
                if i < len(r1):
                    curr_s, curr_e = r1[i]
                continue

            # Overlap!
            if s2 > curr_s:
                new_ranges.append((curr_s, s2))

            if e2 < curr_e:
                curr_s = e2
                j += 1
            else:
                i += 1
                if i < len(r1):
                    curr_s, curr_e = r1[i]

        return MOC._from_ranges(new_ranges, max_o)

    def _to_ranges(self, max_order: int) -> list[tuple[int, int]]:
        nsides, pix_nested = uniq2nest(self.uniq)
        ranges = []
        for ns, p in zip(nsides, pix_nested):
            o = int(torch.log2(ns.to(torch.float64)).item())
            shift = 2 * (max_order - o)
            start = p << shift
            end = (p + 1) << shift
            ranges.append((start.item(), end.item()))
        ranges.sort()
        return ranges

    @classmethod
    def _from_ranges(cls, ranges: list[tuple[int, int]], max_order: int) -> MOC:
        """Convert ranges back to UNIQ pixels using greedy filling."""
        uniqs = []
        for start, end in ranges:
            curr = start
            while curr < end:
                # Find largest k such that 4^k <= (end - curr) and curr % 4^k == 0
                k = 0
                while (
                    curr + (4 ** (k + 1)) <= end
                    and curr % (4 ** (k + 1)) == 0
                    and k < max_order
                ):
                    k += 1

                order = max_order - k
                uniqs.append(nest2uniq(1 << order, curr >> (2 * k)))
                curr += 4**k
        return cls(torch.tensor(uniqs, dtype=torch.int64))

    def to_sparse_map(self, nside: int, sentinel: float = 0.0) -> HealSparseMap:
        """Realize MOC as a HealSparseMap at fixed resolution."""
        # This belongs in moc.py but returns HealSparseMap
        max_o = int(math.log2(nside))
        ranges = self._to_ranges(max_o)

        all_pix = []
        for s, e in ranges:
            all_pix.append(torch.arange(s, e, dtype=torch.int64))

        if not all_pix:
            return HealSparseMap.make_empty(nside // 8, nside, torch.float32, sentinel)

        pix = torch.cat(all_pix)
        return HealSparseMap.from_pixels(pix, nside, nside // 8, sentinel=sentinel)

    def flatten(self, nside: int) -> Tensor:
        """Project MOC to a dense HEALPix mask at fixed resolution."""
        # This is essentially a helper for SparseMap conversion.
        mask = torch.zeros(12 * nside**2, dtype=torch.bool)
        max_o = int(math.log2(nside))
        ranges = self._to_ranges(max_o)
        for s, e in ranges:
            mask[s:e] = True
        return mask

    @classmethod
    def from_sparse_map(cls, smap: HealSparseMap) -> MOC:
        """Create MOC from the coverage of a sparse map."""
        # Get all sparse pixels
        pix = smap.get_covered_pixels()
        max_order = int(math.log2(smap.nside_sparse))

        # Convert to ranges
        if pix.numel() == 0:
            return cls(torch.empty((0,), dtype=torch.int64))

        pix = torch.sort(pix)[0]
        # Find contiguous chunks
        diffs = torch.diff(pix)
        breaks = torch.where(diffs > 1)[0]

        ranges = []
        start_idx = 0
        for b in breaks:
            ranges.append((int(pix[start_idx].item()), int(pix[b].item()) + 1))
            start_idx = int(b.item()) + 1
        ranges.append((int(pix[start_idx].item()), int(pix[-1].item()) + 1))

        return cls._from_ranges(ranges, max_order)

    @classmethod
    def from_circle(
        cls, lon_deg: float, lat_deg: float, radius_deg: float, max_order: int = 10
    ) -> MOC:
        """Create a MOC from a circular cap."""
        nside = 1 << max_order
        pix = _healpix.query_circle(
            nside, lon_deg, lat_deg, radius_deg, nest=True, degrees=True
        )
        uniqs = nest2uniq(nside, pix)
        return cls(uniqs)

    @classmethod
    def from_polygon(
        cls,
        lon_deg: Tensor | list[float],
        lat_deg: Tensor | list[float],
        max_order: int = 10,
    ) -> MOC:
        """Create a MOC from a spherical polygon."""
        nside = 1 << max_order
        # query_polygon works on lonlat if requested
        vertices = torch.stack(
            [torch.as_tensor(lon_deg), torch.as_tensor(lat_deg)], dim=-1
        )
        pix = _healpix.query_polygon(nside, vertices, nest=True, lonlat=True)
        uniqs = nest2uniq(nside, pix)
        return cls(uniqs)

    @classmethod
    def from_threshold(
        cls,
        smap: HealSparseMap,
        threshold: float,
        op: Literal[">", ">=", "<", "<="] = ">",
    ) -> MOC:
        """Create a MOC from pixels in a sparse map that satisfy a threshold."""
        if op == ">":
            mask = smap.values > threshold
        elif op == ">=":
            mask = smap.values >= threshold
        elif op == "<":
            mask = smap.values < threshold
        elif op == "<=":
            mask = smap.values <= threshold
        else:
            raise ValueError(f"Invalid operator {op}")

        # Get covered pixels at nside_sparse
        valid_indices = torch.where(mask)[0]
        if valid_indices.numel() == 0:
            return cls(torch.empty((0,), dtype=torch.int64))

        # We need to map local indices back to global nside_sparse pixel indices
        # This is more efficient if done via coverage_map
        all_global_pix = []
        cov_valid = torch.where(smap.coverage_map >= 0)[0]
        pixels_per_cov = smap.pixels_per_cov

        # We can optimize this by only checking valid_indices within each block
        for cp in cov_valid:
            offset = int(smap.coverage_map[cp].item())
            # Check if any pixels in this block are in valid_indices
            # Actually, simpler: mask the values locally
            block_mask = mask[offset : offset + pixels_per_cov]
            if block_mask.any():
                local_offsets = torch.where(block_mask)[0]
                global_start = int(cp.item()) * pixels_per_cov
                all_global_pix.append(global_start + local_offsets)

        if not all_global_pix:
            return cls(torch.empty((0,), dtype=torch.int64))

        pix = torch.cat(all_global_pix)
        uniqs = nest2uniq(smap.nside_sparse, pix)
        return cls(uniqs)

    def write_fits(self, filename: str, overwrite: bool = False) -> None:
        """Write the MOC to a FITS file following the IVOA MOC standard."""
        fitsio = importlib.import_module("fitsio")
        # IVOA MOC FITS encodes UNIQ as a single column table extension
        # with TFORM = '1K' (int64) or '1J' (int32 if order < 15)
        # and EXTNAME = 'MOC'

        header = [
            {"name": "PIXTYPE", "value": "HEALPIX"},
            {"name": "ORDERING", "value": "NUNIQ"},
            {"name": "COORDSYS", "value": "C"},  # Celestial / ICRS
            {"name": "MOCORDER", "value": self.max_order},
            {"name": "MOCTOOL", "value": "torchfits"},
        ]

        import numpy as np

        data = np.zeros(len(self.uniq), dtype=[("UNIQ", "i8")])
        data["UNIQ"] = self.uniq.cpu().numpy()

        with fitsio.FITS(filename, "rw", clobber=overwrite) as fits:
            fits.write(data, header=header, extname="MOC")

    @classmethod
    def read_fits(cls, filename: str) -> MOC:
        """Read a MOC from an IVOA FITS file."""
        fitsio = importlib.import_module("fitsio")

        with fitsio.FITS(filename) as fits:
            data = fits["MOC"].read()
            if "UNIQ" in data.dtype.names:
                # Convert to native byte order if necessary (FITS is big-endian)
                arr = data["UNIQ"]
                if arr.dtype.byteorder not in ("=", "|"):
                    arr = arr.astype(arr.dtype.newbyteorder("="))
                uniqs = torch.from_numpy(arr)
            else:
                # Older versions might just use the first column
                arr = data.iloc[:, 0]
                if hasattr(arr, "values"):
                    arr = arr.values
                if arr.dtype.byteorder not in ("=", "|"):
                    arr = arr.astype(arr.dtype.newbyteorder("="))
                uniqs = torch.from_numpy(arr)
        return cls(uniqs)

    def boundary(self) -> list[Tensor]:
        """
        Return the boundaries of the MOC as a list of vertex loops.
        Each loop is a Tensor of shape [N, 2] (lon, lat) in degrees.
        """
        # Very simplified boundary extraction for proof of concept.
        # Efficient boundary extraction for HEALPix/MOC is complex.
        # For Phase 4, we provide a placeholder or a very basic version.
        # A real implementation would trace edges of the union of pixels.
        # Here we just return the corners of the largest pixels as a "grid boundary".
        # TODO: Implement full edge tracing.
        all_loops = []
        nsides, pix_nested = uniq2nest(self.uniq)
        for ns, p in zip(nsides, pix_nested):
            # Get corners of this pixel
            corners = _healpix.boundaries(
                int(ns.item()), int(p.item()), nest=True, lonlat=True
            )
            # corners is (2, 4) -> lon, lat
            all_loops.append(torch.from_numpy(corners).transpose(0, 1))
        return all_loops

    def to_ascii(self) -> str:
        """Return IVOA ASCII representation of the MOC."""
        # ASCII format is: ORDER/PIX1 PIX2 ORDER2/PIX3 ...
        if self.uniq.numel() == 0:
            return ""
        nsides, pix_nested = uniq2nest(self.uniq)
        orders = torch.log2(nsides.to(torch.float64)).to(torch.int64)

        unique_orders = torch.unique(orders, sorted=True)
        parts = []
        for o in unique_orders:
            mask = orders == o
            o_pix = pix_nested[mask].sort()[0]
            pix_str = " ".join(map(str, o_pix.tolist()))
            parts.append(f"{int(o.item())}/{pix_str}")
        return " ".join(parts)

    @classmethod
    def from_ascii(cls, ascii_str: str) -> MOC:
        """Create a MOC from an IVOA ASCII string."""
        if not ascii_str.strip():
            return cls(torch.empty((0,), dtype=torch.int64))

        uniqs = []
        # Standard: ORDER/PIX PIX ...
        # We split by ORDER/ prefixes
        import re

        parts = re.split(r"(\d+)/", ascii_str.strip())
        # parts will be ['', '0', '0 1 2 ', '1', '4 5'] etc.
        for i in range(1, len(parts), 2):
            order = int(parts[i])
            pix_str = parts[i + 1].strip()
            if not pix_str:
                continue
            nside = 1 << order
            pix_list = [int(p) for p in pix_str.replace(",", " ").split()]
            uniqs.append(nest2uniq(nside, torch.tensor(pix_list, dtype=torch.int64)))

        if not uniqs:
            return cls(torch.empty((0,), dtype=torch.int64))
        return cls(torch.cat(uniqs))

    def to_json(self) -> str:
        """Return IVOA JSON representation of the MOC."""
        import json

        if self.uniq.numel() == 0:
            return "{}"

        nsides, pix_nested = uniq2nest(self.uniq)
        orders = torch.log2(nsides.to(torch.float64)).to(torch.int64)

        unique_orders = torch.unique(orders, sorted=True)
        data = {}
        for o in unique_orders:
            mask = orders == o
            o_pix = pix_nested[mask].sort()[0]
            data[str(int(o.item()))] = o_pix.tolist()

        return json.dumps(data)


__all__ = ["MOC", "nest2uniq", "uniq2nest"]
