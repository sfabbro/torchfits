from dataclasses import dataclass
from typing import Literal, Any

import torch
from torch import Tensor

from ..wcs import healpix as _healpix


_DEFAULT_FILL = float(_healpix.UNSEEN)
WIDE_NBIT = 64
WIDE_MASK = (1 << WIDE_NBIT) - 1


@dataclass(frozen=True)
class HealSparseMap:
    """
    Sparse HEALPix map using a two-level indexing scheme.

    Aligned with `healsparse` architecture:
    - coverage_map: Map at `nside_coverage` storing offsets into `values`.
      A value of -1 indicates the coverage pixel is not containing any data.
    - values: Flattened array of data values at `nside_sparse`.
    """

    nside_coverage: int
    nside_sparse: int
    coverage_map: Tensor  # [npix_coverage] int64 offsets
    values: Tensor  # [n_sparse_pixels, ...]
    nest: bool = True
    sentinel: Any = _DEFAULT_FILL
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.nside_sparse < self.nside_coverage:
            raise ValueError("nside_sparse must be >= nside_coverage")
        npix_cov = _healpix.nside2npix(self.nside_coverage)
        if self.coverage_map.shape[0] != npix_cov:
            raise ValueError(f"coverage_map must be size {npix_cov}")

    @property
    def pixels_per_cov(self) -> int:
        return (self.nside_sparse // self.nside_coverage) ** 2

    def _get_sparse_indices(self, pixels: Tensor) -> tuple[Tensor, Tensor]:
        """Convert global pixels at nside_sparse to (local_offset, values_index)."""
        pix = torch.as_tensor(pixels, dtype=torch.int64)
        cov_pix = torch.div(pix, self.pixels_per_cov, rounding_mode="floor")
        local_pix = torch.remainder(pix, self.pixels_per_cov)

        base_idx = self.coverage_map.index_select(0, cov_pix)
        valid = base_idx >= 0
        vals_idx = torch.where(
            valid, base_idx + local_pix, torch.full_like(base_idx, -1)
        )
        return vals_idx, valid

    def get_values(self, pixels: Tensor) -> Tensor:
        idx, valid = self._get_sparse_indices(pixels)
        out_shape = list(pixels.shape) + list(self.values.shape[1:])
        out = torch.full(
            out_shape, self.sentinel, dtype=self.values.dtype, device=self.values.device
        )
        if valid.any():
            out[valid] = self.values.index_select(0, idx[valid])
        return out

    def get_values_at(self, lon_deg: Tensor | float, lat_deg: Tensor | float) -> Tensor:
        """Get map values at given (lon, lat) coordinated in degrees."""
        lon = torch.as_tensor(lon_deg, dtype=torch.float64, device=self.values.device)
        lat = torch.as_tensor(lat_deg, dtype=torch.float64, device=self.values.device)
        pix = _healpix.ang2pix(self.nside_sparse, lon, lat, nest=self.nest, lonlat=True)
        return self.get_values(pix)

    def set_values(self, pixels: Tensor, values: Tensor | Any) -> None:
        """Set map values at given pixels (must be within covered regions)."""
        idx, valid = self._get_sparse_indices(pixels)
        if not valid.all():
            raise ValueError(
                "Some pixels are not within currently covered regions. Use from_pixels to expand coverage."
            )

        vals = torch.as_tensor(
            values, dtype=self.values.dtype, device=self.values.device
        )
        # Broadcast if needed
        if vals.dim() < self.values.dim():
            vals = vals.expand(idx.shape[0], *self.values.shape[1:])

        self.values[idx] = vals

    def set_values_at(
        self, lon_deg: Tensor | float, lat_deg: Tensor | float, values: Tensor | Any
    ) -> None:
        """Set map values at given coordinates (must be within covered regions)."""
        lon = torch.as_tensor(lon_deg, dtype=torch.float64, device=self.values.device)
        lat = torch.as_tensor(lat_deg, dtype=torch.float64, device=self.values.device)
        pix = _healpix.ang2pix(self.nside_sparse, lon, lat, nest=self.nest, lonlat=True)
        self.set_values(pix, values)

    def to_dense(self) -> Tensor:
        npix = _healpix.nside2npix(self.nside_sparse)
        dense = torch.full(
            (npix, *self.values.shape[1:]),
            self.sentinel,
            dtype=self.values.dtype,
            device=self.values.device,
        )

        cov_valid = torch.where(self.coverage_map >= 0)[0]
        if cov_valid.numel() > 0:
            for cpix in cov_valid:
                offset = int(self.coverage_map[cpix].item())
                start_pix = int(cpix.item()) * self.pixels_per_cov
                end_pix = start_pix + self.pixels_per_cov
                dense[start_pix:end_pix] = self.values[
                    offset : offset + self.pixels_per_cov
                ]
        return dense

    @classmethod
    def from_pixels(
        cls,
        pixels: Tensor,
        nside_sparse: int,
        nside_coverage: int,
        sentinel: Any = _DEFAULT_FILL,
        values: Tensor | None = None,
    ) -> "HealSparseMap":
        """Create a HealSparseMap from a list of pixels."""
        pix = torch.as_tensor(pixels, dtype=torch.int64)
        npix_cov = _healpix.nside2npix(nside_coverage)
        pixels_per_cov = (nside_sparse // nside_coverage) ** 2

        # Group pixels by coverage pixel
        cpix = pix // pixels_per_cov
        unique_cpix = torch.unique(cpix)

        cov_map = torch.full((npix_cov,), -1, dtype=torch.int64)

        # Sort pix and cpix to align with coverage map layout
        sort_idx = torch.argsort(cpix)
        pix = pix[sort_idx]
        cpix = cpix[sort_idx]

        num_covered = unique_cpix.numel()
        full_values = torch.full(
            (num_covered * pixels_per_cov,),
            sentinel,
            dtype=values.dtype if values is not None else torch.float32,
        )

        # Mapping from cpix to offset
        for i, cp in enumerate(unique_cpix):
            cov_map[cp] = i * pixels_per_cov

        # Fill in specific pixel values
        # offset = (cpix mapping to i) * pixels_per_cov + (pix % pixels_per_cov)
        # To do this vectorized in torch:
        # We need the relative index of each unique_cpix in cov_map
        # Searchsorted can give us the offset!
        cpix_offsets = torch.searchsorted(unique_cpix, cpix) * pixels_per_cov
        rel_idx = pix % pixels_per_cov
        full_idx = cpix_offsets + rel_idx

        if values is not None:
            full_values[full_idx] = torch.as_tensor(values, dtype=full_values.dtype)[
                sort_idx
            ]
        else:
            full_values[full_idx] = 1.0

        return cls(
            nside_coverage=nside_coverage,
            nside_sparse=nside_sparse,
            coverage_map=cov_map,
            values=full_values,
            nest=True,
            sentinel=sentinel,
        )

    def get_covered_pixels(self) -> Tensor:
        """Get all pixel indices that are covered (not missing)."""
        valid_mask = self.coverage_map != -1
        cpix = torch.where(valid_mask)[0]
        pixels_per_cov = (self.nside_sparse // self.nside_coverage) ** 2

        # This is a large expansion if many pixels are covered.
        # But required for some operations.
        all_pix = []
        for cp in cpix:
            all_pix.append(torch.arange(cp * pixels_per_cov, (cp + 1) * pixels_per_cov))

        if not all_pix:
            return torch.empty((0,), dtype=torch.int64)
        return torch.cat(all_pix)

    def ud_grade(self, nside_out: int, reduction: str = "mean") -> "HealSparseMap":
        """Change the resolution of the map."""
        if nside_out == self.nside_sparse:
            return self

        if nside_out > self.nside_sparse:
            # Upgrade (Upsample)
            ratio = (nside_out // self.nside_sparse) ** 2
            # Values are just repeated
            new_values = self.values.repeat_interleave(ratio)

            # Coverage map remains logically similar but nfine_per_cov changes.
            # We need to scaling the offsets in the coverage map?
            # Actually, our indexing is offset into values.
            # If we repeat_interleave, the new offset is old_offset * ratio.
            new_cov = torch.where(
                self.coverage_map == -1, -1, self.coverage_map * ratio
            )

            return HealSparseMap(
                nside_coverage=self.nside_coverage,
                nside_sparse=nside_out,
                coverage_map=new_cov,
                values=new_values,
                nest=self.nest,
                sentinel=self.sentinel,
            )
        else:
            # Downgrade (Downsample)
            ratio = (self.nside_sparse // nside_out) ** 2

            # We need to reduce blocks of size 'ratio'
            # If a block contains sentinels, we handle it based on reduction.
            # For simplicity, we can reshape and reduce.
            v = self.values.reshape(-1, ratio)

            # Special handling for sentinels: treat as NaN?
            if reduction == "mean":
                # Only average non-sentinel values?
                # Standards usually say if all are sentinel, result is sentinel.
                # If some are valid, average valid ones.
                mask = (v != self.sentinel).to(torch.float32)
                v_clean = torch.where(
                    v != self.sentinel, v, torch.tensor(0.0, dtype=v.dtype)
                )
                sums = torch.sum(v_clean, dim=1)
                counts = torch.sum(mask, dim=1)
                new_v = torch.where(
                    counts > 0,
                    sums / counts,
                    torch.tensor(self.sentinel, dtype=v.dtype),
                )
            elif reduction == "sum":
                new_v = torch.sum(v, dim=1)  # This might include sentinels...
                # Better: sum only valid or assume sentinel is 0.
            else:
                # and/or for masks
                if reduction == "or":
                    new_v = torch.any(v > 0, dim=1).to(v.dtype)
                elif reduction == "and":
                    new_v = torch.all(v > 0, dim=1).to(v.dtype)
                else:
                    raise NotImplementedError(f"reduction {reduction} not implemented")

            # Update coverage map.
            # Pixels per coverage pixel also changes.
            # We might want to keep the same nside_coverage if possible.
            (nside_out // self.nside_coverage) ** 2
            # Each old coverage pixel (size self_pix_per_cov) now contains self_pix_per_cov / ratio
            # new pixels at nside_out.
            # The offsets in new_cov should point into new_v.
            new_cov = torch.where(
                self.coverage_map == -1, -1, self.coverage_map // ratio
            )

            return HealSparseMap(
                nside_coverage=self.nside_coverage,
                nside_sparse=nside_out,
                coverage_map=new_cov,
                values=new_v,
                nest=self.nest,
                sentinel=self.sentinel,
            )

    def __add__(self, other: Any) -> "HealSparseMap":
        return self._apply_op(other, torch.add)

    def __sub__(self, other: Any) -> "HealSparseMap":
        return self._apply_op(other, torch.sub)

    def __mul__(self, other: Any) -> "HealSparseMap":
        return self._apply_op(other, torch.mul)

    def __truediv__(self, other: Any) -> "HealSparseMap":
        return self._apply_op(other, torch.true_divide)

    def __and__(self, other: Any) -> "HealSparseMap":
        return self._apply_logical_op(other, torch.bitwise_and)

    def __or__(self, other: Any) -> "HealSparseMap":
        return self._apply_logical_op(other, torch.bitwise_or)

    def __xor__(self, other: Any) -> "HealSparseMap":
        return self._apply_logical_op(other, torch.bitwise_xor)

    def __invert__(self) -> "HealSparseMap":
        if self.values.is_floating_point():
            # For float "masks", invert >0 status
            new_vals = (self.values == 0).to(self.values.dtype)
        else:
            new_vals = ~self.values
        return HealSparseMap(
            self.nside_coverage,
            self.nside_sparse,
            self.coverage_map,
            new_vals,
            nest=self.nest,
            sentinel=self.sentinel,
        )

    def _apply_op(self, other: Any, op_func: Any) -> "HealSparseMap":
        if not isinstance(other, HealSparseMap):
            # Scalar operation
            new_vals = op_func(self.values, other)
            # Sentinel also changes!
            new_sentinel = op_func(torch.tensor(self.sentinel), other).item()
            return HealSparseMap(
                self.nside_coverage,
                self.nside_sparse,
                self.coverage_map,
                new_vals,
                nest=self.nest,
                sentinel=new_sentinel,
            )

        # Map-Map operation
        if self.nside_sparse != other.nside_sparse or self.nest != other.nest:
            raise ValueError("Maps must have same resolution and ordering")

        # Union of coverage
        comb_mask = (self.coverage_map >= 0) | (other.coverage_map >= 0)
        new_cpix = torch.where(comb_mask)[0]

        npix_cov = self.coverage_map.shape[0]
        new_cov = torch.full(
            (npix_cov,), -1, dtype=torch.int64, device=self.values.device
        )

        # For each combined coverage pixel, we get start:end for both
        # This is a bit slow in pure python loop if many cov pixels.
        # But consistent with existing to_dense.
        new_vals_list = []
        for i, cp in enumerate(new_cpix):
            new_cov[cp] = i * self.pixels_per_cov
            v1 = self.get_values_block(int(cp.item()))
            v2 = other.get_values_block(int(cp.item()))
            new_vals_list.append(op_func(v1, v2))

        new_sentinel = op_func(
            torch.tensor(self.sentinel), torch.tensor(other.sentinel)
        ).item()
        return HealSparseMap(
            self.nside_coverage,
            self.nside_sparse,
            new_cov,
            torch.cat(new_vals_list),
            nest=self.nest,
            sentinel=new_sentinel,
        )

    def _apply_logical_op(self, other: Any, op_func: Any) -> "HealSparseMap":
        # Logical ops (and/or) usually want to preserve the "null" as sentinel.
        # If both are sentinels, result is sentinel.
        if isinstance(other, HealSparseMap) and self.values.is_floating_point():
            # For floating point masks, use >0 logic
            def float_op(a, b):
                a_bool = a > 0
                b_bool = b > 0
                return op_func(a_bool, b_bool).to(a.dtype)

            return self._apply_op(other, float_op)

        return self._apply_op(other, op_func)

    def get_values_block(self, cov_pix: int) -> Tensor:
        """Get values for a single coverage pixel."""
        offset = self.coverage_map[cov_pix]
        if offset < 0:
            return torch.full(
                (self.pixels_per_cov, *self.values.shape[1:]),
                self.sentinel,
                dtype=self.values.dtype,
                device=self.values.device,
            )
        return self.values[offset : offset + self.pixels_per_cov]

    @classmethod
    def convert_healpix_map(
        cls,
        healpix_map: Tensor,
        nside_coverage: int,
        nest: bool = True,
        sentinel: Any = 0,
    ) -> "HealSparseMap":
        vals = torch.as_tensor(healpix_map)
        npix = vals.shape[0]
        nside_sparse = _healpix.npix2nside(npix)

        npix_cov = _healpix.nside2npix(nside_coverage)
        pixels_per_cov = (nside_sparse // nside_coverage) ** 2

        cov_map = torch.full((npix_cov,), -1, dtype=torch.int64)
        sparse_values = []

        current_offset = 0
        for cpix in range(npix_cov):
            start = cpix * pixels_per_cov
            end = start + pixels_per_cov
            block = vals[start:end]
            if not torch.all(block == sentinel):
                cov_map[cpix] = current_offset
                sparse_values.append(block)
                current_offset += pixels_per_cov

        if not sparse_values:
            # We need to handle dtype appropriately. If vals is empty, we use a default or float32.
            return cls.make_empty(nside_coverage, nside_sparse, vals.dtype, sentinel)

        return cls(
            nside_coverage=nside_coverage,
            nside_sparse=nside_sparse,
            coverage_map=cov_map,
            values=torch.cat(sparse_values),
            nest=nest,
            sentinel=sentinel,
        )

    def write_fits(self, filename: str, overwrite: bool = False) -> None:
        """Write the map to a FITS file in healsparse format."""
        import fitsio

        # Header for coverage
        c_hdr = [
            {"name": "PIXTYPE", "value": "HEALSPARSE"},
            {"name": "NSIDE", "value": self.nside_coverage},
            {"name": "ORDERING", "value": "NEST"},
        ]

        # Header for sparse data
        s_hdr = [
            {"name": "PIXTYPE", "value": "HEALSPARSE"},
            {"name": "NSIDE", "value": self.nside_sparse},
            {"name": "SENTINEL", "value": self.sentinel},
        ]

        with fitsio.FITS(filename, "rw", clobber=overwrite) as fits:
            # Extension 1: Coverage map
            fits.write(self.coverage_map.numpy(), header=c_hdr, extname="COVERAGE")

            # Extension 2: Sparse values
            fits.write(self.values.numpy(), header=s_hdr, extname="SPARSE")

    @classmethod
    def read_fits(cls, filename: str) -> "HealSparseMap":
        """Read a HealSparseMap from a FITS file."""
        import fitsio

        with fitsio.FITS(filename) as fits:
            # Read coverage
            cov_map = torch.from_numpy(fits["COVERAGE"].read())
            c_hdr = fits["COVERAGE"].read_header()
            nside_coverage = c_hdr["NSIDE"]

            # Read sparse values
            values = torch.from_numpy(fits["SPARSE"].read())
            s_hdr = fits["SPARSE"].read_header()
            nside_sparse = s_hdr["NSIDE"]
            sentinel = s_hdr.get("SENTINEL", _DEFAULT_FILL)

        return cls(
            nside_coverage=nside_coverage,
            nside_sparse=nside_sparse,
            coverage_map=cov_map,
            values=values,
            nest=True,
            sentinel=sentinel,
        )

        return cls(
            nside_coverage=nside_coverage,
            nside_sparse=nside_sparse,
            coverage_map=cov_map,
            values=values,
            nest=True,
            sentinel=sentinel,
        )

    @classmethod
    def from_moc(cls, moc: "MOC", nside: int, sentinel: Any = 0) -> "HealSparseMap":
        """Create a HealSparseMap from a MOC at a fixed resolution."""
        # This is a high-level wrapper around the range-based logic in MOC.to_sparse_map
        # (which was refactored into SparseMap logic in the previous step)
        return moc.to_sparse_map(nside, sentinel=sentinel)

    @classmethod
    def make_empty(
        cls,
        nside_coverage: int,
        nside_sparse: int,
        dtype: torch.dtype,
        sentinel: Any = _DEFAULT_FILL,
    ) -> "HealSparseMap":
        npix_cov = _healpix.nside2npix(nside_coverage)
        cov_map = torch.full((npix_cov,), -1, dtype=torch.int64)
        return cls(
            nside_coverage=nside_coverage,
            nside_sparse=nside_sparse,
            coverage_map=cov_map,
            values=torch.empty((0,), dtype=dtype),
            sentinel=sentinel,
        )


SparseMap = HealSparseMap
SparseHealpixMap = HealSparseMap


class WideBitMask:
    """Support for masks wider than 64 bits."""

    def __init__(self, n_bits: int, device: torch.device | None = None):
        self.n_bits = n_bits
        self.n_words = (n_bits + WIDE_NBIT - 1) // WIDE_NBIT
        self.device = device

    def create_null(self, n_pixels: int) -> Tensor:
        return torch.zeros(
            (n_pixels, self.n_words), dtype=torch.int64, device=self.device
        )


class SkyMaskPipe:
    """Pipeline for assembling complex sky masks from stages."""

    def __init__(self, nside: int, nest: bool = True):
        self.nside = nside
        self.nest = nest
        self.stages: dict[str, HealSparseMap] = {}

    def add_stage(self, name: str, mask: HealSparseMap):
        if mask.nside_sparse != self.nside or mask.nest != self.nest:
            raise ValueError("mask is incompatible with pipeline settings")
        self.stages[name] = mask

    def combine(self, operation: Literal["and", "or"] = "and") -> HealSparseMap:
        if not self.stages:
            raise ValueError("no stages to combine")

        stage_names = list(self.stages.keys())
        result = self.stages[stage_names[0]]

        for name in stage_names[1:]:
            other = self.stages[name]
            if operation == "and":
                result = result & other
            else:
                result = result | other
        return result

    @staticmethod
    def _dummy():
        pass


__all__ = ["HealSparseMap", "SparseMap", "SkyMaskPipe", "WideBitMask"]
