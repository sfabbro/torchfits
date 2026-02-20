"""Coverage-aware sparse HEALPix map container."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from ..wcs import healpix as _healpix


_DEFAULT_FILL = float(_healpix.UNSEEN)


@dataclass(frozen=True)
class SparseHealpixMap:
    """Sparse HEALPix map storing values only on covered pixels."""

    nside: int
    nest: bool
    pixels: Tensor
    values: Tensor
    fill_value: float = _DEFAULT_FILL

    def __post_init__(self) -> None:
        _healpix.nside2npix(int(self.nside))
        pix = torch.as_tensor(self.pixels, dtype=torch.int64).reshape(-1)
        vals = torch.as_tensor(self.values)
        if vals.ndim == 0:
            raise ValueError("values must have at least 1 dimension")
        if vals.shape[-1] != pix.numel():
            raise ValueError("values last dimension must match number of pixels")

        if pix.numel() == 0:
            object.__setattr__(self, "pixels", pix)
            object.__setattr__(self, "values", vals)
            return

        order = torch.argsort(pix)
        pix_s = pix.index_select(0, order)
        vals_s = vals.index_select(-1, order)

        uniq, counts = torch.unique_consecutive(pix_s, return_counts=True)
        if uniq.numel() != pix_s.numel():
            # Keep last value for repeated pixels.
            idx_last = torch.cumsum(counts, dim=0) - 1
            pix_s = pix_s.index_select(0, idx_last)
            vals_s = vals_s.index_select(-1, idx_last)

        npix = _healpix.nside2npix(int(self.nside))
        if torch.any((pix_s < 0) | (pix_s >= npix)):
            raise ValueError("pixel index out of range for nside")

        object.__setattr__(self, "pixels", pix_s)
        object.__setattr__(self, "values", vals_s)

    @property
    def npix_total(self) -> int:
        return _healpix.nside2npix(int(self.nside))

    @property
    def coverage_fraction(self) -> float:
        return float(self.pixels.numel()) / float(self.npix_total)

    @property
    def covered(self) -> Tensor:
        return self.pixels

    @property
    def coverage_mask(self) -> Tensor:
        mask = torch.zeros((self.npix_total,), dtype=torch.bool, device=self.pixels.device)
        if self.pixels.numel() > 0:
            mask[self.pixels] = True
        return mask

    def to_dense(self) -> Tensor:
        shape = tuple(self.values.shape[:-1]) + (self.npix_total,)
        dense = torch.full(shape, float(self.fill_value), dtype=self.values.dtype, device=self.values.device)
        if self.pixels.numel() > 0:
            dense.index_copy_(-1, self.pixels, self.values)
        return dense

    @classmethod
    def from_dense(
        cls,
        map_values: Tensor | list[float] | list[list[float]],
        *,
        nside: int | None = None,
        nest: bool = False,
        valid_mask: Tensor | None = None,
        coverage_mode: Literal["all", "any"] = "all",
        fill_value: float = _DEFAULT_FILL,
    ) -> "SparseHealpixMap":
        vals = torch.as_tensor(map_values)
        if vals.ndim < 1:
            raise ValueError("map_values must have at least 1 dimension")
        npix = int(vals.shape[-1])
        ns = _healpix.npix2nside(npix) if nside is None else int(nside)
        if _healpix.nside2npix(ns) != npix:
            raise ValueError("map_values length does not match nside")

        if valid_mask is None:
            if vals.is_floating_point() or vals.is_complex():
                finite = torch.isfinite(vals)
                base = vals.real if vals.is_complex() else vals
                bad = torch.isclose(
                    base,
                    torch.as_tensor(fill_value, dtype=base.dtype, device=vals.device),
                )
                valid_all = finite & (~bad)
            else:
                valid_all = torch.ones_like(vals, dtype=torch.bool)
            if vals.ndim == 1:
                valid = valid_all
            else:
                reduce_dims = tuple(range(vals.ndim - 1))
                if coverage_mode == "all":
                    valid = torch.all(valid_all, dim=reduce_dims)
                elif coverage_mode == "any":
                    valid = torch.any(valid_all, dim=reduce_dims)
                else:
                    raise ValueError("coverage_mode must be one of {'all', 'any'}")
        else:
            valid = torch.as_tensor(valid_mask, dtype=torch.bool, device=vals.device)
            if valid.shape != (npix,):
                raise ValueError("valid_mask must have shape (npix,)")

        pixels = torch.nonzero(valid, as_tuple=False).reshape(-1).to(torch.int64)
        sparse_vals = vals.index_select(-1, pixels) if pixels.numel() > 0 else vals[..., :0]
        return cls(nside=ns, nest=nest, pixels=pixels, values=sparse_vals, fill_value=float(fill_value))

    def _global_to_local(self, pix: Tensor) -> tuple[Tensor, Tensor]:
        pix_t = torch.as_tensor(pix, dtype=torch.int64, device=self.pixels.device)
        idx = torch.searchsorted(self.pixels, pix_t)
        in_bounds = idx < self.pixels.numel()
        ok = torch.zeros_like(in_bounds, dtype=torch.bool)
        if bool(in_bounds.any()):
            ok[in_bounds] = self.pixels.index_select(0, idx[in_bounds]) == pix_t[in_bounds]
        return idx, ok

    def interpolate(
        self,
        lon_deg: Tensor | float,
        lat_deg: Tensor | float,
        *,
        method: Literal["nearest", "bilinear"] = "bilinear",
    ) -> Tensor:
        lon_t = torch.as_tensor(lon_deg, dtype=torch.float64)
        lat_t = torch.as_tensor(lat_deg, dtype=torch.float64)
        lon_t, lat_t = torch.broadcast_tensors(lon_t, lat_t)
        shape = lon_t.shape
        lon_f = lon_t.reshape(-1)
        lat_f = lat_t.reshape(-1)

        if method == "nearest":
            pix = _healpix.ang2pix(self.nside, lon_f, lat_f, nest=self.nest, lonlat=True)
            idx, ok = self._global_to_local(pix.to(self.pixels.device))
            out_shape = tuple(self.values.shape[:-1]) + (lon_f.numel(),)
            out = torch.full(out_shape, float(self.fill_value), dtype=self.values.dtype, device=self.values.device)
            if bool(ok.any()):
                sel = idx[ok]
                gathered = self.values.index_select(-1, sel)
                out[..., ok] = gathered
            return out.reshape(*self.values.shape[:-1], *shape)

        if method != "bilinear":
            raise ValueError("method must be one of {'nearest', 'bilinear'}")

        pix4, w4 = _healpix.get_interp_weights(self.nside, lon_f, lat_f, nest=self.nest, lonlat=True)
        pix4 = pix4.to(self.pixels.device).reshape(4, -1)
        w4 = w4.to(self.values.device, dtype=self.values.real.dtype if self.values.is_complex() else self.values.dtype).reshape(4, -1)

        out_shape = tuple(self.values.shape[:-1]) + (lon_f.numel(),)
        accum = torch.zeros(out_shape, dtype=self.values.dtype, device=self.values.device)
        wsum = torch.zeros((lon_f.numel(),), dtype=w4.dtype, device=self.values.device)

        for k in range(4):
            idx, ok = self._global_to_local(pix4[k])
            if not bool(ok.any()):
                continue
            sel = idx[ok]
            wk = w4[k, ok]
            vals = self.values.index_select(-1, sel)
            wshape = [1] * (vals.ndim - 1) + [wk.numel()]
            accum[..., ok] = accum[..., ok] + vals * wk.reshape(wshape)
            wsum[ok] = wsum[ok] + wk

        out = torch.full(out_shape, float(self.fill_value), dtype=self.values.dtype, device=self.values.device)
        nz = wsum > 0
        if bool(nz.any()):
            wshape = [1] * (accum.ndim - 1) + [int(nz.sum().item())]
            out[..., nz] = accum[..., nz] / wsum[nz].reshape(wshape)
        return out.reshape(*self.values.shape[:-1], *shape)

    def ud_grade(
        self,
        nside_out: int,
        *,
        pess: bool = False,
        power: float | None = None,
        fill_value: float | None = None,
    ) -> "SparseHealpixMap":
        nside_out = int(nside_out)
        _healpix.nside2npix(nside_out)
        fv = self.fill_value if fill_value is None else float(fill_value)

        if nside_out == int(self.nside):
            if fill_value is None:
                return self
            return SparseHealpixMap(
                nside=self.nside,
                nest=self.nest,
                pixels=self.pixels,
                values=self.values,
                fill_value=fv,
            )

        nside_in = int(self.nside)
        up = nside_out > nside_in
        ratio_num = nside_out if up else nside_in
        ratio_den = nside_in if up else nside_out
        if ratio_num % ratio_den != 0:
            dense = self.to_dense()
            order = "NEST" if self.nest else "RING"
            dense_out = _healpix.ud_grade(
                dense,
                nside_out,
                pess=pess,
                badval=fv,
                order_in=order,
                order_out=order,
                power=power,
            )
            valid = torch.any(~_healpix.mask_bad(dense_out, badval=fv) & torch.isfinite(dense_out), dim=tuple(range(dense_out.ndim - 1))) if dense_out.ndim > 1 else (~_healpix.mask_bad(dense_out, badval=fv) & torch.isfinite(dense_out))
            return SparseHealpixMap.from_dense(
                dense_out,
                nside=nside_out,
                nest=self.nest,
                valid_mask=valid,
                fill_value=fv,
            )

        ratio = ratio_num // ratio_den
        if ratio <= 0 or (ratio & (ratio - 1)) != 0:
            dense = self.to_dense()
            order = "NEST" if self.nest else "RING"
            dense_out = _healpix.ud_grade(
                dense,
                nside_out,
                pess=pess,
                badval=fv,
                order_in=order,
                order_out=order,
                power=power,
            )
            valid = torch.any(~_healpix.mask_bad(dense_out, badval=fv) & torch.isfinite(dense_out), dim=tuple(range(dense_out.ndim - 1))) if dense_out.ndim > 1 else (~_healpix.mask_bad(dense_out, badval=fv) & torch.isfinite(dense_out))
            return SparseHealpixMap.from_dense(
                dense_out,
                nside=nside_out,
                nest=self.nest,
                valid_mask=valid,
                fill_value=fv,
            )

        # Sparse fast path for power-of-two NSIDE ratios in NEST hierarchy.
        if self.pixels.numel() == 0:
            return SparseHealpixMap(
                nside=nside_out,
                nest=self.nest,
                pixels=self.pixels.new_empty((0,)),
                values=self.values[..., :0],
                fill_value=fv,
            )

        pix_n = self.pixels if self.nest else _healpix.ring2nest(nside_in, self.pixels)
        val = self.values
        scale = (float(nside_out) / float(nside_in)) ** float(power) if power is not None else 1.0

        if up:
            child_mult = ratio * ratio
            offs = torch.arange(child_mult, dtype=torch.int64, device=pix_n.device)
            out_pix_n = (pix_n.unsqueeze(1) * child_mult + offs.unsqueeze(0)).reshape(-1)
            out_val = val.repeat_interleave(child_mult, dim=-1)
            if power is not None:
                out_val = out_val * scale
            out_pix = out_pix_n if self.nest else _healpix.nest2ring(nside_out, out_pix_n)
            return SparseHealpixMap(
                nside=nside_out,
                nest=self.nest,
                pixels=out_pix,
                values=out_val,
                fill_value=fv,
            )

        parent_mult = ratio * ratio
        order = torch.argsort(pix_n)
        pix_sorted = pix_n.index_select(0, order)
        val_sorted = val.index_select(-1, order)
        parents = torch.div(pix_sorted, parent_mult, rounding_mode="floor")
        uniq, counts = torch.unique_consecutive(parents, return_counts=True)
        group_ids = torch.repeat_interleave(
            torch.arange(uniq.numel(), dtype=torch.int64, device=parents.device),
            counts,
        )
        sum_vals = torch.zeros(
            (*val_sorted.shape[:-1], uniq.numel()),
            dtype=val_sorted.dtype,
            device=val_sorted.device,
        )
        if val_sorted.is_floating_point() or val_sorted.is_complex():
            base = val_sorted.real if val_sorted.is_complex() else val_sorted
            goods = torch.isfinite(val_sorted) & (~torch.isclose(base, torch.as_tensor(fv, dtype=base.dtype, device=base.device)))
        else:
            goods = torch.ones_like(val_sorted, dtype=torch.bool)

        sum_vals.index_add_(-1, group_ids, val_sorted * goods.to(dtype=val_sorted.dtype))
        nhit = torch.zeros((*val_sorted.shape[:-1], uniq.numel()), dtype=torch.float64, device=val_sorted.device)
        nhit.index_add_(-1, group_ids, goods.to(dtype=torch.float64))

        nhit_f = nhit
        if power is not None:
            nhit_f = nhit_f / scale
        out_val = torch.zeros_like(sum_vals)
        nz = nhit_f != 0
        out_val[nz] = sum_vals[nz] / nhit_f[nz].to(dtype=sum_vals.dtype)
        badout = (nhit != float(parent_mult)) if pess else (nhit == 0.0)
        out_val = out_val.clone()
        out_val[badout] = float(fv)

        if out_val.ndim == 1:
            keep = ~badout
        else:
            keep = torch.any(~badout, dim=tuple(range(out_val.ndim - 1)))
        out_pix_n = uniq[keep]
        out_val = out_val[..., keep]
        out_pix = out_pix_n if self.nest else _healpix.nest2ring(nside_out, out_pix_n)
        return SparseHealpixMap(
            nside=nside_out,
            nest=self.nest,
            pixels=out_pix,
            values=out_val,
            fill_value=fv,
        )


__all__ = ["SparseHealpixMap"]
