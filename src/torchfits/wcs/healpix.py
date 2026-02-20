import math
from typing import Sequence, Tuple

import torch
from torch import Tensor

try:
    import torchfits.cpp as _cpp
except Exception:  # pragma: no cover - optional fast-path
    _cpp = None


_JRLL = torch.tensor([2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4], dtype=torch.int64)
_JPLL = torch.tensor([1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7], dtype=torch.int64)
_FACE_CONST_CACHE: dict[tuple[str, int], tuple[Tensor, Tensor]] = {}
_CAP_START_CACHE: dict[tuple[str, int, int], Tensor] = {}
_MAX_PIXEL_RADIUS_CACHE: dict[int, float] = {}
_NB_XOFFSET = torch.tensor([-1, -1, 0, 1, 1, 1, 0, -1], dtype=torch.int64)
_NB_YOFFSET = torch.tensor([0, 1, 1, 1, 0, -1, -1, -1], dtype=torch.int64)
_NB_FACEARRAY = torch.tensor(
    [
        [8, 9, 10, 11, -1, -1, -1, -1, 10, 11, 8, 9],
        [5, 6, 7, 4, 8, 9, 10, 11, 9, 10, 11, 8],
        [-1, -1, -1, -1, 5, 6, 7, 4, -1, -1, -1, -1],
        [4, 5, 6, 7, 11, 8, 9, 10, 11, 8, 9, 10],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [1, 2, 3, 0, 0, 1, 2, 3, 5, 6, 7, 4],
        [-1, -1, -1, -1, 7, 4, 5, 6, -1, -1, -1, -1],
        [3, 0, 1, 2, 3, 0, 1, 2, 4, 5, 6, 7],
        [2, 3, 0, 1, -1, -1, -1, -1, 0, 1, 2, 3],
    ],
    dtype=torch.int64,
)
_NB_SWAPARRAY = torch.tensor(
    [
        [0, 0, 3],
        [0, 0, 6],
        [0, 0, 0],
        [0, 0, 5],
        [0, 0, 0],
        [5, 0, 0],
        [0, 0, 0],
        [6, 0, 0],
        [3, 0, 0],
    ],
    dtype=torch.int64,
)
UNSEEN = -1.6375e30


def _float_dtype_for_device(device: torch.device) -> torch.dtype:
    if device.type == "mps":
        return torch.float32
    return torch.float64


def _validate_nside(nside: int) -> None:
    if nside <= 0 or (nside & (nside - 1)) != 0:
        raise ValueError("nside must be a positive power of two")


def _as_float64(x: Tensor | float | list[float]) -> Tensor:
    t = torch.as_tensor(x)
    return t.to(dtype=_float_dtype_for_device(t.device))


def _as_int64(x: Tensor | int | list[int]) -> Tensor:
    return torch.as_tensor(x, dtype=torch.int64)


def _face_consts(device: torch.device) -> Tuple[Tensor, Tensor]:
    idx = -1 if device.index is None else int(device.index)
    key = (device.type, idx)
    cached = _FACE_CONST_CACHE.get(key)
    if cached is not None:
        return cached
    jrll = _JRLL.to(device=device)
    jpll = _JPLL.to(device=device)
    _FACE_CONST_CACHE[key] = (jrll, jpll)
    return jrll, jpll


def _cap_ring_starts(nside: int, device: torch.device) -> Tensor:
    idx = -1 if device.index is None else int(device.index)
    key = (device.type, idx, nside)
    cached = _CAP_START_CACHE.get(key)
    if cached is not None:
        return cached
    ir = torch.arange(1, nside + 1, dtype=torch.int64, device=device)
    starts = 2 * ir * (ir - 1)
    _CAP_START_CACHE[key] = starts
    return starts


def _isqrt(v: Tensor) -> Tensor:
    v = v.to(torch.int64)
    # Use native integer sqrt when available, fallback to corrected float path.
    if hasattr(torch, "isqrt"):
        try:
            return torch.isqrt(v)
        except RuntimeError:
            pass
    f_dtype = _float_dtype_for_device(v.device)
    r = torch.floor(torch.sqrt(v.to(f_dtype))).to(torch.int64)
    r = torch.where((r + 1) * (r + 1) <= v, r + 1, r)
    r = torch.where(r * r > v, r - 1, r)
    return r


def spread_bits(x: Tensor) -> Tensor:
    """Interleave bits for Morton code (x -> x0x1x2...)."""
    x = x.to(torch.int64)
    x = (x | (x << 16)) & 0x0000FFFF0000FFFF
    x = (x | (x << 8)) & 0x00FF00FF00FF00FF
    x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0F
    x = (x | (x << 2)) & 0x3333333333333333
    x = (x | (x << 1)) & 0x5555555555555555
    return x


def compact_bits(x: Tensor) -> Tensor:
    """Extract interleaved bits from Morton code."""
    x = x.to(torch.int64) & 0x5555555555555555
    x = (x | (x >> 1)) & 0x3333333333333333
    x = (x | (x >> 2)) & 0x0F0F0F0F0F0F0F0F
    x = (x | (x >> 4)) & 0x00FF00FF00FF00FF
    x = (x | (x >> 8)) & 0x0000FFFF0000FFFF
    x = (x | (x >> 16)) & 0x00000000FFFFFFFF
    return x


def _xyf2nest(nside: int, ix: Tensor, iy: Tensor, face_num: Tensor) -> Tensor:
    npface = nside * nside
    return face_num * npface + spread_bits(ix) + (spread_bits(iy) << 1)


def _nest2xyf(nside: int, pix: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    npface = nside * nside
    face_num = pix // npface
    ipf = pix % npface
    ix = compact_bits(ipf)
    iy = compact_bits(ipf >> 1)
    return ix, iy, face_num


def _xyf2ring(nside: int, ix: Tensor, iy: Tensor, face_num: Tensor) -> Tensor:
    nl4 = 4 * nside
    ncap = 2 * nside * (nside - 1)
    npix = 12 * nside * nside
    jrll, jpll = _face_consts(face_num.device)
    jr = jrll[face_num] * nside - ix - iy - 1

    north = jr < nside
    south = jr > (3 * nside)
    equat = ~(north | south)

    nr_south = nl4 - jr
    nr = torch.where(
        north, jr, torch.where(south, nr_south, torch.full_like(jr, nside))
    )
    n_before_north = 2 * nr * (nr - 1)
    n_before_south = npix - 2 * (nr + 1) * nr
    n_before_equat = ncap + (jr - nside) * nl4
    n_before = torch.where(
        north, n_before_north, torch.where(south, n_before_south, n_before_equat)
    )
    kshift = torch.where(equat, (jr - nside) & 1, torch.zeros_like(jr))

    jp = (jpll[face_num] * nr + ix - iy + 1 + kshift) // 2
    jp = torch.where(jp > nl4, jp - nl4, jp)
    jp = torch.where(jp < 1, jp + nl4, jp)

    return n_before + jp - 1


def _ring2xyf(nside: int, pix: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    ncap = 2 * nside * (nside - 1)
    npix = 12 * nside * nside
    nl2 = 2 * nside
    pix = pix.to(torch.int64)
    jrll, jpll = _face_consts(pix.device)
    starts = _cap_ring_starts(nside, pix.device)

    iring = torch.empty_like(pix)
    iphi = torch.empty_like(pix)
    kshift = torch.empty_like(pix)
    nr = torch.empty_like(pix)
    face_num = torch.empty_like(pix)

    north = pix < ncap
    equat = (pix >= ncap) & (pix < (npix - ncap))
    south = ~(north | equat)

    if north.any():
        p = pix[north]
        ir = torch.bucketize(p, starts, right=True)
        iph = (p + 1) - 2 * ir * (ir - 1)
        iring[north] = ir
        iphi[north] = iph
        kshift[north] = 0
        nr[north] = ir
        face_num[north] = (iph - 1) // ir

    if equat.any():
        p = pix[equat]
        ip = p - ncap
        ir = (ip // (4 * nside)) + nside
        iph = (ip % (4 * nside)) + 1
        ks = (ir + nside) & 1
        ire = ir - nside + 1
        irm = nl2 + 2 - ire
        ifm = (iph - (ire // 2) + nside - 1) // nside
        ifp = (iph - (irm // 2) + nside - 1) // nside
        f = torch.where(ifp == ifm, ifp | 4, torch.where(ifp < ifm, ifp, ifm + 8))

        iring[equat] = ir
        iphi[equat] = iph
        kshift[equat] = ks
        nr[equat] = nside
        face_num[equat] = f

    if south.any():
        p = pix[south]
        ip = npix - p
        irs = torch.bucketize(ip - 1, starts, right=True)
        iph = 4 * irs + 1 - (ip - 2 * irs * (irs - 1))
        ir = 2 * nl2 - irs
        f = 8 + (iph - 1) // irs

        iring[south] = ir
        iphi[south] = iph
        kshift[south] = 0
        nr[south] = irs
        face_num[south] = f

    irt = iring - jrll[face_num] * nside + 1
    ipt = 2 * iphi - jpll[face_num] * nr - kshift - 1
    ipt = torch.where(ipt >= nl2, ipt - 8 * nside, ipt)

    ix = torch.div(ipt - irt, 2, rounding_mode="floor")
    iy = torch.div(-(ipt + irt), 2, rounding_mode="floor")
    return ix, iy, face_num


def ang2pix_ring(nside: int, ra: Tensor, dec: Tensor) -> Tensor:
    """Convert RA/Dec (degrees) to HEALPix RING indices."""
    _validate_nside(nside)
    ra_t = _as_float64(ra)
    dec_t = _as_float64(dec)
    ra_t, dec_t = torch.broadcast_tensors(ra_t, dec_t)
    if ra_t.device.type == "mps":
        # MPS lacks float64; route through CPU for parity-grade angular indexing.
        return ang2pix_ring(nside, ra_t.cpu(), dec_t.cpu()).to(device=ra_t.device)
    if _cpp is not None and ra_t.device.type == "cpu" and dec_t.device.type == "cpu":
        if hasattr(_cpp, "healpix_ang2pix_ring_torch_cpu"):
            return _cpp.healpix_ang2pix_ring_torch_cpu(
                nside,
                ra_t.contiguous(),
                dec_t.contiguous(),
            )
    f_dtype = ra_t.dtype

    z = torch.sin(torch.deg2rad(dec_t))
    za = torch.abs(z)
    tt = torch.remainder(torch.deg2rad(ra_t), 2.0 * math.pi) * (2.0 / math.pi)

    pix = torch.empty_like(z, dtype=torch.int64)

    equat = za <= (2.0 / 3.0)
    if equat.any():
        ze = z[equat]
        tte = tt[equat]

        temp1 = nside * (0.5 + tte)
        temp2 = nside * ze * 0.75
        jp = (temp1 - temp2).to(torch.int64)
        jm = (temp1 + temp2).to(torch.int64)

        ir = nside + 1 + jp - jm
        kshift = 1 - (ir & 1)

        ip = ((jp + jm - nside + kshift + 1) // 2) + 1
        ip = torch.where(ip > (4 * nside), ip - (4 * nside), ip)
        ip = torch.where(ip < 1, ip + (4 * nside), ip)

        pix[equat] = 2 * nside * (nside - 1) + (ir - 1) * (4 * nside) + (ip - 1)

    pol = ~equat
    if pol.any():
        zp = z[pol]
        ttp = tt[pol]

        tp = ttp - ttp.to(torch.int64).to(f_dtype)
        tmp = nside * torch.sqrt(3.0 * (1.0 - torch.abs(zp)))

        jp = (tp * tmp).to(torch.int64)
        jm = ((1.0 - tp) * tmp).to(torch.int64)

        ir = jp + jm + 1
        ip = torch.remainder((ttp * ir.to(f_dtype)).to(torch.int64), 4 * ir)

        north = zp > 0
        ppix = torch.empty_like(ir)
        ppix[north] = 2 * ir[north] * (ir[north] - 1) + ip[north]
        ppix[~north] = (
            12 * nside * nside - 2 * ir[~north] * (ir[~north] + 1) + ip[~north]
        )
        pix[pol] = ppix

    return pix


def _pix2thetaphi_ring(nside: int, pix: Tensor) -> tuple[Tensor, Tensor]:
    """Convert HEALPix RING indices to (theta, phi) in radians."""
    _validate_nside(nside)
    pix_t = _as_int64(pix)
    if pix_t.device.type == "mps":
        theta_cpu, phi_cpu = _pix2thetaphi_ring(nside, pix_t.cpu())
        out_dtype = _float_dtype_for_device(pix_t.device)
        return theta_cpu.to(device=pix_t.device, dtype=out_dtype), phi_cpu.to(
            device=pix_t.device, dtype=out_dtype
        )
    f_dtype = _float_dtype_for_device(pix_t.device)

    ncap = 2 * nside * (nside - 1)
    npix = 12 * nside * nside
    if torch.any((pix_t < 0) | (pix_t >= npix)):
        raise ValueError("pixel index out of range for nside")

    fact2 = 4.0 / npix

    z = torch.empty_like(pix_t, dtype=f_dtype)
    theta = torch.empty_like(pix_t, dtype=f_dtype)
    phi = torch.empty_like(pix_t, dtype=f_dtype)

    north = pix_t < ncap
    equat = (pix_t >= ncap) & (pix_t < (npix - ncap))
    south = ~(north | equat)

    if north.any():
        p = pix_t[north]
        iring = (1 + _isqrt(1 + 2 * p)) >> 1
        iphi = (p + 1) - 2 * iring * (iring - 1)
        tmp = (iring.to(f_dtype) ** 2) * fact2
        z[north] = 1.0 - tmp
        sint = torch.sqrt(torch.clamp(tmp * (2.0 - tmp), min=0.0))
        theta[north] = torch.atan2(sint, z[north])
        phi[north] = (iphi.to(f_dtype) - 0.5) * ((math.pi / 2.0) / iring.to(f_dtype))

    if equat.any():
        p = pix_t[equat]
        fact1 = (2 * nside) * fact2
        ip = p - ncap
        iring = (ip // (4 * nside)) + nside
        iphi = (ip % (4 * nside)) + 1
        fodd = torch.where(((iring + nside) & 1).bool(), 1.0, 0.5)
        z[equat] = (2 * nside - iring).to(f_dtype) * fact1
        theta[equat] = torch.acos(torch.clamp(z[equat], -1.0, 1.0))
        phi[equat] = (iphi.to(f_dtype) - fodd) * (math.pi / (2 * nside))

    if south.any():
        p = pix_t[south]
        ip = npix - p
        iring = (1 + _isqrt(2 * ip - 1)) >> 1
        iphi = 4 * iring + 1 - (ip - 2 * iring * (iring - 1))
        tmp = (iring.to(f_dtype) ** 2) * fact2
        z[south] = -1.0 + tmp
        sint = torch.sqrt(torch.clamp(tmp * (2.0 - tmp), min=0.0))
        theta[south] = math.pi - torch.atan2(sint, 1.0 - tmp)
        phi[south] = (iphi.to(f_dtype) - 0.5) * ((math.pi / 2.0) / iring.to(f_dtype))

    phi = torch.remainder(phi, 2.0 * math.pi)
    return theta, phi


def pix2ang_ring(nside: int, pix: Tensor) -> Tuple[Tensor, Tensor]:
    """Convert HEALPix RING indices to RA/Dec (degrees)."""
    _validate_nside(nside)
    pix_t = _as_int64(pix)
    if pix_t.device.type == "mps":
        # MPS float32 trig accumulates boundary error; compute on CPU and move back.
        ra_cpu, dec_cpu = pix2ang_ring(nside, pix_t.cpu())
        out_dtype = _float_dtype_for_device(pix_t.device)
        return ra_cpu.to(device=pix_t.device, dtype=out_dtype), dec_cpu.to(
            device=pix_t.device, dtype=out_dtype
        )
    if _cpp is not None and pix_t.device.type == "cpu":
        if hasattr(_cpp, "healpix_pix2ang_ring_torch_cpu"):
            return _cpp.healpix_pix2ang_ring_torch_cpu(nside, pix_t.contiguous())

    theta, phi = _pix2thetaphi_ring(nside, pix_t)
    ra = torch.remainder(torch.rad2deg(phi), 360.0)
    dec = 90.0 - torch.rad2deg(theta)
    return ra, dec


def ang2pix_nested(nside: int, ra: Tensor, dec: Tensor) -> Tensor:
    """Convert RA/Dec (degrees) to HEALPix NESTED indices."""
    _validate_nside(nside)
    ra_t = _as_float64(ra)
    dec_t = _as_float64(dec)
    ra_t, dec_t = torch.broadcast_tensors(ra_t, dec_t)
    if ra_t.device.type == "mps":
        return ang2pix_nested(nside, ra_t.cpu(), dec_t.cpu()).to(device=ra_t.device)
    if _cpp is not None and ra_t.device.type == "cpu" and dec_t.device.type == "cpu":
        if hasattr(_cpp, "healpix_ang2pix_nested_torch_cpu"):
            return _cpp.healpix_ang2pix_nested_torch_cpu(
                nside,
                ra_t.contiguous(),
                dec_t.contiguous(),
            )
    f_dtype = ra_t.dtype

    z = torch.sin(torch.deg2rad(dec_t))
    za = torch.abs(z)
    tt = torch.remainder(torch.deg2rad(ra_t), 2.0 * math.pi) * (2.0 / math.pi)

    face_num = torch.empty_like(z, dtype=torch.int64)
    ix = torch.empty_like(z, dtype=torch.int64)
    iy = torch.empty_like(z, dtype=torch.int64)

    equat = za <= (2.0 / 3.0)
    if equat.any():
        ze = z[equat]
        tte = tt[equat]
        temp1 = nside * (0.5 + tte)
        temp2 = nside * (ze * 0.75)
        jp = (temp1 - temp2).to(torch.int64)
        jm = (temp1 + temp2).to(torch.int64)

        ifp = jp // nside
        ifm = jm // nside
        face = torch.where(ifp == ifm, ifp | 4, torch.where(ifp < ifm, ifp, ifm + 8))

        face_num[equat] = face
        ix[equat] = jm & (nside - 1)
        iy[equat] = nside - (jp & (nside - 1)) - 1

    pol = ~equat
    if pol.any():
        zp = z[pol]
        ttp = tt[pol]
        ntt = ttp.to(torch.int64)
        ntt = torch.where(ntt >= 4, torch.full_like(ntt, 3), ntt)
        tp = ttp - ntt.to(f_dtype)

        tmp = nside * torch.sqrt(3.0 * (1.0 - torch.abs(zp)))
        jp = (tp * tmp).to(torch.int64)
        jm = ((1.0 - tp) * tmp).to(torch.int64)
        jp = torch.minimum(jp, torch.full_like(jp, nside - 1))
        jm = torch.minimum(jm, torch.full_like(jm, nside - 1))

        north = zp >= 0
        f = torch.empty_like(jp)
        x = torch.empty_like(jp)
        y = torch.empty_like(jp)

        f[north] = ntt[north]
        x[north] = nside - jm[north] - 1
        y[north] = nside - jp[north] - 1

        f[~north] = ntt[~north] + 8
        x[~north] = jp[~north]
        y[~north] = jm[~north]

        face_num[pol] = f
        ix[pol] = x
        iy[pol] = y

    return _xyf2nest(nside, ix, iy, face_num)


def pix2ang_nested(nside: int, pix: Tensor) -> Tuple[Tensor, Tensor]:
    """Convert HEALPix NESTED indices to RA/Dec (degrees)."""
    _validate_nside(nside)
    pix_t = _as_int64(pix)
    if pix_t.device.type == "mps":
        ra_cpu, dec_cpu = pix2ang_nested(nside, pix_t.cpu())
        out_dtype = _float_dtype_for_device(pix_t.device)
        return ra_cpu.to(device=pix_t.device, dtype=out_dtype), dec_cpu.to(
            device=pix_t.device, dtype=out_dtype
        )
    if _cpp is not None and pix_t.device.type == "cpu":
        if hasattr(_cpp, "healpix_pix2ang_nested_torch_cpu"):
            return _cpp.healpix_pix2ang_nested_torch_cpu(nside, pix_t.contiguous())
    f_dtype = _float_dtype_for_device(pix_t.device)

    npix = 12 * nside * nside
    if torch.any((pix_t < 0) | (pix_t >= npix)):
        raise ValueError("pixel index out of range for nside")

    ix, iy, face_num = _nest2xyf(nside, pix_t)

    nl4 = 4 * nside
    fact2 = 4.0 / npix

    jr = _JRLL.to(face_num.device)[face_num] * nside - ix - iy - 1

    nr = torch.empty_like(jr)
    z = torch.empty_like(jr, dtype=f_dtype)
    kshift = torch.zeros_like(jr)

    north = jr < nside
    south = jr > (3 * nside)
    equat = ~(north | south)

    nr[north] = jr[north]
    z[north] = 1.0 - (nr[north].to(f_dtype) ** 2) * fact2

    nr[south] = nl4 - jr[south]
    z[south] = (nr[south].to(f_dtype) ** 2) * fact2 - 1.0

    if equat.any():
        nr[equat] = nside
        fact1 = (2 * nside) * fact2
        z[equat] = (2 * nside - jr[equat]).to(f_dtype) * fact1
        kshift[equat] = (jr[equat] - nside) & 1

    jp = (_JPLL.to(face_num.device)[face_num] * nr + ix - iy + 1 + kshift) // 2
    jp = torch.where(jp > nl4, jp - nl4, jp)
    jp = torch.where(jp < 1, jp + nl4, jp)

    phi = (jp.to(f_dtype) - 0.5 * (kshift.to(f_dtype) + 1.0)) * (
        (math.pi / 2.0) / nr.to(f_dtype)
    )
    ra = torch.remainder(torch.rad2deg(phi), 360.0)
    dec = torch.rad2deg(torch.asin(torch.clamp(z, -1.0, 1.0)))
    return ra, dec


def ring2nest(nside: int, pix_ring: Tensor) -> Tensor:
    """Convert RING pixel indices to NESTED."""
    _validate_nside(nside)
    pix_ring_t = _as_int64(pix_ring)
    if pix_ring_t.device.type == "mps":
        return ring2nest(nside, pix_ring_t.cpu()).to(device=pix_ring_t.device)
    if _cpp is not None and pix_ring_t.device.type == "cpu":
        if hasattr(_cpp, "healpix_ring2nest_torch_cpu"):
            return _cpp.healpix_ring2nest_torch_cpu(nside, pix_ring_t.contiguous())
    ix, iy, face_num = _ring2xyf(nside, pix_ring_t)
    return _xyf2nest(nside, ix, iy, face_num)


def nest2ring(nside: int, pix_nest: Tensor) -> Tensor:
    """Convert NESTED pixel indices to RING."""
    _validate_nside(nside)
    pix_nest_t = _as_int64(pix_nest)
    if pix_nest_t.device.type == "mps":
        return nest2ring(nside, pix_nest_t.cpu()).to(device=pix_nest_t.device)
    if _cpp is not None and pix_nest_t.device.type == "cpu":
        if hasattr(_cpp, "healpix_nest2ring_torch_cpu"):
            return _cpp.healpix_nest2ring_torch_cpu(nside, pix_nest_t.contiguous())
    ix, iy, face_num = _nest2xyf(nside, pix_nest_t)
    return _xyf2ring(nside, ix, iy, face_num)


def nside2npix(nside: int) -> int:
    """Return number of pixels for a given NSIDE."""
    _validate_nside(nside)
    return 12 * nside * nside


def order2nside(order: int) -> int:
    """Return NSIDE for a given HEALPix order (nside = 2**order)."""
    if order < 0:
        raise ValueError("order must be non-negative")
    return 1 << order


def nside2order(nside: int) -> int:
    """Return HEALPix order for a given NSIDE (order = log2(nside))."""
    _validate_nside(nside)
    return int(nside.bit_length() - 1)


def npix2nside(npix: int) -> int:
    """Return NSIDE for a given number of pixels."""
    if npix <= 0 or (npix % 12) != 0:
        raise ValueError("npix must be positive and divisible by 12")
    nside_sq = npix // 12
    nside = int(math.isqrt(nside_sq))
    if nside * nside != nside_sq:
        raise ValueError("npix does not correspond to a valid HEALPix nside")
    _validate_nside(nside)
    return nside


def isnsideok(nside: Tensor | int | list[int], nest: bool = False) -> bool | Tensor:
    """
    Check whether NSIDE values are valid.

    Returns bool for scalar input and bool tensor for array-like input.
    """
    t = torch.as_tensor(nside)
    int_like = torch.ones_like(t, dtype=torch.bool)
    if t.is_floating_point():
        int_like = torch.isfinite(t) & (t == torch.floor(t))
    v = t.to(torch.int64)
    ok = int_like & (v > 0) & ((v & (v - 1)) == 0)
    if not nest:
        # Keep ring mode semantics simple and consistent with power-of-two NSIDE.
        ok = ok & torch.isfinite(v.to(torch.float64))
    if t.ndim == 0:
        return bool(ok.item())
    return ok


def isnpixok(npix: Tensor | int | list[int]) -> bool | Tensor:
    """
    Check whether NPIX values are valid HEALPix sizes.

    Returns bool for scalar input and bool tensor for array-like input.
    """
    t = torch.as_tensor(npix)
    int_like = torch.ones_like(t, dtype=torch.bool)
    if t.is_floating_point():
        int_like = torch.isfinite(t) & (t == torch.floor(t))
    v = t.to(torch.int64)
    base = int_like & (v > 0) & ((v % 12) == 0)
    nside_sq = torch.where(base, v // 12, torch.ones_like(v))
    nside = _isqrt(nside_sq)
    ok = base & (nside * nside == nside_sq) & ((nside & (nside - 1)) == 0)
    if t.ndim == 0:
        return bool(ok.item())
    return ok


def nside2pixarea(nside: int, degrees: bool = False) -> float:
    """Return pixel area for a given NSIDE."""
    _validate_nside(nside)
    area_sr = 4.0 * math.pi / float(nside2npix(nside))
    if not degrees:
        return area_sr
    return area_sr * ((180.0 / math.pi) ** 2)


def nside2resol(nside: int, arcmin: bool = False) -> float:
    """Return approximate resolution (sqrt pixel area)."""
    resol_rad = math.sqrt(nside2pixarea(nside, degrees=False))
    if not arcmin:
        return resol_rad
    return resol_rad * (180.0 * 60.0 / math.pi)


def max_pixrad(nside: int, degrees: bool = False) -> float:
    """Healpy-style alias for `max_pixel_radius`."""
    return max_pixel_radius(nside, degrees=degrees)


def ang2pix(
    nside: int, theta: Tensor, phi: Tensor, nest: bool = False, lonlat: bool = False
) -> Tensor:
    """
    Convert angles to pixel indices.

    If lonlat is False, inputs are (theta, phi) in radians with theta=colatitude.
    If lonlat is True, inputs are (lon, lat) in degrees.
    """
    if lonlat:
        lon = _as_float64(theta)
        lat = _as_float64(phi)
    else:
        theta_t = _as_float64(theta)
        phi_t = _as_float64(phi)
        lon = torch.rad2deg(phi_t)
        lat = 90.0 - torch.rad2deg(theta_t)
    if nest:
        return ang2pix_nested(nside, lon, lat)
    return ang2pix_ring(nside, lon, lat)


def ang2vec(theta: Tensor | float, phi: Tensor | float, lonlat: bool = False) -> Tensor:
    """
    Convert angles to unit vectors.

    If lonlat=False, input is (theta, phi) in radians with theta=colatitude.
    If lonlat=True, input is (lon, lat) in degrees.
    Returns vectors with last dimension 3.
    """
    if lonlat:
        lon = _as_float64(theta)
        lat = _as_float64(phi)
    else:
        theta_t = _as_float64(theta)
        phi_t = _as_float64(phi)
        lon = torch.rad2deg(phi_t)
        lat = 90.0 - torch.rad2deg(theta_t)
    x, y, z = lonlat_to_xyz(lon, lat)
    return torch.stack([x, y, z], dim=-1)


def pix2ang(
    nside: int, ipix: Tensor, nest: bool = False, lonlat: bool = False
) -> Tuple[Tensor, Tensor]:
    """
    Convert pixel indices to angles.

    If lonlat is False, returns (theta, phi) in radians with theta=colatitude.
    If lonlat is True, returns (lon, lat) in degrees.
    """
    if nest:
        lon, lat = pix2ang_nested(nside, ipix)
    else:
        lon, lat = pix2ang_ring(nside, ipix)
    if lonlat:
        return lon, lat
    theta = torch.deg2rad(90.0 - lat)
    phi = torch.deg2rad(lon)
    return theta, phi


def vec2ang(
    vectors: Tensor | Sequence[Sequence[float]], lonlat: bool = False
) -> Tuple[Tensor, Tensor]:
    """
    Convert vectors to angles.

    Accepts shape (..., 3) or (3, ...). If lonlat=False returns (theta, phi)
    in radians, otherwise returns (lon, lat) in degrees.
    """
    v = torch.as_tensor(vectors)
    if v.ndim == 0:
        raise ValueError("vectors must be at least 1D")
    if v.shape[-1] == 3:
        x, y, z = v[..., 0], v[..., 1], v[..., 2]
    elif v.shape[0] == 3:
        x, y, z = v[0], v[1], v[2]
    else:
        raise ValueError("vectors must have shape (...,3) or (3,...)")
    lon, lat = xyz_to_lonlat(x, y, z)
    lon = lon.reshape(-1)
    lat = lat.reshape(-1)
    if lonlat:
        return lon, lat
    theta = torch.deg2rad(90.0 - lat)
    phi = torch.deg2rad(lon)
    return theta, phi


def vec2pix(nside: int, x: Tensor, y: Tensor, z: Tensor, nest: bool = False) -> Tensor:
    """Convert Cartesian vectors to pixel indices."""
    lon, lat = xyz_to_lonlat(x, y, z)
    if nest:
        return ang2pix_nested(nside, lon, lat)
    return ang2pix_ring(nside, lon, lat)


def pix2vec(
    nside: int, ipix: Tensor, nest: bool = False
) -> Tuple[Tensor, Tensor, Tensor]:
    """Convert pixel indices to Cartesian vectors."""
    lon, lat = pix2ang(nside, ipix, nest=nest, lonlat=True)
    return lonlat_to_xyz(lon, lat)


def boundaries(nside: int, pix: Tensor, step: int = 1, nest: bool = False) -> Tensor:
    """
    Return pixel boundary points as Cartesian vectors.

    Output shape:
    - scalar `pix`: (3, 4*step)
    - 1D `pix`: (Npix, 3, 4*step)
    """
    _validate_nside(nside)
    if step <= 0:
        raise ValueError("step must be positive")

    pix_t = _as_int64(pix)
    if pix_t.ndim > 1:
        raise ValueError("Array has to be one dimensional")
    scalar_input = pix_t.ndim == 0
    if scalar_input:
        pix_t = pix_t.reshape(1)

    if not nest:
        pix_t = ring2nest(nside, pix_t)

    npix = nside2npix(nside)
    if torch.any((pix_t < 0) | (pix_t >= npix)):
        raise ValueError("pixel index out of range for nside")

    ix, iy, face_num = _nest2xyf(nside, pix_t)
    f_dtype = _float_dtype_for_device(pix_t.device)
    ix_f = ix.to(f_dtype)
    iy_f = iy.to(f_dtype)

    u = (torch.arange(step, device=pix_t.device, dtype=f_dtype) / step).unsqueeze(0)
    x = torch.cat(
        [
            (ix_f + 1.0).unsqueeze(1) - u,
            ix_f.unsqueeze(1).expand(-1, step),
            ix_f.unsqueeze(1) + u,
            (ix_f + 1.0).unsqueeze(1).expand(-1, step),
        ],
        dim=1,
    )
    y = torch.cat(
        [
            (iy_f + 1.0).unsqueeze(1).expand(-1, step),
            (iy_f + 1.0).unsqueeze(1) - u,
            iy_f.unsqueeze(1).expand(-1, step),
            iy_f.unsqueeze(1) + u,
        ],
        dim=1,
    )

    face_e = face_num.unsqueeze(1).expand_as(x)
    jrll = _JRLL.to(device=pix_t.device)[face_e].to(f_dtype)
    jpll = _JPLL.to(device=pix_t.device)[face_e].to(f_dtype)

    jr = jrll * nside - x - y
    fact2 = 4.0 / npix
    fact1 = (2 * nside) * fact2
    nr = torch.empty_like(jr)
    z = torch.empty_like(jr)

    north = jr < nside
    south = jr > (3 * nside)
    equat = ~(north | south)

    nr[north] = jr[north]
    z[north] = 1.0 - (nr[north] ** 2) * fact2
    nr[south] = 4 * nside - jr[south]
    z[south] = -1.0 + (nr[south] ** 2) * fact2
    nr[equat] = nside
    z[equat] = (2 * nside - jr[equat]) * fact1

    nr_safe = torch.where(nr == 0.0, torch.ones_like(nr), nr)
    phi = (jpll * nr + x - y) * (math.pi * 0.25) / nr_safe
    phi = torch.where(nr == 0.0, torch.zeros_like(phi), phi)

    lat = torch.asin(torch.clamp(z, -1.0, 1.0))
    c = torch.cos(lat)
    xyz = torch.stack([c * torch.cos(phi), c * torch.sin(phi), torch.sin(lat)], dim=1)

    if scalar_input:
        return xyz[0]
    return xyz


def ring_to_nested(nside: int, ipix: Tensor) -> Tensor:
    """Alias for ring2nest."""
    return ring2nest(nside, ipix)


def nested_to_ring(nside: int, ipix: Tensor) -> Tensor:
    """Alias for nest2ring."""
    return nest2ring(nside, ipix)


def neighbors(nside: int, ipix: Tensor | int, nest: bool = False) -> Tensor:
    """
    Return 8 neighbors (SW, W, NW, N, NE, E, SE, S) for input pixels.

    Output shape is `ipix.shape + (8,)` for array inputs and `(8,)` for scalars.
    Missing neighbors are set to -1.
    """
    _validate_nside(nside)
    pix_t = _as_int64(ipix)
    scalar_input = pix_t.ndim == 0
    if scalar_input:
        pix_flat = pix_t.reshape(1)
    else:
        pix_flat = pix_t.reshape(-1)

    npix = nside2npix(nside)
    if torch.any((pix_flat < 0) | (pix_flat >= npix)):
        raise ValueError("ipix out of range for nside")

    if pix_flat.device.type == "cpu" and _cpp is not None:
        if nest and hasattr(_cpp, "healpix_neighbors_nested_torch_cpu"):
            out_cpp = _cpp.healpix_neighbors_nested_torch_cpu(nside, pix_t.contiguous())
            return out_cpp
        if (not nest) and hasattr(_cpp, "healpix_neighbors_ring_torch_cpu"):
            out_cpp = _cpp.healpix_neighbors_ring_torch_cpu(nside, pix_t.contiguous())
            return out_cpp

    if nest:
        ix, iy, face = _nest2xyf(nside, pix_flat)
    else:
        ix, iy, face = _ring2xyf(nside, pix_flat)

    nsm1 = nside - 1
    out = torch.full(
        (pix_flat.numel(), 8), -1, dtype=torch.int64, device=pix_flat.device
    )
    interior = (ix > 0) & (ix < nsm1) & (iy > 0) & (iy < nsm1)

    xoff = _NB_XOFFSET.to(device=pix_flat.device)
    yoff = _NB_YOFFSET.to(device=pix_flat.device)

    if torch.any(interior):
        ix_i = ix[interior]
        iy_i = iy[interior]
        face_i = face[interior]
        x_int = ix_i.unsqueeze(1) + xoff.unsqueeze(0)
        y_int = iy_i.unsqueeze(1) + yoff.unsqueeze(0)
        f_int = face_i.unsqueeze(1).expand_as(x_int)
        if nest:
            vals = _xyf2nest(
                nside, x_int.reshape(-1), y_int.reshape(-1), f_int.reshape(-1)
            )
        else:
            vals = _xyf2ring(
                nside, x_int.reshape(-1), y_int.reshape(-1), f_int.reshape(-1)
            )
        out[interior] = vals.reshape(-1, 8)

    boundary = ~interior
    if torch.any(boundary):
        ix_b = ix[boundary]
        iy_b = iy[boundary]
        face_b = face[boundary]
        out_b = out[boundary]
        facearr = _NB_FACEARRAY.to(device=pix_flat.device)
        swaparr = _NB_SWAPARRAY.to(device=pix_flat.device)
        band = face_b >> 2

        for m in range(8):
            x = ix_b + xoff[m]
            y = iy_b + yoff[m]
            nbnum = torch.full_like(x, 4)

            lx = x < 0
            gx = x >= nside
            x = torch.where(lx, x + nside, x)
            x = torch.where(gx, x - nside, x)
            nbnum = torch.where(lx, nbnum - 1, nbnum)
            nbnum = torch.where(gx, nbnum + 1, nbnum)

            ly = y < 0
            gy = y >= nside
            y = torch.where(ly, y + nside, y)
            y = torch.where(gy, y - nside, y)
            nbnum = torch.where(ly, nbnum - 3, nbnum)
            nbnum = torch.where(gy, nbnum + 3, nbnum)

            f = facearr[nbnum, face_b]
            valid = f >= 0
            if not torch.any(valid):
                continue

            bits = swaparr[nbnum, band]
            xv = x[valid]
            yv = y[valid]
            bv = bits[valid]

            flip_x = (bv & 1) != 0
            flip_y = (bv & 2) != 0
            swap_xy = (bv & 4) != 0

            xv = torch.where(flip_x, nside - xv - 1, xv)
            yv = torch.where(flip_y, nside - yv - 1, yv)
            x_new = torch.where(swap_xy, yv, xv)
            y_new = torch.where(swap_xy, xv, yv)

            vals = (
                _xyf2nest(nside, x_new, y_new, f[valid])
                if nest
                else _xyf2ring(nside, x_new, y_new, f[valid])
            )
            tmp = out_b[:, m]
            tmp[valid] = vals
            out_b[:, m] = tmp
        out[boundary] = out_b

    if scalar_input:
        return out[0]
    return out.reshape(*pix_t.shape, 8)


def neighbours(nside: int, ipix: Tensor | int, nest: bool = False) -> Tensor:
    """British spelling alias for `neighbors`."""
    return neighbors(nside, ipix, nest=nest)


def get_all_neighbours(
    nside: int,
    theta: Tensor | float | int,
    phi: Tensor | float | None = None,
    nest: bool = False,
    lonlat: bool = False,
) -> Tensor:
    """
    Healpy-compatible neighbors API.

    If `phi` is None, `theta` is interpreted as pixel index. Otherwise,
    `(theta, phi)` are interpreted as angular coordinates.
    """
    _validate_nside(nside)
    if phi is None:
        pix = _as_int64(theta)
    else:
        pix = ang2pix(nside, theta, phi, nest=nest, lonlat=lonlat)
    neigh = neighbors(nside, pix, nest=nest)
    if neigh.ndim == 1:
        return neigh
    return torch.movedim(neigh, -1, 0)


def get_interp_weights(
    nside: int,
    theta: Tensor | float | int,
    phi: Tensor | float | None = None,
    nest: bool = False,
    lonlat: bool = False,
) -> tuple[Tensor, Tensor]:
    """
    Healpy-compatible bilinear interpolation neighbors and weights.

    Returns `(pixels, weights)` with shape `(4, ...)`.
    """
    _validate_nside(nside)
    if phi is None:
        pix_in = _as_int64(theta)
        if pix_in.device.type == "cpu" and _cpp is not None:
            # Fast path for pixel-center queries: convert to lon/lat and reuse
            # C++ interpolation-weight kernels. This matches healpy semantics.
            lon_t, lat_t = pix2ang(nside, pix_in, nest=nest, lonlat=True)
            if nest and hasattr(_cpp, "healpix_get_interp_weights_nested_torch_cpu"):
                pix_cpp, w_cpp = _cpp.healpix_get_interp_weights_nested_torch_cpu(
                    nside,
                    lon_t.contiguous(),
                    lat_t.contiguous(),
                )
                return pix_cpp.reshape(4, *pix_in.shape), w_cpp.reshape(
                    4, *pix_in.shape
                )
            if (not nest) and hasattr(
                _cpp, "healpix_get_interp_weights_ring_torch_cpu"
            ):
                pix_cpp, w_cpp = _cpp.healpix_get_interp_weights_ring_torch_cpu(
                    nside,
                    lon_t.contiguous(),
                    lat_t.contiguous(),
                )
                return pix_cpp.reshape(4, *pix_in.shape), w_cpp.reshape(
                    4, *pix_in.shape
                )

        pix_ring = nest2ring(nside, pix_in) if nest else pix_in
        theta_t, phi_t = _pix2thetaphi_ring(nside, pix_ring)
    else:
        lon_or_theta = _as_float64(theta)
        lat_or_phi = _as_float64(phi)
        lon_or_theta, lat_or_phi = torch.broadcast_tensors(lon_or_theta, lat_or_phi)
        if lonlat:
            lon_t = lon_or_theta
            lat_t = lat_or_phi
        else:
            lon_t = torch.rad2deg(lat_or_phi)
            lat_t = 90.0 - torch.rad2deg(lon_or_theta)

        if lon_t.device.type == "cpu" and _cpp is not None:
            if nest and hasattr(_cpp, "healpix_get_interp_weights_nested_torch_cpu"):
                pix_cpp, w_cpp = _cpp.healpix_get_interp_weights_nested_torch_cpu(
                    nside,
                    lon_t.contiguous(),
                    lat_t.contiguous(),
                )
                return pix_cpp.reshape(4, *lon_t.shape), w_cpp.reshape(4, *lon_t.shape)
            if (not nest) and hasattr(
                _cpp, "healpix_get_interp_weights_ring_torch_cpu"
            ):
                pix_cpp, w_cpp = _cpp.healpix_get_interp_weights_ring_torch_cpu(
                    nside,
                    lon_t.contiguous(),
                    lat_t.contiguous(),
                )
                return pix_cpp.reshape(4, *lon_t.shape), w_cpp.reshape(4, *lon_t.shape)

        theta_t = torch.deg2rad(90.0 - lat_t)
        phi_t = torch.deg2rad(lon_t)

    theta_f = _as_float64(theta_t).reshape(-1).to(torch.float64)
    phi_f = torch.remainder(
        _as_float64(phi_t).reshape(-1).to(torch.float64), 2.0 * math.pi
    )
    shape = theta_t.shape

    z = torch.cos(theta_f)
    ir1 = _ring_above(nside, z)
    ir2 = ir1 + 1

    n = theta_f.numel()
    pix0 = torch.full((n,), -1, dtype=torch.int64, device=theta_f.device)
    pix1 = torch.full((n,), -1, dtype=torch.int64, device=theta_f.device)
    pix2 = torch.full((n,), -1, dtype=torch.int64, device=theta_f.device)
    pix3 = torch.full((n,), -1, dtype=torch.int64, device=theta_f.device)
    w0 = torch.zeros((n,), dtype=torch.float64, device=theta_f.device)
    w1 = torch.zeros((n,), dtype=torch.float64, device=theta_f.device)
    w2 = torch.zeros((n,), dtype=torch.float64, device=theta_f.device)
    w3 = torch.zeros((n,), dtype=torch.float64, device=theta_f.device)

    theta1 = torch.zeros((n,), dtype=torch.float64, device=theta_f.device)
    theta2 = torch.zeros((n,), dtype=torch.float64, device=theta_f.device)

    m1 = ir1 > 0
    if torch.any(m1):
        sp, nr, th, shift = _get_ring_info2(nside, ir1[m1])
        theta1[m1] = th
        dphi = (2.0 * math.pi) / nr.to(torch.float64)
        sh = shift.to(torch.float64)
        tmp = phi_f[m1] / dphi - 0.5 * sh
        i1 = torch.floor(tmp).to(torch.int64)
        wt = (phi_f[m1] - (i1.to(torch.float64) + 0.5 * sh) * dphi) / dphi
        i2 = i1 + 1
        i1 = torch.where(i1 < 0, i1 + nr, i1)
        i2 = torch.where(i2 >= nr, i2 - nr, i2)
        pix0[m1] = sp + i1
        pix1[m1] = sp + i2
        w0[m1] = 1.0 - wt
        w1[m1] = wt

    m2 = ir2 < (4 * nside)
    if torch.any(m2):
        sp, nr, th, shift = _get_ring_info2(nside, ir2[m2])
        theta2[m2] = th
        dphi = (2.0 * math.pi) / nr.to(torch.float64)
        sh = shift.to(torch.float64)
        tmp = phi_f[m2] / dphi - 0.5 * sh
        i1 = torch.floor(tmp).to(torch.int64)
        wt = (phi_f[m2] - (i1.to(torch.float64) + 0.5 * sh) * dphi) / dphi
        i2 = i1 + 1
        i1 = torch.where(i1 < 0, i1 + nr, i1)
        i2 = torch.where(i2 >= nr, i2 - nr, i2)
        pix2[m2] = sp + i1
        pix3[m2] = sp + i2
        w2[m2] = 1.0 - wt
        w3[m2] = wt

    top = ir1 == 0
    if torch.any(top):
        wtheta = theta_f[top] / theta2[top]
        w2[top] = w2[top] * wtheta
        w3[top] = w3[top] * wtheta
        fac = (1.0 - wtheta) * 0.25
        w0[top] = fac
        w1[top] = fac
        w2[top] = w2[top] + fac
        w3[top] = w3[top] + fac
        pix0[top] = (pix2[top] + 2) & 3
        pix1[top] = (pix3[top] + 2) & 3

    bottom = ir2 == (4 * nside)
    if torch.any(bottom):
        wtheta = (theta_f[bottom] - theta1[bottom]) / (math.pi - theta1[bottom])
        w0[bottom] = w0[bottom] * (1.0 - wtheta)
        w1[bottom] = w1[bottom] * (1.0 - wtheta)
        fac = wtheta * 0.25
        w0[bottom] = w0[bottom] + fac
        w1[bottom] = w1[bottom] + fac
        w2[bottom] = fac
        w3[bottom] = fac
        npix = nside2npix(nside)
        pix2[bottom] = ((pix0[bottom] + 2) & 3) + npix - 4
        pix3[bottom] = ((pix1[bottom] + 2) & 3) + npix - 4

    middle = (~top) & (~bottom)
    if torch.any(middle):
        wtheta = (theta_f[middle] - theta1[middle]) / (theta2[middle] - theta1[middle])
        w0[middle] = w0[middle] * (1.0 - wtheta)
        w1[middle] = w1[middle] * (1.0 - wtheta)
        w2[middle] = w2[middle] * wtheta
        w3[middle] = w3[middle] * wtheta

    pix = torch.stack([pix0, pix1, pix2, pix3], dim=0)
    wgt = torch.stack([w0, w1, w2, w3], dim=0)
    if nest:
        pix = ring2nest(nside, pix.reshape(-1)).reshape(4, n)

    pix = pix.reshape(4, *shape)
    wgt = wgt.reshape(4, *shape)
    return pix, wgt


def get_interp_val(
    m: Tensor | list[float],
    theta: Tensor | float,
    phi: Tensor | float,
    nest: bool = False,
    lonlat: bool = False,
) -> Tensor:
    """
    Healpy-compatible bilinear interpolation of map values.

    `m` can be shape `(npix,)` or `(nmaps, npix)`.
    """
    m_t = torch.as_tensor(m)
    one_map = m_t.ndim == 1
    if one_map:
        maps = m_t.unsqueeze(0)
    elif m_t.ndim == 2:
        maps = m_t
    else:
        raise ValueError("m must be shape (npix,) or (nmaps, npix)")

    npix = int(maps.shape[1])
    nside = npix2nside(npix)

    if (
        _cpp is not None
        and maps.device.type == "cpu"
        and maps.is_floating_point()
        and nest
    ):
        lon_or_theta = _as_float64(theta)
        lat_or_phi = _as_float64(phi)
        lon_or_theta, lat_or_phi = torch.broadcast_tensors(lon_or_theta, lat_or_phi)
        if lonlat:
            lon_t = lon_or_theta
            lat_t = lat_or_phi
        else:
            lon_t = torch.rad2deg(lat_or_phi)
            lat_t = 90.0 - torch.rad2deg(lon_or_theta)
        if lon_t.device.type == "cpu":
            maps_in = m_t if one_map else maps
            if hasattr(_cpp, "healpix_get_interp_val_nested_torch_cpu"):
                return _cpp.healpix_get_interp_val_nested_torch_cpu(
                    nside,
                    maps_in.contiguous(),
                    lon_t.contiguous(),
                    lat_t.contiguous(),
                )

    pix, wgt = get_interp_weights(nside, theta, phi, nest=nest, lonlat=lonlat)
    qshape = pix.shape[1:]
    pix_f = pix.reshape(4, -1)
    w_f = wgt.reshape(4, -1).to(dtype=maps.dtype)
    # Gather all 4 neighbors in one advanced-indexing pass to reduce
    # index_select dispatch overhead on large query batches.
    neigh_vals = maps[:, pix_f]  # [nmaps,4,nq]
    out = (neigh_vals * w_f.unsqueeze(0)).sum(dim=1)
    out = out.reshape(maps.shape[0], *qshape)
    if one_map:
        return out[0]
    return out


def _normalize_xyz(
    x: Tensor, y: Tensor, z: Tensor, eps: float = 1.0e-15
) -> tuple[Tensor, Tensor, Tensor]:
    n = torch.sqrt(x * x + y * y + z * z).clamp_min(eps)
    return x / n, y / n, z / n


def _pixel_range_chunks(
    npix: int, chunk_size: int = 1_000_000
) -> list[tuple[int, int]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    out: list[tuple[int, int]] = []
    start = 0
    while start < npix:
        end = min(npix, start + chunk_size)
        out.append((start, end))
        start = end
    return out


def _ring_above(nside: int, z: Tensor) -> Tensor:
    """Vectorized HEALPix ring_above(z) helper."""
    zf = torch.as_tensor(z, dtype=torch.float64)
    az = torch.abs(zf)
    eq = az <= (2.0 / 3.0)

    ir_eq = torch.floor(nside * (2.0 - 1.5 * zf)).to(torch.int64)
    ir_p = torch.floor(nside * torch.sqrt(torch.clamp(3.0 * (1.0 - az), min=0.0))).to(
        torch.int64
    )
    ir = torch.where(eq, ir_eq, torch.where(zf > 0.0, ir_p, 4 * nside - ir_p - 1))
    return ir


def _get_ring_info2(nside: int, ring: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Vectorized get_ring_info2 helper for interpolation."""
    r = torch.as_tensor(ring, dtype=torch.int64)
    npix = nside2npix(nside)
    ncap = 2 * nside * (nside - 1)
    fact2 = 4.0 / float(npix)
    fact1 = (2.0 * float(nside)) * fact2

    northr = torch.where(r > 2 * nside, 4 * nside - r, r)
    northf = northr.to(torch.float64)
    polar = northr < nside

    tmp = northf * northf * fact2
    costheta_p = 1.0 - tmp
    sintheta_p = torch.sqrt(torch.clamp(tmp * (2.0 - tmp), min=0.0))
    theta_p = torch.atan2(sintheta_p, costheta_p)
    ringpix_p = 4 * northr
    shifted_p = torch.ones_like(polar, dtype=torch.bool)
    startpix_p = 2 * northr * (northr - 1)

    theta_e = torch.acos(torch.clamp((2.0 * float(nside) - northf) * fact1, -1.0, 1.0))
    ringpix_e = torch.full_like(r, 4 * nside)
    shifted_e = ((northr - nside) & 1) == 0
    startpix_e = ncap + (northr - nside) * ringpix_e

    theta = torch.where(polar, theta_p, theta_e)
    ringpix = torch.where(polar, ringpix_p, ringpix_e)
    shifted = torch.where(polar, shifted_p, shifted_e)
    startpix = torch.where(polar, startpix_p, startpix_e)

    south = northr != r
    theta = torch.where(south, math.pi - theta, theta)
    startpix = torch.where(south, npix - startpix - ringpix, startpix)

    return startpix, ringpix, theta, shifted


def max_pixel_radius(nside: int, degrees: bool = False) -> float:
    """
    Return a conservative maximum center-to-corner radius for pixels at `nside`.

    Uses representative polar/equatorial/southern pixels and caches result.
    """
    _validate_nside(nside)
    cached = _MAX_PIXEL_RADIUS_CACHE.get(nside)
    if cached is None:
        npix = nside2npix(nside)
        reps = torch.tensor([0, 2 * nside * (nside - 1), npix - 1], dtype=torch.int64)
        lon_c, lat_c = pix2ang_ring(nside, reps)
        b = boundaries(nside, reps, step=1, nest=False)  # (N,3,4)
        lon_b, lat_b = xyz_to_lonlat(b[:, 0, :], b[:, 1, :], b[:, 2, :])
        d = angular_distance_deg(
            lon_c.unsqueeze(-1),
            lat_c.unsqueeze(-1),
            lon_b,
            lat_b,
        )
        cached = float(torch.max(d).item())
        _MAX_PIXEL_RADIUS_CACHE[nside] = cached
    if degrees:
        return cached
    return math.radians(cached)


def pixel_resolution_to_nside(
    resolution: float,
    *,
    degrees: bool = False,
    arcmin: bool = False,
    round_to_power_of_two: bool = True,
) -> int:
    """Approximate NSIDE from angular resolution."""
    if resolution <= 0.0:
        raise ValueError("resolution must be positive")
    if degrees and arcmin:
        raise ValueError("set only one of degrees/arcmin")

    if arcmin:
        res_rad = math.radians(resolution / 60.0)
    elif degrees:
        res_rad = math.radians(resolution)
    else:
        res_rad = resolution

    n_est = math.sqrt(math.pi / 3.0) / res_rad
    if n_est < 1.0:
        return 1
    if not round_to_power_of_two:
        return max(1, int(round(n_est)))

    log2n = math.log2(n_est)
    nside = 1 << max(0, int(round(log2n)))
    _validate_nside(nside)
    return nside


def query_disc(
    nside: int,
    vec: Tensor | Sequence[float],
    radius: float,
    *,
    inclusive: bool = False,
    fact: int = 4,  # noqa: ARG001 - kept for API compatibility
    nest: bool = False,
    chunk_size: int = 1_000_000,
) -> Tensor:
    """
    Return pixels whose centers fall in a spherical disc.

    `vec` is a 3-vector (x, y, z); `radius` is in radians.
    If `inclusive=True`, use a conservative radius expansion by max pixel radius.
    """
    _validate_nside(nside)
    if radius <= 0.0:
        return torch.empty((0,), dtype=torch.int64)

    v = torch.as_tensor(vec, dtype=torch.float64).reshape(-1)
    if v.numel() != 3:
        raise ValueError("vec must be length-3 Cartesian vector")
    vx, vy, vz = _normalize_xyz(v[0], v[1], v[2])
    radius_eff = radius + (max_pixel_radius(nside, degrees=False) if inclusive else 0.0)
    cos_lim = math.cos(radius_eff)

    if (
        _cpp is not None
        and hasattr(_cpp, "healpix_query_disc_torch_cpu")
        and v.device.type == "cpu"
    ):
        vec_t = torch.stack([vx, vy, vz]).to(torch.float64).contiguous()
        return _cpp.healpix_query_disc_torch_cpu(nside, vec_t, float(cos_lim), nest)

    npix = nside2npix(nside)
    selected: list[Tensor] = []
    for start, end in _pixel_range_chunks(npix, chunk_size=chunk_size):
        pix = torch.arange(start, end, dtype=torch.int64)
        x, y, z = pix2vec(nside, pix, nest=nest)
        dots = x * vx + y * vy + z * vz
        keep = dots >= cos_lim
        if torch.any(keep):
            selected.append(pix[keep])
    if not selected:
        return torch.empty((0,), dtype=torch.int64)
    return torch.cat(selected, dim=0)


def query_circle(
    nside: int,
    lon: Tensor | float,
    lat: Tensor | float,
    radius: float,
    *,
    degrees: bool = True,
    inclusive: bool = False,
    nest: bool = False,
    chunk_size: int = 1_000_000,
) -> Tensor:
    """Lon/lat convenience wrapper around `query_disc`."""
    lon_t = torch.as_tensor(lon, dtype=torch.float64)
    lat_t = torch.as_tensor(lat, dtype=torch.float64)
    x, y, z = lonlat_to_xyz(lon_t, lat_t)
    if degrees:
        radius = math.radians(radius)
    return query_disc(
        nside,
        (float(x.item()), float(y.item()), float(z.item())),
        radius,
        inclusive=inclusive,
        nest=nest,
        chunk_size=chunk_size,
    )


def query_circle_vec(
    nside: int,
    vec: Tensor | Sequence[float],
    radius: float,
    *,
    inclusive: bool = False,
    fact: int = 4,  # noqa: ARG001 - kept for API compatibility
    nest: bool = False,
    chunk_size: int = 1_000_000,
) -> Tensor:
    """Cartesian-vector convenience wrapper around `query_disc`."""
    return query_disc(
        nside,
        vec,
        radius,
        inclusive=inclusive,
        fact=fact,
        nest=nest,
        chunk_size=chunk_size,
    )


def query_strip(
    nside: int,
    theta1: float,
    theta2: float,
    *,
    nest: bool = False,
    inclusive: bool = False,
    chunk_size: int = 1_000_000,
) -> Tensor:
    """
    Return pixels with center colatitude in [theta1, theta2] (radians).
    """
    _validate_nside(nside)
    lo = min(theta1, theta2)
    hi = max(theta1, theta2)
    if inclusive:
        pad = max_pixel_radius(nside, degrees=False)
        lo -= pad
        hi += pad
    lo = max(0.0, lo)
    hi = min(math.pi, hi)

    npix = nside2npix(nside)
    selected: list[Tensor] = []
    for start, end in _pixel_range_chunks(npix, chunk_size=chunk_size):
        pix = torch.arange(start, end, dtype=torch.int64)
        lon, lat = pix2ang(nside, pix, nest=nest, lonlat=True)
        theta = torch.deg2rad(90.0 - lat)
        keep = (theta >= lo) & (theta <= hi)
        if torch.any(keep):
            selected.append(pix[keep])
    if not selected:
        return torch.empty((0,), dtype=torch.int64)
    return torch.cat(selected, dim=0)


def query_polygon(
    nside: int,
    vertices: Tensor | Sequence[Sequence[float]],
    *,
    inclusive: bool = False,
    nest: bool = False,
    lonlat: bool = False,
    chunk_size: int = 1_000_000,
) -> Tensor:
    """
    Return pixels with centers inside a (convex) spherical polygon.

    `vertices` accepts:
    - Cartesian shape (N,3) or (3,N), if `lonlat=False`
    - Lon/lat degrees shape (N,2) or (2,N), if `lonlat=True`
    """
    _validate_nside(nside)
    v = torch.as_tensor(vertices, dtype=torch.float64)
    if v.ndim != 2:
        raise ValueError("vertices must be 2D")

    if lonlat:
        if v.shape[-1] == 2:
            lon = v[:, 0]
            lat = v[:, 1]
        elif v.shape[0] == 2:
            lon = v[0]
            lat = v[1]
        else:
            raise ValueError("lonlat vertices must be shape (N,2) or (2,N)")
        vx, vy, vz = lonlat_to_xyz(lon, lat)
        vv = torch.stack([vx, vy, vz], dim=-1)
    else:
        if v.shape[-1] == 3:
            vv = v
        elif v.shape[0] == 3:
            vv = v.transpose(0, 1)
        else:
            raise ValueError("Cartesian vertices must be shape (N,3) or (3,N)")

    if vv.shape[0] < 3:
        raise ValueError("at least 3 vertices required")
    vv = vv / torch.sqrt((vv * vv).sum(dim=-1, keepdim=True)).clamp_min(1.0e-15)

    # Spherical half-space test: for each oriented edge (v_i -> v_{i+1}),
    # inside points lie on the same side of the edge great-circle plane.
    m = vv.shape[0]
    vnext = torch.roll(vv, shifts=-1, dims=0)
    edge_normals = torch.cross(vv, vnext, dim=1)
    edge_norm = torch.sqrt((edge_normals * edge_normals).sum(dim=1))
    if torch.any(edge_norm <= 1.0e-15):
        raise ValueError("degenerate polygon: repeated or colinear adjacent vertices")
    edge_normals = edge_normals / edge_norm.unsqueeze(1)

    centroid = vv.mean(dim=0)
    centroid_norm = torch.sqrt((centroid * centroid).sum())
    if centroid_norm <= 1.0e-15:
        raise ValueError("degenerate polygon: centroid norm too small")
    centroid = centroid / centroid_norm
    signs = torch.empty((m,), dtype=vv.dtype)
    for i in range(m):
        probe = vv[(i + 2) % m]
        s = torch.dot(probe, edge_normals[i])
        if torch.abs(s) <= 1.0e-12:
            s = torch.dot(centroid, edge_normals[i])
        if torch.abs(s) <= 1.0e-12:
            raise ValueError("degenerate polygon orientation")
        signs[i] = torch.sign(s)
    edge_normals = edge_normals * signs.unsqueeze(1)

    tol = math.sin(max_pixel_radius(nside, degrees=False)) if inclusive else 0.0
    npix = nside2npix(nside)
    selected: list[Tensor] = []
    for start, end in _pixel_range_chunks(npix, chunk_size=chunk_size):
        pix = torch.arange(start, end, dtype=torch.int64)
        x, y, z = pix2vec(nside, pix, nest=nest)
        p = torch.stack([x, y, z], dim=-1)
        side = p @ edge_normals.transpose(0, 1)
        keep = torch.all(side >= -tol, dim=1)
        if torch.any(keep):
            selected.append(pix[keep])
    if not selected:
        return torch.empty((0,), dtype=torch.int64)
    return torch.cat(selected, dim=0)


def query_polygon_vec(
    nside: int,
    vertices: Tensor | Sequence[Sequence[float]],
    *,
    inclusive: bool = False,
    fact: int = 4,  # noqa: ARG001 - kept for API compatibility
    nest: bool = False,
    chunk_size: int = 1_000_000,
) -> Tensor:
    """Cartesian-vector convenience wrapper around `query_polygon`."""
    return query_polygon(
        nside,
        vertices,
        inclusive=inclusive,
        nest=nest,
        lonlat=False,
        chunk_size=chunk_size,
    )


def pixel_ranges_to_pixels(
    pixel_ranges: Tensor | Sequence[Sequence[int]], *, inclusive: bool = False
) -> Tensor:
    """
    Expand pixel ranges into explicit pixels.

    Range semantics match `hpgeom.pixel_ranges_to_pixels`:
    - `inclusive=False`: each row is [start, stop) (half-open)
    - `inclusive=True`: each row is [start, stop] (closed)
    """
    ranges = torch.as_tensor(pixel_ranges, dtype=torch.int64)
    if ranges.ndim != 2 or ranges.shape[1] != 2:
        raise ValueError("pixel_ranges must have shape (N, 2)")
    if ranges.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=ranges.device)

    parts: list[Tensor] = []
    for i in range(ranges.shape[0]):
        start = int(ranges[i, 0].item())
        stop = int(ranges[i, 1].item()) + (1 if inclusive else 0)
        if stop <= start:
            continue
        parts.append(torch.arange(start, stop, dtype=torch.int64, device=ranges.device))
    if not parts:
        return torch.empty((0,), dtype=torch.int64, device=ranges.device)
    return torch.cat(parts, dim=0)


def pixels_to_pixel_ranges(pixels: Tensor | Sequence[int]) -> Tensor:
    """
    Compress explicit pixels into sorted disjoint half-open ranges [start, stop).

    Unlike `hpgeom`, this helper is torchfits-native and canonicalizes by
    sorting + deduplicating input pixels before range construction.
    """
    pix = torch.as_tensor(pixels, dtype=torch.int64).reshape(-1)
    if pix.numel() == 0:
        return torch.empty((0, 2), dtype=torch.int64, device=pix.device)

    pix = torch.unique(pix, sorted=True)
    if pix.numel() == 1:
        return torch.stack([pix, pix + 1], dim=1)

    gap = pix[1:] != (pix[:-1] + 1)
    start_idx = torch.cat(
        [
            torch.zeros((1,), dtype=torch.int64, device=pix.device),
            torch.nonzero(gap, as_tuple=False).flatten() + 1,
        ]
    )
    end_idx = torch.cat(
        [
            torch.nonzero(gap, as_tuple=False).flatten() + 1,
            torch.tensor([pix.numel()], dtype=torch.int64, device=pix.device),
        ]
    )
    starts = pix.index_select(0, start_idx)
    stops = pix.index_select(0, end_idx - 1) + 1
    return torch.stack([starts, stops], dim=1)


def upgrade_pixel_ranges(
    nside: int,
    pixel_ranges: Tensor | Sequence[Sequence[int]],
    nside_upgrade: int,
) -> Tensor:
    """
    Upgrade NESTED pixel ranges from `nside` to `nside_upgrade`.

    Semantics match `hpgeom.upgrade_pixel_ranges`, where half-open ranges
    [start, stop) scale by `4**levels`.
    """
    _validate_nside(nside)
    _validate_nside(nside_upgrade)
    if nside_upgrade < nside:
        raise ValueError("The value for nside_upgrade must be >= nside.")
    ranges = torch.as_tensor(pixel_ranges, dtype=torch.int64)
    if ranges.ndim != 2 or ranges.shape[1] != 2:
        raise ValueError("pixel_ranges must have shape (N, 2)")
    if ranges.numel() == 0:
        return torch.empty((0, 2), dtype=torch.int64, device=ranges.device)
    if nside_upgrade == nside:
        return ranges.clone()

    levels = nside2order(nside_upgrade) - nside2order(nside)
    factor = 1 << (2 * levels)
    return ranges * factor


def query_box(
    nside: int,
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
    *,
    nest: bool = False,
    inclusive: bool = False,
    chunk_size: int = 1_000_000,
) -> Tensor:
    """
    Return pixels whose centers lie in lon/lat box (degrees).
    Handles longitude wrap-around.
    """
    _validate_nside(nside)
    lon_min = float(lon_min)
    lon_max = float(lon_max)
    lat_lo = min(float(lat_min), float(lat_max))
    lat_hi = max(float(lat_min), float(lat_max))

    if inclusive:
        pad_deg = max_pixel_radius(nside, degrees=True)
        lat_lo -= pad_deg
        lat_hi += pad_deg
        lon_min -= pad_deg
        lon_max += pad_deg
    lat_lo = max(-90.0, lat_lo)
    lat_hi = min(90.0, lat_hi)

    lon_min = lon_min % 360.0
    lon_max = lon_max % 360.0
    wraps = lon_min > lon_max

    npix = nside2npix(nside)
    selected: list[Tensor] = []
    for start, end in _pixel_range_chunks(npix, chunk_size=chunk_size):
        pix = torch.arange(start, end, dtype=torch.int64)
        lon, lat = pix2ang(nside, pix, nest=nest, lonlat=True)
        lat_keep = (lat >= lat_lo) & (lat <= lat_hi)
        if wraps:
            lon_keep = (lon >= lon_min) | (lon <= lon_max)
        else:
            lon_keep = (lon >= lon_min) & (lon <= lon_max)
        keep = lat_keep & lon_keep
        if torch.any(keep):
            selected.append(pix[keep])
    if not selected:
        return torch.empty((0,), dtype=torch.int64)
    return torch.cat(selected, dim=0)


def reorder(m: Tensor, *, r2n: bool = False, n2r: bool = False) -> Tensor:
    """Reorder a HEALPix map between RING and NESTED layouts along axis 0."""
    if r2n == n2r:
        raise ValueError("exactly one of r2n/n2r must be True")
    t = torch.as_tensor(m)
    if t.ndim == 0:
        raise ValueError("map must have at least 1 dimension")
    npix = int(t.shape[0])
    nside = npix2nside(npix)
    idx = torch.arange(npix, dtype=torch.int64, device=t.device)
    perm = ring2nest(nside, idx) if r2n else nest2ring(nside, idx)
    out = torch.empty_like(t)
    out[perm] = t
    return out


def get_nside(m: Tensor | Sequence[float] | Sequence[Sequence[float]]) -> int:
    """Return NSIDE from a map (shape `(npix,)` or `(nmaps, npix)`)."""
    t = torch.as_tensor(m)
    if t.ndim == 1:
        npix = int(t.shape[0])
    elif t.ndim == 2:
        npix = int(t.shape[1])
    else:
        raise ValueError("map must be 1D or 2D")
    return npix2nside(npix)


def get_map_size(m: Tensor | Sequence[float] | Sequence[Sequence[float]]) -> int:
    """Return map size (`npix`) for a map input."""
    return nside2npix(get_nside(m))


def mask_bad(
    m: Tensor | float,
    badval: float = UNSEEN,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-8,
) -> Tensor:
    """Boolean mask for bad (UNSEEN/non-finite) map values."""
    t = torch.as_tensor(m)
    if not t.is_floating_point():
        return torch.zeros_like(t, dtype=torch.bool)
    bad = torch.tensor(float(badval), dtype=t.dtype, device=t.device)
    return (~torch.isfinite(t)) | torch.isclose(t, bad, rtol=rtol, atol=atol)


def _normalize_order(order: str | None) -> str:
    if order is None:
        raise ValueError("order must be provided")
    s = str(order).upper()
    if s.startswith("RING"):
        return "RING"
    if s.startswith("NEST"):
        return "NEST"
    raise ValueError("order must be 'RING' or 'NEST'")


def _resolve_dtype(dtype: object | None, fallback: torch.dtype) -> torch.dtype:
    if dtype is None:
        return fallback
    if isinstance(dtype, torch.dtype):
        return dtype
    name = str(dtype).lower()
    if "float16" in name:
        return torch.float16
    if "float32" in name:
        return torch.float32
    if "float64" in name or "double" in name:
        return torch.float64
    if "int16" in name:
        return torch.int16
    if "int32" in name:
        return torch.int32
    if "int64" in name or "int" in name:
        return torch.int64
    return fallback


def _ud_grade_core_nested(
    m_nest: Tensor,
    nside_out: int,
    *,
    pess: bool = False,
    power: float | None = None,
    badval: float = UNSEEN,
) -> Tensor:
    nside_in = npix2nside(int(m_nest.numel()))
    npix_in = int(m_nest.numel())
    npix_out = nside2npix(nside_out)

    if power is not None:
        ratio = (float(nside_out) / float(nside_in)) ** float(power)
    else:
        ratio = 1.0

    if nside_out > nside_in:
        if (nside_out % nside_in) != 0:
            raise ValueError(
                "nside_out must be an integer multiple of nside_in for upgrade"
            )
        rat2 = npix_out // npix_in
        out = m_nest.reshape(npix_in, 1).expand(npix_in, rat2).reshape(npix_out)
        return out * ratio

    if nside_out < nside_in:
        if (nside_in % nside_out) != 0:
            raise ValueError(
                "nside_in must be an integer multiple of nside_out for degrade"
            )
        rat2 = npix_in // npix_out
        mr = m_nest.reshape(npix_out, rat2)
        goods = (~mask_bad(mr, badval=badval)) & torch.isfinite(mr)
        # Match healpy semantics: use multiplication by 0/1 mask instead of
        # `where`, so masked NaN/Inf may still propagate as NaN via `x * 0`.
        summed = torch.sum(mr * goods.to(dtype=mr.dtype), dim=1)
        nhit = goods.sum(dim=1)
        badout = (nhit != rat2) if pess else (nhit == 0)

        nhit_f = nhit.to(dtype=summed.dtype)
        if power is not None:
            nhit_f = nhit_f / ratio
        out = torch.zeros_like(summed)
        nz = nhit_f != 0
        out[nz] = summed[nz] / nhit_f[nz]
        if out.is_floating_point():
            out = out.clone()
            out[badout] = float(badval)
        return out

    return m_nest


def ud_grade(
    map_in: Tensor | Sequence[float] | Sequence[Sequence[float]],
    nside_out: int,
    pess: bool = False,
    badval: float = UNSEEN,
    order_in: str = "RING",
    order_out: str | None = None,
    power: float | None = None,
    dtype: object | None = None,
) -> Tensor:
    """
    Upgrade or degrade resolution of HEALPix map(s), healpy-compatible shape semantics.

    Supports map input shapes `(npix,)` and `(nmaps, npix)`.
    """
    _validate_nside(nside_out)
    in_order = _normalize_order(order_in)
    out_order = in_order if order_out is None else _normalize_order(order_out)

    t = torch.as_tensor(map_in)
    single = t.ndim == 1
    if single:
        maps = t.unsqueeze(0)
    elif t.ndim == 2:
        maps = t
    else:
        raise ValueError("map_in must be 1D or 2D")

    out_dtype = _resolve_dtype(dtype, maps.dtype)
    out_maps: list[Tensor] = []
    for i in range(maps.shape[0]):
        m = maps[i]
        npix = int(m.numel())
        _ = npix2nside(npix)  # validate map length
        m_work = reorder(m, r2n=True) if in_order == "RING" else m
        mout = _ud_grade_core_nested(
            m_work, nside_out, pess=pess, power=power, badval=badval
        )
        if out_order == "RING":
            mout = reorder(mout, n2r=True)
        out_maps.append(mout.to(dtype=out_dtype))

    stacked = torch.stack(out_maps, dim=0)
    return stacked[0] if single else stacked


def lonlat_to_xyz(ra: Tensor, dec: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Convert spherical lon/lat in degrees to unit Cartesian coordinates."""
    ra_t = _as_float64(ra)
    dec_t = _as_float64(dec)
    ra_t, dec_t = torch.broadcast_tensors(ra_t, dec_t)

    lon = torch.deg2rad(ra_t)
    lat = torch.deg2rad(dec_t)
    c = torch.cos(lat)

    x = c * torch.cos(lon)
    y = c * torch.sin(lon)
    z = torch.sin(lat)
    return x, y, z


def xyz_to_lonlat(
    x: Tensor, y: Tensor, z: Tensor, eps: float = 1.0e-15
) -> Tuple[Tensor, Tensor]:
    """Convert Cartesian coordinates to lon/lat in degrees."""
    x_t = _as_float64(x)
    y_t = _as_float64(y)
    z_t = _as_float64(z)
    x_t, y_t, z_t = torch.broadcast_tensors(x_t, y_t, z_t)

    n = torch.sqrt(x_t * x_t + y_t * y_t + z_t * z_t).clamp_min(eps)
    x_n = x_t / n
    y_n = y_t / n
    z_n = z_t / n

    ra = torch.remainder(torch.rad2deg(torch.atan2(y_n, x_n)), 360.0)
    dec = torch.rad2deg(torch.asin(torch.clamp(z_n, -1.0, 1.0)))
    return ra, dec


def angular_distance_deg(
    ra1: Tensor, dec1: Tensor, ra2: Tensor, dec2: Tensor
) -> Tensor:
    """Great-circle angular distance between two lon/lat points in degrees."""
    ra1_t = _as_float64(ra1)
    dec1_t = _as_float64(dec1)
    ra2_t = _as_float64(ra2)
    dec2_t = _as_float64(dec2)
    ra1_t, dec1_t, ra2_t, dec2_t = torch.broadcast_tensors(ra1_t, dec1_t, ra2_t, dec2_t)

    r1 = torch.deg2rad(ra1_t)
    d1 = torch.deg2rad(dec1_t)
    r2 = torch.deg2rad(ra2_t)
    d2 = torch.deg2rad(dec2_t)

    dr = r1 - r2
    dd = d1 - d2
    a = torch.sin(dd * 0.5) ** 2 + torch.cos(d1) * torch.cos(d2) * (
        torch.sin(dr * 0.5) ** 2
    )
    return torch.rad2deg(2.0 * torch.asin(torch.sqrt(torch.clamp(a, 0.0, 1.0))))


def spherical_fourier_features(
    ra: Tensor,
    dec: Tensor,
    num_frequencies: int = 6,
    min_frequency: float = 1.0,
    base: float = 2.0,
    include_xyz: bool = True,
    include_lonlat: bool = True,
) -> Tensor:
    """
    Build multi-scale periodic features for spherical positional encoders.

    Output shape is `ra.shape + (D,)`, where:
    - xyz block contributes 3 channels if `include_xyz=True`
    - each frequency contributes 12 channels for xyz sin/cos (if enabled)
    - each frequency contributes 4 channels for lon/lat sin/cos (if enabled)
    """
    if num_frequencies <= 0:
        raise ValueError("num_frequencies must be positive")
    if min_frequency <= 0.0:
        raise ValueError("min_frequency must be positive")
    if base <= 0.0:
        raise ValueError("base must be positive")

    ra_t = _as_float64(ra)
    dec_t = _as_float64(dec)
    ra_t, dec_t = torch.broadcast_tensors(ra_t, dec_t)

    lon = torch.deg2rad(ra_t)
    lat = torch.deg2rad(dec_t)
    x, y, z = lonlat_to_xyz(ra_t, dec_t)

    exponents = torch.arange(num_frequencies, dtype=ra_t.dtype, device=ra_t.device)
    freqs = min_frequency * (base**exponents)

    feats = []
    if include_xyz:
        feats.extend([x, y, z])

    for f in freqs:
        if include_xyz:
            feats.extend(
                [
                    torch.sin(f * x),
                    torch.cos(f * x),
                    torch.sin(f * y),
                    torch.cos(f * y),
                    torch.sin(f * z),
                    torch.cos(f * z),
                ]
            )
        if include_lonlat:
            feats.extend(
                [
                    torch.sin(f * lon),
                    torch.cos(f * lon),
                    torch.sin(f * lat),
                    torch.cos(f * lat),
                ]
            )

    if not feats:
        raise ValueError("at least one of include_xyz/include_lonlat must be True")
    return torch.stack(feats, dim=-1)


def nest_parent(pix_nest: Tensor, levels: int = 1) -> Tensor:
    """Return NESTED parent index after reducing resolution by `levels`."""
    if levels <= 0:
        raise ValueError("levels must be positive")
    pix_nest_t = _as_int64(pix_nest)
    return pix_nest_t >> (2 * levels)


def nest_children(pix_nest: Tensor, levels: int = 1) -> Tensor:
    """Return all NESTED children after increasing resolution by `levels`."""
    if levels <= 0:
        raise ValueError("levels must be positive")
    pix_nest_t = _as_int64(pix_nest)
    n_children = 1 << (2 * levels)
    offsets = torch.arange(n_children, dtype=torch.int64, device=pix_nest_t.device)
    return (pix_nest_t << (2 * levels)).unsqueeze(-1) + offsets
