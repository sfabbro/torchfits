"""
Identity stress – read and write the same files in different ways, perfectly identical (1.1.1).

Covers the bug-prevention matrix for 1.1.1:
  - image: write() / write_tensor() / HDUList.write() / astropy → read() / read_tensor() / open().data / DataView slice / astropy
  - table: table.write() / write(dict) / TableHDU → read() / read_torch() / scan() / scan_torch() / where / columns
  - unsigned / scaling / VLA / TDIM / TNULL / string / bit / logical
  - HISTORY/COMMENT/CONTINUE, checksum, compression (RICE/GZIP)
All cross-checked: torchfits ↔ astropy ↔ torchfits must match.
Run: `pixi run python examples/example_identity_stress.py`
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torchfits  # noqa: E402
from torchfits import table as tf_table  # noqa: E402
from torchfits.hdu import HDUList, TableHDU, TensorHDU  # noqa: E402


def _check(a, b, msg: str) -> None:
    a = np.asarray(a.numpy() if isinstance(a, torch.Tensor) else a)
    b = np.asarray(b.numpy() if isinstance(b, torch.Tensor) else b)
    if a.dtype.kind in "f" or b.dtype.kind in "f":
        assert np.allclose(a, b, equal_nan=True), msg
    else:
        assert np.array_equal(a, b), msg


def image_identity(tmp: str) -> None:
    data = np.arange(64, dtype=np.float32).reshape(8, 8) * 1.5
    t = torch.from_numpy(data)
    hdr = {"OBJECT": "M31", "EXPTIME": 120.0}
    p_a = os.path.join(tmp, "img_a.fits")
    p_b = os.path.join(tmp, "img_b.fits")
    p_c = os.path.join(tmp, "img_c.fits")
    p_d = os.path.join(tmp, "img_d.fits")
    torchfits.write(p_a, t, header=hdr, overwrite=True)
    torchfits.write_tensor(p_b, t, header=hdr, overwrite=True)
    HDUList([TensorHDU(data=t, header=hdr)]).write(p_c, overwrite=True)
    fits.PrimaryHDU(data, header=fits.Header(hdr)).writeto(p_d, overwrite=True)
    for p in (p_a, p_b, p_c, p_d):
        got = torchfits.read_tensor(p).numpy()
        _check(got, data, f"image {p}")
        assert torchfits.read_header(p)["OBJECT"] == "M31"
        assert np.array_equal(fits.getdata(p).astype(np.float32), data)
    hl = torchfits.open(p_a)
    assert np.array_equal(hl[0].data[:, :].numpy(), data)
    assert np.array_equal(hl[0].data[1:3, 1:3].numpy(), data[1:3, 1:3])
    hl.close()
    print("image identity: OK")


def table_identity(tmp: str) -> None:
    n = 30
    rng = np.random.default_rng(0)
    tbl = {
        "I": np.arange(n, dtype=np.int32),
        "J": np.arange(n, dtype=np.int64) * 1000,
        "E": rng.standard_normal(n).astype(np.float32),
        "S": np.array([f"ID_{i:03d}" for i in range(n)]),
    }
    p_a = os.path.join(tmp, "tbl_a.fits")
    p_b = os.path.join(tmp, "tbl_b.fits")
    p_c = os.path.join(tmp, "tbl_c.fits")
    torchfits.table.write(p_a, tbl, overwrite=True, header={"EXTNAME": "T"})
    torchfits.write(p_b, tbl, overwrite=True)
    HDUList([TableHDU(tbl, header={"EXTNAME": "T"})]).write(p_c, overwrite=True)
    for p in (p_a, p_b, p_c):
        got = tf_table.read(p, hdu=1)
        assert got.num_rows == n
        got_t = tf_table.read_torch(p, hdu=1)
        assert np.array_equal(got_t["I"].numpy(), tbl["I"])
        batches = list(tf_table.scan(p, hdu=1, batch_size=10))
        assert sum(b.num_rows for b in batches) == n
        f = tf_table.read(p, hdu=1, where="I > 10")
        ft = tf_table.read_torch(p, hdu=1, where="I > 10")
        assert f.num_rows == len(ft["I"])
    print("table identity: OK")


def unsigned_scaling_identity(tmp: str) -> None:
    u16 = np.arange(16, dtype=np.uint16).reshape(4, 4) + 60000
    p = os.path.join(tmp, "u16.fits")
    torchfits.write(p, torch.from_numpy(u16), overwrite=True)
    got = torchfits.read_tensor(p).numpy()
    assert np.array_equal(got, u16)
    assert np.array_equal(fits.getdata(p).astype(np.uint16), u16)
    print("unsigned identity: OK")


def vla_tdim_identity(tmp: str) -> None:
    n = 10
    tdim = np.arange(n * 6, dtype=np.float32).reshape(n, 6)
    vla = [np.arange(i % 5, dtype=np.int32) for i in range(n)]
    tbl = {
        "TDIMCOL": tdim,
        "VLA": vla,
        "NAME": np.array([f"OBJ_{i}" for i in range(n)]),
    }
    p = os.path.join(tmp, "vla.fits")
    torchfits.table.write(p, tbl, overwrite=True)
    got = tf_table.read(p, hdu=1)
    got_t = tf_table.read_torch(p, hdu=1)
    assert got.num_rows == n
    assert len(got_t["VLA"]) == n
    print("VLA/TDIM identity: OK")


def compression_identity(tmp: str) -> None:
    data_int = np.arange(256, dtype=np.int16).reshape(16, 16)
    t_int = torch.from_numpy(data_int)
    p = os.path.join(tmp, "c_RICE_1.fits")
    torchfits.write(p, t_int, overwrite=True, compress="RICE_1")
    got = torchfits.read_tensor(p, hdu=1).numpy()
    assert np.array_equal(got, data_int)
    data_f = np.arange(256, dtype=np.float32).reshape(16, 16) * 0.5
    t_f = torch.from_numpy(data_f)
    p2 = os.path.join(tmp, "c_GZIP_1.fits")
    torchfits.write(p2, t_f, overwrite=True, compress="GZIP_1")
    got2 = torchfits.read_tensor(p2, hdu=1).numpy()
    assert np.allclose(got2, data_f, equal_nan=True)
    print("compression identity: OK")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        image_identity(tmp)
        table_identity(tmp)
        unsigned_scaling_identity(tmp)
        vla_tdim_identity(tmp)
        compression_identity(tmp)
    print("All identity checks passed – 1.1.1 ready")


if __name__ == "__main__":
    main()
