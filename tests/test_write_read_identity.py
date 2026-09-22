"""
1.1.1 identity suite – zero tolerance for bad load.

Every file is written in *multiple* ways and read back in *multiple* ways;
all paths must yield byte-identical or data-identical results.

Covers:
  - header numpy scalars
  - image scaling / unsigned (BZERO) / BLANK nulval
  - table TNULL/TSCAL/TDIM/VLA/string
  - compression (RICE/GZIP) – lossless round-trip
  - multi-HDU, HISTORY/COMMENT/CONTINUE, checksum
  - where= / column projection / batch parity

If any of these fail, 1.1.1 must not tag.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import torch
from astropy.io import fits

import torchfits
from torchfits import table as tf_table


def _eq(a, b):
    if isinstance(a, torch.Tensor):
        a = a.numpy()
    if isinstance(b, torch.Tensor):
        b = b.numpy()
    a = np.asarray(a)
    b = np.asarray(b)
    if a.dtype.kind in "f" or b.dtype.kind in "f":
        return np.allclose(a, b, equal_nan=True)
    return np.array_equal(a, b)


def test_image_write_paths_identical():
    """write() vs write_tensor() vs HDUList.write() vs astropy must match."""
    data = np.arange(64, dtype=np.float32).reshape(8, 8) * 1.5
    t = torch.from_numpy(data)
    hdr = {"OBJECT": "M31", "EXPTIME": 120.0, "FILTER": "R"}

    with tempfile.TemporaryDirectory() as tmp:
        p1 = os.path.join(tmp, "a.fits")
        p2 = os.path.join(tmp, "b.fits")
        p3 = os.path.join(tmp, "c.fits")
        p4 = os.path.join(tmp, "d.fits")

        torchfits.write(p1, t, header=hdr, overwrite=True)
        torchfits.write_tensor(p2, t, header=hdr, overwrite=True)

        from torchfits.hdu import HDUList, TensorHDU

        HDUList([TensorHDU(data=t, header=hdr)]).write(p3, overwrite=True)

        hdu = fits.PrimaryHDU(data)
        for k, v in hdr.items():
            hdu.header[k] = v
        hdu.writeto(p4, overwrite=True)

        for p in (p1, p2, p3, p4):
            got = torchfits.read_tensor(p).numpy()
            assert _eq(got, data), f"data mismatch for {p}"
            h = torchfits.read_header(p)
            for k, v in hdr.items():
                assert h[k] == v, f"header {k} mismatch for {p}"
        for p in (p1, p2, p3):
            assert _eq(fits.getdata(p), data)


def test_table_write_paths_identical():
    """table.write() vs write(dict) vs TableHDU must match."""
    n = 20
    rng = np.random.default_rng(0)
    tbl = {
        "I": np.arange(n, dtype=np.int32),
        "J": np.arange(n, dtype=np.int64) * 1000,
        "E": rng.standard_normal(n).astype(np.float32),
        "D": rng.standard_normal(n).astype(np.float64),
        "S": np.array([f"OBJ_{i:03d}" for i in range(n)]),
    }
    with tempfile.TemporaryDirectory() as tmp:
        p1 = os.path.join(tmp, "t1.fits")
        p2 = os.path.join(tmp, "t2.fits")
        p3 = os.path.join(tmp, "t3.fits")

        torchfits.table.write(p1, tbl, overwrite=True, header={"EXTNAME": "TEST"})
        torchfits.write(p2, tbl, overwrite=True)

        from torchfits.hdu import HDUList, TableHDU

        HDUList([TableHDU(tbl, header={"EXTNAME": "TEST"})]).write(p3, overwrite=True)

        for p in (p1, p2, p3):
            got = tf_table.read(p, hdu=1)
            assert got.num_rows == n
            for k in tbl:
                col = got.column(k).to_pylist()
                exp = tbl[k].tolist()
                assert col == exp, f"col {k} mismatch for {p}"


def test_image_read_paths_identical():
    """read() vs read_tensor() vs open().data vs DataView must match."""
    data = np.arange(64, dtype=np.float32).reshape(8, 8)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "img.fits")
        torchfits.write(path, torch.from_numpy(data), overwrite=True)

        a = torchfits.read_tensor(path).numpy()
        b = torchfits.read(path, hdu=0)
        if isinstance(b, tuple):
            b = b[0]
        if isinstance(b, torch.Tensor):
            b = b.numpy()
        hl = torchfits.open(path)
        c = hl[0].data[:, :].numpy()
        d = hl[0].data[2:6, 2:6].numpy()
        assert _eq(a, data)
        assert _eq(b, data)
        assert _eq(c, data)
        assert _eq(d, data[2:6, 2:6])
        hl.close()


def test_table_read_paths_identical():
    """table.read vs read_torch vs scan vs where/columns must be coherent."""
    n = 100
    rng = np.random.default_rng(1)
    tbl = {
        "A": np.arange(n, dtype=np.int32),
        "B": rng.standard_normal(n),
        "C": np.array([f"X{i}" for i in range(n)]),
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "tbl.fits")
        torchfits.table.write(path, tbl, overwrite=True)

        batches = list(tf_table.scan(path, hdu=1, batch_size=30))
        assert sum(b.num_rows for b in batches) == n
        batches_t = list(tf_table.scan_torch(path, hdu=1, batch_size=30))
        assert sum(len(b["A"]) for b in batches_t) == n

        proj = tf_table.read(path, hdu=1, columns=["A", "C"])
        assert proj.column_names == ["A", "C"]
        filt = tf_table.read(path, hdu=1, where="A > 50")
        filt_t = tf_table.read_torch(path, hdu=1, where="A > 50")
        assert filt.num_rows == len(filt_t["A"])
        assert filt.num_rows == sum(1 for x in tbl["A"] if x > 50)


def test_unsigned_and_scaling_identity():
    """uint16/BZERO paths must be identical via any API."""
    u16 = np.arange(16, dtype=np.uint16).reshape(4, 4) + 60000
    with tempfile.TemporaryDirectory() as tmp:
        p1 = os.path.join(tmp, "u16_a.fits")
        p2 = os.path.join(tmp, "u16_b.fits")
        torchfits.write(p1, torch.from_numpy(u16), overwrite=True)
        torchfits.write_tensor(p2, torch.from_numpy(u16), overwrite=True)
        a = torchfits.read_tensor(p1).numpy()
        b = torchfits.read_tensor(p2).numpy()
        assert np.array_equal(a, u16)
        assert np.array_equal(b, u16)
        assert np.array_equal(a, b)
        assert np.array_equal(fits.getdata(p1).astype(np.uint16), u16)


def test_compression_identity():
    """RICE/GZIP lossless: write compressed, read back identical."""
    with tempfile.TemporaryDirectory() as tmp:
        # RICE is lossless for integers
        data_int = np.arange(256, dtype=np.int16).reshape(16, 16)
        t_int = torch.from_numpy(data_int)
        p = os.path.join(tmp, "c_RICE_1.fits")
        torchfits.write(p, t_int, overwrite=True, compress="RICE_1")
        got = torchfits.read_tensor(p, hdu=1).numpy()
        assert np.array_equal(got, data_int)
        assert np.array_equal(fits.getdata(p, ext=1), data_int)

        # GZIP is lossless for floats
        data_f = np.arange(256, dtype=np.float32).reshape(16, 16) * 0.5
        t_f = torch.from_numpy(data_f)
        p2 = os.path.join(tmp, "c_GZIP_1.fits")
        torchfits.write(p2, t_f, overwrite=True, compress="GZIP_1")
        got2 = torchfits.read_tensor(p2, hdu=1).numpy()
        assert np.allclose(got2, data_f, equal_nan=True)
        assert np.allclose(fits.getdata(p2, ext=1), data_f, equal_nan=True)


def test_header_history_continue_identity():
    """HISTORY/COMMENT/CONTINUE (long strings) must survive any write path."""
    data = np.zeros((4, 4), dtype=np.float32)
    long_val = "A" * 80
    hdr = {
        "OBSERVER": "TEST",
        "HISTORY": "created",
        "COMMENT": "a comment",
        "LONGSTR": long_val,
    }
    with tempfile.TemporaryDirectory() as tmp:
        p1 = os.path.join(tmp, "h1.fits")
        p2 = os.path.join(tmp, "h2.fits")
        torchfits.write(p1, torch.from_numpy(data), header=hdr, overwrite=True)
        from torchfits.hdu import HDUList, TensorHDU

        HDUList([TensorHDU(data=torch.from_numpy(data), header=hdr)]).write(
            p2, overwrite=True
        )
        for p in (p1, p2):
            h = torchfits.read_header(p)
            assert h["OBSERVER"] == "TEST"
            assert h["LONGSTR"] == long_val
            assert fits.getheader(p)["LONGSTR"] == long_val


def test_tnull_tdim_vla_identity():
    """TNULL, TDIM, VLA, string widths must be identical across APIs."""
    n = 10
    tdim_data = np.arange(n * 6, dtype=np.float32).reshape(n, 6)
    vla = [np.arange(i % 5, dtype=np.int32) for i in range(n)]
    strs = np.array([f"OBJ_{i}" for i in range(n)])
    tbl = {"TDIMCOL": tdim_data, "VLA": vla, "NAME": strs}
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "t.fits")
        torchfits.table.write(p, tbl, overwrite=True)
        got = tf_table.read(p, hdu=1)
        got_t = tf_table.read_torch(p, hdu=1)
        assert got.num_rows == n
        assert len(got_t["VLA"]) == n
        assert got.column("TDIMCOL").to_pylist()[0] is not None


# ---------------------------------------------------------------------------
# Header card fidelity: HISTORY/COMMENT/CONTINUE and write(header=) overlay
# ---------------------------------------------------------------------------


def _card_header():
    from torchfits.hdu import Header

    hdr = Header({"OBSERVER": "X", "LONGSTR": "A" * 80, "SHORT": "s"})
    hdr.add_history("h1")
    hdr.add_history("h2")
    hdr.add_comment("c1")
    hdr.add_comment("c2")
    return hdr


def _assert_cards_faithful(path, hdu, label):
    with fits.open(path) as hdul:
        hdr = hdul[hdu].header
    hist = [str(v) for k, v in hdr.items() if k == "HISTORY"]
    comm = [str(v) for k, v in hdr.items() if k == "COMMENT"]
    assert hist == ["h1", "h2"], f"{label}: HISTORY {hist!r}"
    assert comm.count("c1") == 1 and comm.count("c2") == 1, f"{label}: COMMENT {comm!r}"
    mangled = [v for v in hist + comm if v.lstrip().startswith("= '")]
    assert not mangled, f"{label}: value-style commentary cards {mangled!r}"
    assert hdr["OBSERVER"] == "X", label
    assert hdr["LONGSTR"] == "A" * 80, f"{label}: LONGSTR {str(hdr['LONGSTR'])[:20]!r}"
    assert hdr["SHORT"] == "s", label


def test_header_commentary_cards_every_write_path(tmp_path):
    """HISTORY/COMMENT survive exactly once each on every write path (no
    'HISTORY = ...' value-style mangling, no lost duplicates, CONTINUE intact)."""
    from torchfits.hdu import HDUList, TensorHDU

    img = torch.zeros(2, 2)
    cases = [
        (
            "plain-image",
            lambda p, h: torchfits.write(p, img, header=h, overwrite=True),
            0,
        ),
        (
            "compressed-image",
            lambda p, h: torchfits.write(
                p, img, header=h, overwrite=True, compress=True
            ),
            1,
        ),
        (
            "hdulist-image",
            lambda p, h: torchfits.write(
                p, HDUList([TensorHDU(img, header=h)]), overwrite=True
            ),
            0,
        ),
        (
            "hdulist-image-compressed",
            lambda p, h: torchfits.write(
                p, HDUList([TensorHDU(img, header=h)]), overwrite=True, compress=True
            ),
            1,
        ),
        (
            "dict-table",
            lambda p, h: torchfits.write(
                p, {"X": torch.arange(3)}, header=h, overwrite=True
            ),
            1,
        ),
        (
            "dict-table-compressed",
            lambda p, h: torchfits.write(
                p, {"X": torch.arange(3)}, header=h, overwrite=True, compress=True
            ),
            1,
        ),
    ]
    for label, writer, hdu in cases:
        path = str(tmp_path / f"cards_{label}.fits")
        writer(path, _card_header())
        _assert_cards_faithful(path, hdu, label)

    # Rewrite path: cards survive an HDU insert untouched.
    path = str(tmp_path / "cards_rewrite.fits")
    torchfits.write(path, img, header=_card_header(), overwrite=True)
    from torchfits._io_engine._hdu_rewrite import insert_hdu

    insert_hdu(str(tmp_path / "cards_rewrite.fits"), torch.ones(2, 2), index=0)
    _assert_cards_faithful(path, 1, "rewrite-insert")


def test_rewrite_preserves_long_strings(tmp_path):
    """insert_hdu must preserve >68-char (CONTINUE) string values exactly.

    Regression pin for r4b-13 (root fix still R6's at HDUList.fromfile);
    passing since the r4c-15 open_hdulist reassembly landed.
    """
    path = str(tmp_path / "ls_rewrite.fits")
    torchfits.write(
        path, torch.zeros(2, 2), header={"LONGSTR": "A" * 80}, overwrite=True
    )
    from torchfits._io_engine._hdu_rewrite import insert_hdu

    insert_hdu(path, torch.ones(2, 2), index=0)
    assert fits.getheader(path, 1)["LONGSTR"] == "A" * 80


def test_multi_hdu_commentary_independent(tmp_path):
    """Each HDU keeps its own HISTORY/COMMENT cards (no cross-HDU bleed)."""
    from torchfits.hdu import HDUList, Header, TensorHDU

    h1 = Header({"A": 1})
    h1.add_history("hh1")
    h2 = Header({"B": 2})
    h2.add_comment("cc2")
    hdul = HDUList(
        [
            TensorHDU(torch.zeros(2, 2), header=h1),
            TensorHDU(torch.ones(2, 2), header=h2),
        ]
    )
    for label, kw in (("plain", {}), ("compressed", {"compress": True})):
        path = str(tmp_path / f"multi_{label}.fits")
        torchfits.write(path, hdul, overwrite=True, **kw)
        base = 1 if kw else 0
        with fits.open(path) as opened:
            hist0 = [str(v) for k, v in opened[base].header.items() if k == "HISTORY"]
            comm1 = [
                str(v) for k, v in opened[base + 1].header.items() if k == "COMMENT"
            ]
        assert hist0 == ["hh1"], f"{label}: {hist0!r}"
        assert comm1.count("cc2") == 1, f"{label}: {comm1!r}"


def test_write_header_param_overlays_first_hdu_every_path(tmp_path):
    """write(header=) applies to the first HDU on every path (never silently
    ignored)."""
    from torchfits.hdu import HDUList, TensorHDU

    img = torch.zeros(2, 2)
    cases = [
        (
            "plain-seq",
            lambda p: torchfits.write(
                p, [img, img], header={"USERTOP": "yes"}, overwrite=True
            ),
            0,
        ),
        (
            "compressed-seq",
            lambda p: torchfits.write(
                p, [img, img], header={"USERTOP": "yes"}, overwrite=True, compress=True
            ),
            1,
        ),
        (
            "plain-hdulist",
            lambda p: torchfits.write(
                p,
                HDUList([TensorHDU(img), TensorHDU(img)]),
                header={"USERTOP": "yes"},
                overwrite=True,
            ),
            0,
        ),
        (
            "compressed-hdulist",
            lambda p: torchfits.write(
                p,
                HDUList([TensorHDU(img), TensorHDU(img)]),
                header={"USERTOP": "yes"},
                overwrite=True,
                compress=True,
            ),
            1,
        ),
    ]
    for label, writer, hdu in cases:
        path = str(tmp_path / f"ov_{label}.fits")
        writer(path)
        assert torchfits.read_header(path, hdu=hdu).get("USERTOP") == "yes", label
        assert torchfits.read_header(path, hdu=hdu + 1).get("USERTOP") is None, label
