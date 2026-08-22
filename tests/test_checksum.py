from __future__ import annotations

import torch

import torchfits


def test_file_checksum_roundtrip_and_corruption(tmp_path):
    path = tmp_path / "chk.fits"
    data = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    torchfits.write(str(path), data, header={"FOO": 1}, overwrite=True)

    # No keywords yet — ok=True (not corrupt, just nothing to verify).
    out = torchfits.verify_checksums(str(path), hdu=0)
    assert out["datastatus"] == 0
    assert out["hdustatus"] == 0
    assert out["ok"] is True
    assert out["status"] == "no_checksums"

    torchfits.write_checksums(str(path), hdu=0)
    out = torchfits.verify_checksums(str(path), hdu=0)
    assert out["ok"] is True

    # Corrupt an unrelated header keyword: DATASUM should remain valid, CHECKSUM should fail.
    with open(path, "r+b") as f:
        raw = f.read()
        needle = b"FOO     ="
        idx = raw.find(needle)
        assert idx != -1
        card = bytearray(raw[idx : idx + 80])
        one = card.find(b"1")
        assert one != -1
        card[one : one + 1] = b"2"
        f.seek(idx)
        f.write(card)

    out = torchfits.verify_checksums(str(path), hdu=0)
    assert out["datastatus"] == 1
    assert out["hdustatus"] == -1
    assert out["ok"] is False


def test_write_checksum_kwarg_covers_every_path(tmp_path):
    """write(..., checksum=True) stamps DATASUM/CHECKSUM on all HDUs."""
    import torchfits as tf
    from torchfits.hdu import HDUList, TensorHDU

    # Single image.
    p = tmp_path / "img.fits"
    tf.write(str(p), torch.ones(4, 4), overwrite=True, checksum=True)
    assert tf.verify_checksums(str(p), hdu=0)["status"] == "ok"

    # Dict table (extension HDU).
    t = tmp_path / "tab.fits"
    tf.write(str(t), {"x": torch.arange(5)}, overwrite=True, checksum=True)
    assert tf.verify_checksums(str(t), hdu=1)["status"] == "ok"

    # HDUList.
    m = tmp_path / "multi.fits"
    hdul = HDUList([TensorHDU(torch.zeros(3, 3)), TensorHDU(torch.ones(2, 2))])
    tf.write(str(m), hdul, overwrite=True, checksum=True)
    for hdu in range(tf.read_num_hdus(str(m))):
        assert tf.verify_checksums(str(m), hdu=hdu)["status"] == "ok"

    # Overwrite-in-place path (temp + os.replace).
    tf.write(str(m), hdul, overwrite=True, checksum=True)
    assert tf.verify_checksums(str(m), hdu=0)["status"] == "ok"

    # Compressed image and compressed table.
    c = tmp_path / "comp.fits"
    tf.write(
        str(c),
        torch.arange(64, dtype=torch.float32).reshape(8, 8),
        overwrite=True,
        compress="RICE_1",
        checksum=True,
    )
    assert tf.verify_checksums(str(c), hdu=0)["status"] == "ok"
    ct = tmp_path / "comp_table.fits"
    tf.write(
        str(ct),
        {"x": torch.arange(5)},
        overwrite=True,
        compress="RICE_1",
        checksum=True,
    )
    assert tf.verify_checksums(str(ct), hdu=1)["status"] == "ok"

    # Default stays unchanged: no keywords written.
    d = tmp_path / "plain.fits"
    tf.write(str(d), torch.ones(2, 2), overwrite=True)
    assert tf.verify_checksums(str(d), hdu=0)["status"] == "no_checksums"
