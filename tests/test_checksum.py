from __future__ import annotations

import torch

import torchfits


def test_file_checksum_roundtrip_and_corruption(tmp_path):
    path = tmp_path / "chk.fits"
    data = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    torchfits.write(str(path), data, header={"FOO": 1}, overwrite=True)

    # No keywords yet.
    out = torchfits.verify_checksums(str(path), hdu=0)
    assert out["datastatus"] == 0
    assert out["hdustatus"] == 0
    assert out["ok"] is False

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
