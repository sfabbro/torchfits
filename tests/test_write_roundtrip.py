import torch
import torchfits as tf


def test_write_image_roundtrip(tmp_path):
    data = torch.randint(0, 1000, (32, 32), dtype=torch.int16)
    header = {"OBJECT": "UnitTest", "EXPTIME": "123.4"}
    out = tmp_path / "image_rt.fits"
    tf.write(str(out), data, header)
    rt_data, rt_header = tf.read(str(out), hdu=0)
    assert torch.equal(data.to(rt_data.dtype), rt_data)
    assert rt_header.get("OBJECT") == "UnitTest"
    assert float(rt_header.get("EXPTIME")) == 123.4


def test_write_table_roundtrip(tmp_path):
    table = {"RA": torch.linspace(0, 1, 10), "DEC": torch.linspace(-1, 0, 10)}
    out = tmp_path / "table_rt.fits"
    tf.write(str(out), table, {"EXTNAME": "CAT"})
    rt_table = tf.read(str(out), hdu=1, format="table")
    assert set(rt_table.data.keys()) == {"RA", "DEC"}
    assert torch.allclose(rt_table.data["RA"], table["RA"])
    assert torch.allclose(rt_table.data["DEC"], table["DEC"])


def test_append_hdu(tmp_path):
    base = tmp_path / "mef.fits"
    primary = torch.zeros(8, 8)
    tf.write(str(base), primary, {"EXTNAME": "PRIMARY"})
    extra = torch.ones(4, 4)
    tf.append_hdu(str(base), extra, {"EXTNAME": "SCI"})
    f = tf.FITS(str(base))
    assert len(f) >= 2
    d0, _ = tf.read(str(base), hdu=0)
    d1, _ = tf.read(str(base), hdu=1)
    assert d0.shape == (8, 8)
    assert d1.shape == (4, 4)
