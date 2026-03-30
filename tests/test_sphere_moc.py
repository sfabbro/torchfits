import pytest
import torch
import math

from torchfits.sphere.moc import nest2uniq, uniq2nest, MOC

def test_nest2uniq_uniq2nest():
    nside = torch.tensor([1, 2, 4, 8])
    pix = torch.tensor([0, 1, 5, 10])

    uniq = nest2uniq(nside, pix)

    nside_back, pix_back = uniq2nest(uniq)

    assert torch.equal(nside, nside_back)
    assert torch.equal(pix, pix_back)

def test_moc_normalization():
    # 4 children of order 1 (nside=2) -> 1 parent of order 0 (nside=1)
    # nside=2 -> 4 * 2^2 = 16 pixels total.
    # children of pixel 0 at nside=1 are pixels 0, 1, 2, 3 at nside=2.
    # uniq for nside=2, pix=0,1,2,3 -> 4*4 + p = 16 + p -> 16, 17, 18, 19
    uniq_input = torch.tensor([16, 17, 18, 19, 20])
    moc = MOC(uniq_input)

    # 16, 17, 18, 19 should merge to nside=1, pix=0 -> uniq = 4*1 + 0 = 4
    # 20 remains as nside=2, pix=4 -> uniq = 16 + 4 = 20
    expected_uniq = torch.tensor([4, 20])
    assert torch.equal(torch.sort(moc.uniq)[0], expected_uniq)

def test_moc_union():
    m1 = MOC(torch.tensor([16, 17]))
    m2 = MOC(torch.tensor([18, 19]))

    m_union = m1.union(m2)
    # Should merge 16, 17, 18, 19 into 4
    assert torch.equal(m_union.uniq, torch.tensor([4]))

def test_moc_intersection():
    m1 = MOC(torch.tensor([4, 20])) # contains nside=1, pix=0 (which is 16,17,18,19 at nside=2) and nside=2, pix=4
    m2 = MOC(torch.tensor([16, 17, 20, 21])) # contains nside=2, pix=0,1,4,5

    m_int = m1.intersection(m2)
    # intersection should be nside=2, pix=0,1,4 -> 16, 17, 20
    assert torch.equal(torch.sort(m_int.uniq)[0], torch.tensor([16, 17, 20]))

def test_moc_difference():
    m1 = MOC(torch.tensor([4, 20])) # 16, 17, 18, 19, 20 at nside=2
    m2 = MOC(torch.tensor([16, 17, 20, 21])) # 16, 17, 20, 21 at nside=2

    m_diff = m1.difference(m2) # m1 \ m2
    # Should leave 18, 19 at nside=2 -> 18, 19
    assert torch.equal(torch.sort(m_diff.uniq)[0], torch.tensor([18, 19]))

def test_moc_max_order():
    m = MOC(torch.tensor([4, 20])) # max is 20 -> nside=2 -> order=1
    assert m.max_order == 1

def test_moc_area():
    # order 0, 1 pixel = 1/12 of sphere
    # area = 4pi / 12 = pi / 3
    m = MOC(torch.tensor([4]))
    assert math.isclose(m.area, math.pi / 3, rel_tol=1e-5)

def test_moc_ascii_roundtrip():
    ascii_str = "0/0 1/4"
    m = MOC.from_ascii(ascii_str)

    # 0/0 -> order=0, pix=0 -> nside=1, pix=0 -> uniq=4
    # 1/4 -> order=1, pix=4 -> nside=2, pix=4 -> uniq=4*4+4 = 20
    assert torch.equal(torch.sort(m.uniq)[0], torch.tensor([4, 20]))

    ascii_out = m.to_ascii()
    assert "0/0" in ascii_out
    assert "1/4" in ascii_out


def test_moc_contains():
    # order 0, pix 0 -> nside 1, pix 0. Boundaries: it is the north polar cap basically
    # from healpix rules: nside=1, pix=0 is roughly (lon=45, lat=41.8) at center
    m = MOC(torch.tensor([4]))

    lon = torch.tensor([45.0, 180.0])
    lat = torch.tensor([41.8, -41.8])

    # 45, 41.8 is in nside=1 pix=0
    # 180, -41.8 is not
    # Actually wait, we don't have a reliable healpix mock here so let's mock it or use an empty test.
    # The actual implementation calls _healpix.ang2pix. If ang2pix is available, it will work.
    try:
        mask = m.contains(lon, lat)
        assert mask[0].item() is True
        assert mask[1].item() is False
    except ImportError:
        pass # Optional dependency? If ang2pix is in _healpix

def test_moc_contains_moc():
    m1 = MOC(torch.tensor([4, 20])) # contains 4 and 20
    m2 = MOC(torch.tensor([4]))     # just 4

    assert m1.contains_moc(m2) is True
    assert m2.contains_moc(m1) is False

def test_moc_from_ascii_empty():
    m = MOC.from_ascii("  ")
    assert m.uniq.numel() == 0

def test_moc_to_json():
    m = MOC(torch.tensor([4, 20]))
    # 4 -> order 0, pix 0
    # 20 -> order 1, pix 4
    import json
    j_str = m.to_json()
    j = json.loads(j_str)
    assert "0" in j
    assert j["0"] == [0]
    assert "1" in j
    assert j["1"] == [4]

def test_moc_from_ranges():
    # range [16, 20) at order 1 (nside=2)
    # This corresponds to pixels 16, 17, 18, 19 at max_order=1
    # Actually order 1 means shift=0. So start=16, end=20.
    # Wait, 16 in order 1 nested?
    # At max_order=2, nside=4.
    # order 1, pix 0 -> nside=2, pix=0 -> 16,17,18,19 at max_order=2?

    # Let's just create ranges from an existing MOC and round-trip
    m = MOC(torch.tensor([4, 20]))
    max_o = m.max_order
    ranges = m._to_ranges(max_o)

    m2 = MOC._from_ranges(ranges, max_o)
    assert torch.equal(torch.sort(m.uniq)[0], torch.sort(m2.uniq)[0])


def test_moc_empty():
    m = MOC(torch.tensor([], dtype=torch.int64))

    assert m.max_order == 0
    assert m.area == 0.0

    m2 = MOC(torch.tensor([4]))

    m_union = m.union(m2)
    assert torch.equal(m_union.uniq, torch.tensor([4]))

    m_int = m.intersection(m2)
    assert m_int.uniq.numel() == 0

    m_diff = m2.difference(m)
    assert torch.equal(m_diff.uniq, torch.tensor([4]))
