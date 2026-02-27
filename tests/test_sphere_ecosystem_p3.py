import torch
import math
import json
from torchfits.sphere import MOC, HealSparseMap, nest2uniq


def test_moc_area():
    # Full sky MOC (order 0, all 12 pixels)
    full_sky = MOC(nest2uniq(1, torch.arange(12)))
    assert math.isclose(full_sky.area, 4.0 * math.pi, rel_tol=1e-9)
    assert math.isclose(full_sky.area_sq_deg, 41252.96, rel_tol=1e-2)


def test_moc_filtering():
    # MOC covering northern hemisphere roughly
    moc = MOC.from_circle(0.0, 90.0, 10.0, max_order=5)
    lons = torch.tensor([0.0, 180.0, 0.0])
    lats = torch.tensor([90.0, 0.0, 85.0])

    mask = moc.contains(lons, lats)
    assert mask[0]
    assert not mask[1]
    assert mask[2]

    lon_f, lat_f = moc.filter_catalog(lons, lats)
    assert lon_f.numel() == 2
    # Ensure comparison is done in float64
    expected_lat = torch.tensor([90.0, 85.0], dtype=torch.float64)
    assert torch.allclose(lat_f.to(torch.float64), expected_lat)


def test_moc_serialization_ascii():
    uniqs = nest2uniq(1, torch.tensor([0, 1, 2]))
    moc = MOC(uniqs)
    ascii_str = moc.to_ascii()
    assert "0/0 1 2" in ascii_str

    moc2 = MOC.from_ascii(ascii_str)
    assert torch.all(moc2.uniq == moc.uniq)


def test_moc_serialization_json():
    uniqs = nest2uniq(1, torch.tensor([0, 1, 2]))
    moc = MOC(uniqs)
    js = moc.to_json()
    data = json.loads(js)
    assert data["0"] == [0, 1, 2]

    # We don't have from_json yet in the plan but let's check internal consistency
    # (MOC constructor handles uniqs)


def test_moc_geom_construction():
    # Circle
    moc_c = MOC.from_circle(0.0, 0.0, 1.0, max_order=6)
    assert moc_c.area > 0
    assert moc_c.contains(0.0, 0.0)

    # Polygon (Square around equator)
    lons = [359.0, 1.0, 1.0, 359.0]
    lats = [-1.0, -1.0, 1.0, 1.0]
    moc_p = MOC.from_polygon(lons, lats, max_order=6)
    assert moc_p.contains(0.0, 0.0)
    assert not moc_p.contains(10.0, 10.0)


def test_sparse_from_moc():
    moc = MOC.from_circle(0.0, 0.0, 5.0, max_order=6)
    smap = HealSparseMap.from_moc(moc, nside=64)
    assert isinstance(smap, HealSparseMap)
    assert smap.get_covered_pixels().numel() > 0
