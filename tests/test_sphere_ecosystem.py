import torch
from torchfits.sphere import (
    LonLat,
    UnitVector3d,
    Box,
    Circle,
    Ellipse,
    HealSparseMap,
    SkyMaskPipe,
    MOC,
    nest2uniq,
    uniq2nest,
)


def test_geom_primitives():
    # Coordinates
    ll = LonLat(180, 0)
    uv = ll.to_unit_xyz()
    assert isinstance(uv, UnitVector3d)
    assert uv.x == -1.0

    ll_rev = uv.to_lonlat()
    assert ll_rev.lon == 180.0
    assert ll_rev.lat == 0.0

    # Box
    box = Box(10, 20, -5, 5)
    assert box.contains(15, 0).item()
    assert not box.contains(25, 0).item()

    # Circle (Cap)
    circ = Circle(0, 0, 1.0)
    assert circ.contains(0.5, 0).item()
    assert not circ.contains(2.0, 0).item()

    # Ellipse
    ell = Ellipse(0, 0, 2.0, 1.0, pa_deg=90)
    assert ell.contains(1.5, 0).item()
    assert not ell.contains(0, 1.5).item()


def test_sparse_map():
    nside_cov = 1

    # Create from dense
    dense = torch.zeros(12 * 4**2)
    dense[0:16] = 1.0  # First coverage pixel

    sm = HealSparseMap.convert_healpix_map(dense, nside_cov)
    assert sm.nside_coverage == 1
    assert sm.nside_sparse == 4
    assert sm.coverage_map[0] == 0
    assert (sm.coverage_map[1:] == -1).all()

    # Get values
    vals = sm.get_values(torch.tensor([0, 1, 16]))
    assert vals[0] == 1.0
    assert vals[2] == sm.sentinel


def test_moc():
    # UNIQ conversion
    u = nest2uniq(4, 0)
    ns, p = uniq2nest(u)
    assert ns == 4
    assert p == 0

    moc = MOC(torch.tensor([nest2uniq(4, 0), nest2uniq(4, 1)]))
    assert (
        not moc.contains(0, 0).item()
    )  # Center of pixel 0 is lon=45 if nside=1...
    # Let's check a point we know is in pix 0 for nside=4 nest
    # actually better to just check via contains(lon, lat) logic
    from torchfits.wcs import healpix as hpg

    lon, lat = hpg.pix2ang(4, torch.tensor([0, 1]), nest=True, lonlat=True)
    assert moc.contains(lon, lat).all()


def test_mask_pipe():
    pipe = SkyMaskPipe(nside=4)

    dense1 = torch.zeros(192)
    dense1[0:16] = 1
    m1 = HealSparseMap.convert_healpix_map(dense1, nside_coverage=1, sentinel=0)

    dense2 = torch.zeros(192)
    dense2[8:24] = 1
    m2 = HealSparseMap.convert_healpix_map(dense2, nside_coverage=1, sentinel=0)

    pipe.add_stage("s1", m1)
    pipe.add_stage("s2", m2)

    combined = pipe.combine(operation="and")
    # Intersection of 0:16 and 8:24 is 8:16
    dense_c = combined.to_dense()
    assert (dense_c[0:8] == 0).all()
    assert (dense_c[8:16] == 1).all()
    assert (dense_c[16:] == 0).all()
