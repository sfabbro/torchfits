import torch
import os
from torchfits.sphere import HealSparseMap, MOC, nest2uniq


def test_moc_normalization():
    # 4 children should merge into parent
    # Parent (order 0, pix 0): UNIQ = 4*4^0 + 0 = 4
    # Children (order 1, pix 0-3): UNIQ = 4*4^1 + {0,1,2,3} = 16, 17, 18, 19
    children = torch.tensor([16, 17, 18, 19], dtype=torch.int64)
    moc = MOC(children)
    assert moc.uniq.numel() == 1
    assert moc.uniq[0].item() == 4


def test_moc_ops():
    # MOC A: pixels 0, 1 at nside 4 (NESTED)
    # MOC B: pixels 1, 2 at nside 4 (NESTED)
    u_a = torch.tensor([nest2uniq(4, 0), nest2uniq(4, 1)], dtype=torch.int64)
    u_b = torch.tensor([nest2uniq(4, 1), nest2uniq(4, 2)], dtype=torch.int64)

    moc_a = MOC(u_a)
    moc_b = MOC(u_b)

    # Union
    union = moc_a.union(moc_b)
    assert union.uniq.numel() == 3  # 0, 1, 2

    # Intersection
    inter = moc_a.intersection(moc_b)
    assert inter.uniq.numel() == 1
    assert inter.uniq[0].item() == nest2uniq(4, 1)

    # Difference
    diff = moc_a.difference(moc_b)
    assert diff.uniq.numel() == 1
    assert diff.uniq[0].item() == nest2uniq(4, 0)


def test_sparse_udgrade():
    # Create sparse map at nside 4
    pixels = torch.arange(16)  # All pixels in first coverage pixel
    smap = HealSparseMap.from_pixels(
        pixels, nside_sparse=4, nside_coverage=1, values=torch.ones(16)
    )

    # Downgrade to nside 2
    # pixels 0,1,2,3 at nside 4 -> pixel 0 at nside 2
    # pixels 4,5,6,7 at nside 4 -> pixel 1 at nside 2
    # ...
    smap_down = smap.ud_grade(nside_out=2, reduction="mean")
    assert smap_down.nside_sparse == 2
    assert smap_down.values.numel() == 4  # 16 / 4
    assert (smap_down.values == 1.0).all()

    # Upgrade to nside 8
    smap_up = smap.ud_grade(nside_out=8)
    assert smap_up.nside_sparse == 8
    assert smap_up.values.numel() == 16 * 4  # (8/4)^2 = 4


def test_sparse_fits_io(tmp_path):
    path = str(tmp_path / "test.hs")
    pixels = torch.tensor([0, 10, 100])
    smap = HealSparseMap.from_pixels(
        pixels, nside_sparse=128, nside_coverage=8, values=torch.tensor([1.0, 2.0, 3.0])
    )

    smap.write_fits(path, overwrite=True)
    assert os.path.exists(path)

    smap_read = HealSparseMap.read_fits(path)
    assert smap_read.nside_sparse == smap.nside_sparse
    assert smap_read.nside_coverage == smap.nside_coverage
    assert torch.allclose(smap_read.values, smap.values)
    assert torch.equal(smap_read.coverage_map, smap.coverage_map)
