import pytest
from torchfits.core import FITSCore

def test_parse_cutout_spec_valid():
    file_path, hdu_index, slices = FITSCore.parse_cutout_spec("myimage.fits[1][10:20,30:40]")
    assert file_path == "myimage.fits"
    assert hdu_index == 1
    assert slices == (slice(10, 20, None), slice(30, 40, None))

def test_parse_cutout_spec_single_slice():
    file_path, hdu_index, slices = FITSCore.parse_cutout_spec("myimage.fits[0][100:200]")
    assert file_path == "myimage.fits"
    assert hdu_index == 0
    assert slices == (slice(100, 200, None),)

def test_parse_cutout_spec_no_start():
    file_path, hdu_index, slices = FITSCore.parse_cutout_spec("myimage.fits[2][:100,50:]")
    assert file_path == "myimage.fits"
    assert hdu_index == 2
    assert slices == (slice(None, 100, None), slice(50, None, None))

def test_parse_cutout_spec_single_index():
    file_path, hdu_index, slices = FITSCore.parse_cutout_spec("myimage.fits[3][5,10:20]")
    assert file_path == "myimage.fits"
    assert hdu_index == 3
    assert slices == (slice(5, 6, None), slice(10, 20, None))

def test_parse_cutout_spec_invalid():
    with pytest.raises(ValueError):
        FITSCore.parse_cutout_spec("myimage.fits")
    with pytest.raises(ValueError):
        FITSCore.parse_cutout_spec("myimage.fits[1]")
    with pytest.raises(ValueError):
        FITSCore.parse_cutout_spec("myimage.fits[]")
    with pytest.raises(ValueError):
        FITSCore.parse_cutout_spec("myimage.fits[1][,10]")
