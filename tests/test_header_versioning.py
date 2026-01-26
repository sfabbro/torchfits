import sys
import types
from unittest.mock import MagicMock
import torch

# Mock setups must be done before importing torchfits
# This mock setup allows running this test without compiling the C++ extension
if "torchfits.cpp" not in sys.modules:
    frame_mock = MagicMock()
    frame_mock.TensorFrame = object
    sys.modules["torch_frame"] = frame_mock

    cpp_mock = MagicMock()
    cpp_mock.WCS = MagicMock()
    wcs_instance = MagicMock()
    # Mock minimal interface needed for WCS init
    wcs_instance.pc.clone.return_value = torch.eye(2)
    wcs_instance.cdelt.clone.return_value = torch.tensor([-0.01, 0.01])
    wcs_instance.crpix.clone.return_value = torch.tensor([50.0, 50.0])
    wcs_instance.crval.clone.return_value = torch.tensor([0.0, 0.0])
    wcs_instance.naxis = 2
    wcs_instance.ctype = ["RA---TAN", "DEC--TAN"]
    wcs_instance.cunit = ["deg", "deg"]
    wcs_instance.lonpole = 180.0
    wcs_instance.latpole = 0.0
    cpp_mock.WCS.return_value = wcs_instance

    module = types.ModuleType("torchfits.cpp")
    module.WCS = cpp_mock.WCS
    sys.modules["torchfits.cpp"] = module

from torchfits.hdu import Header, TensorHDU


def test_header_versioning():
    h = Header()
    assert h._version == 0

    h["a"] = 1
    assert h._version == 1

    h.update({"b": 2})
    assert h._version == 2

    h.setdefault("c", 3)
    assert h._version == 3

    # setdefault existing
    h.setdefault("a", 10)
    assert h._version == 4  # We decided to increment anyway

    val = h.pop("a")
    assert val == 1
    assert h._version == 5

    del h["b"]
    assert h._version == 6

    h.clear()
    assert h._version == 7


def test_tensorhdu_wcs_caching():
    h = Header()
    h["NAXIS"] = 2
    h["CTYPE1"] = "RA---TAN"
    h["CTYPE2"] = "DEC--TAN"

    hdu = TensorHDU(header=h)

    # First access
    wcs1 = hdu.wcs
    assert wcs1 is not None

    # Second access - cached
    wcs2 = hdu.wcs
    assert wcs1 is wcs2

    # Modify header
    h["CRPIX1"] = 100

    # Third access - invalidated
    wcs3 = hdu.wcs
    assert wcs3 is not wcs1

    # Fourth access - cached again
    wcs4 = hdu.wcs
    assert wcs4 is wcs3


def test_plain_dict_header():
    # If header is a plain dict, caching should be disabled (fallback)
    h_dict = {"NAXIS": 2, "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN"}

    # TensorHDU accepts dict as header
    hdu = TensorHDU(header=h_dict)

    wcs1 = hdu.wcs
    wcs2 = hdu.wcs

    # Since plain dict doesn't have _version, caching is disabled.
    assert wcs1 is not wcs2
