
import sys
import os
import types
from unittest.mock import MagicMock

# Add src to path
sys.path.insert(0, os.path.abspath("src"))

# Mock setups must be done before importing torchfits
def setup_mocks():
    if "torchfits.cpp" in sys.modules:
        return

    def mock_module(name):
        m = types.ModuleType(name)
        sys.modules[name] = m
        return m

    # Numpy
    if "numpy" not in sys.modules:
        numpy = mock_module("numpy")
        numpy.uint8 = MagicMock()
        numpy.int16 = MagicMock()
        numpy.int32 = MagicMock()
        numpy.int64 = MagicMock()
        numpy.float32 = MagicMock()
        numpy.float64 = MagicMock()
        numpy.integer = int
        numpy.dtype = MagicMock()
        numpy.ndarray = MagicMock()

    # Torch
    if "torch" not in sys.modules:
        torch = mock_module("torch")
        torch.Tensor = MagicMock()
        torch.empty = MagicMock()
        torch.dtype = MagicMock()
        torch.float32 = MagicMock()
        torch.float64 = MagicMock()
        torch.int32 = MagicMock()
        torch.int16 = MagicMock()
        torch.int64 = MagicMock()
        torch.uint8 = MagicMock()
        torch.long = MagicMock()
        torch.device = MagicMock()
        torch.cuda = MagicMock()
        torch.cuda.is_available.return_value = False

    # Torch Utils
    if "torch.utils" not in sys.modules:
        mock_module("torch.utils")

    if "torch.utils.data" not in sys.modules:
        torch_utils_data = mock_module("torch.utils.data")
        torch_utils_data.DataLoader = MagicMock()
        torch_utils_data.Dataset = MagicMock()
        torch_utils_data.IterableDataset = MagicMock()
        torch_utils_data.Sampler = MagicMock()
        torch_utils_data.DistributedSampler = MagicMock()

    if "torch.utils.data.distributed" not in sys.modules:
        torch_utils_data_dist = mock_module("torch.utils.data.distributed")
        torch_utils_data_dist.DistributedSampler = MagicMock()

    # Torch Frame
    if "torch_frame" not in sys.modules:
        frame_mock = mock_module("torch_frame")
        class MockTensorFrame:
            def __init__(self, *args, **kwargs): pass
        frame_mock.TensorFrame = MockTensorFrame

    # Psutil
    if "psutil" not in sys.modules:
        sys.modules["psutil"] = MagicMock()

    # Torchfits CPP
    if "torchfits.cpp" not in sys.modules:
        sys.modules["torchfits.cpp"] = MagicMock()

    # Avoid loading dataloader which causes typing issues with mocks
    if "torchfits.dataloader" not in sys.modules:
        sys.modules["torchfits.dataloader"] = MagicMock()

    if "torchfits.transforms" not in sys.modules:
        sys.modules["torchfits.transforms"] = MagicMock()

    if "torchfits.core" not in sys.modules:
        sys.modules["torchfits.core"] = MagicMock()

setup_mocks()

# Now import
from torchfits.hdu import Header, HDUList, TensorHDU  # noqa: E402

def test_header_repr_html():
    cards = [
        ("SIMPLE", True, "conform to FITS standard"),
        ("BITPIX", -32, "array data type"),
        ("COMMENT", "", "This is a comment"),
        ("HISTORY", "", "Created by torchfits"),
        ("SPECIAL", "<script>alert('xss')</script>", "malicious comment"),
    ]
    h = Header(cards)

    html_out = h._repr_html_()

    # Basic structure
    assert "<table" in html_out
    assert "<thead" in html_out
    assert "<tbody" in html_out
    assert "FITS Header" in html_out
    assert "5 cards" in html_out  # summary count

    # Accessibility
    assert "scope='col'" in html_out

    # Content
    assert "SIMPLE" in html_out
    assert "BITPIX" in html_out
    assert "conform to FITS standard" in html_out

    # Escaping
    assert "&lt;script&gt;" in html_out
    assert "<script>" not in html_out

def test_hdulist_repr_html():
    h = Header([("EXTNAME", "PRIMARY")])
    hdu = TensorHDU(header=h)
    hdul = HDUList([hdu])

    html_out = hdul._repr_html_()

    assert "<table" in html_out
    assert "scope='col'" in html_out
    assert "PRIMARY" in html_out
