import pytest
import torch
from unittest import mock

from torchfits import io

@pytest.fixture
def mock_cpp():
    with mock.patch("torchfits.io.cpp") as m_cpp:
        # Set default return values for the mock methods
        m_cpp.read_full.return_value = torch.zeros((10, 10))
        m_cpp.read_full_raw.return_value = torch.zeros((10, 10))
        m_cpp.read_full_unmapped_raw.return_value = torch.zeros((10, 10))
        m_cpp.read_full_unmapped.return_value = torch.zeros((10, 10))
        m_cpp.read_full_nocache.return_value = torch.zeros((10, 10))
        m_cpp.read_hdus_batch.return_value = [torch.zeros((10, 10)), torch.zeros((10, 10))]
        m_cpp.read_images_batch.return_value = [torch.zeros((10, 10)), torch.zeros((10, 10))]
        # read_full_raw_with_scale returns: data, scaled, bscale, bzero
        m_cpp.read_full_raw_with_scale.return_value = (torch.zeros((10, 10)), False, 1.0, 0.0)
        yield m_cpp

@pytest.fixture
def mock_flags():
    with mock.patch("torchfits.io._HAS_READ_HDUS_BATCH", True), \
         mock.patch("torchfits.io._HAS_READ_FULL_RAW_WITH_SCALE", True), \
         mock.patch("torchfits.io._HAS_READ_FULL_RAW", True), \
         mock.patch("torchfits.io._HAS_READ_FULL_UNMAPPED_RAW", True), \
         mock.patch("torchfits.io._HAS_READ_FULL_UNMAPPED", True), \
         mock.patch("torchfits.io._HAS_READ_FULL_NOCACHE", True):
        yield


class TestIORead:
    """Test the low-level torchfits.io.read function directly with mocked cpp."""

    def test_read_empty_path(self):
        """Test read with empty path raises ValueError."""
        with pytest.raises(ValueError, match="Path must be a non-empty string"):
            io.read("")

    def test_read_invalid_path_type(self):
        """Test read with invalid path type raises ValueError."""
        with pytest.raises(ValueError, match="Path must be a string or list of strings"):
            io.read(123)

    def test_read_batch_path_invalid_element_type(self):
        """Test read with batch paths containing invalid types raises ValueError."""
        with pytest.raises(ValueError, match="Path must be a string or list of strings"):
            io.read(["file1.fits", 123], hdu=0)

    def test_read_batch_path_invalid_hdu(self):
        """Test read with batch paths and invalid hdu raises ValueError."""
        with pytest.raises(ValueError, match="Batch read requires a single integer HDU"):
            io.read(["file1.fits", "file2.fits"], hdu=[0, 1])

    def test_read_invalid_hdu_type(self):
        """Test read with invalid hdu type raises ValueError."""
        with pytest.raises(ValueError, match="HDU index must be a non-negative integer"):
            io.read("file.fits", hdu=-1)

    def test_read_invalid_device(self):
        """Test read with invalid device raises ValueError."""
        with pytest.raises(ValueError, match="device must be 'cpu', 'cuda', 'mps' or 'cuda:N'"):
            io.read("file.fits", device="invalid_device")

    def test_read_basic(self, mock_cpp, mock_flags):
        """Test basic single file read."""
        io.read("file.fits", hdu=0, scale_on_device=False)
        mock_cpp.read_full.assert_called_once_with("file.fits", 0, True)

    def test_read_batch_paths(self, mock_cpp, mock_flags):
        """Test reading a list of paths."""
        io.read(["file1.fits", "file2.fits"], hdu=0)
        mock_cpp.read_images_batch.assert_called_once_with(["file1.fits", "file2.fits"], 0)

    def test_read_batch_hdus(self, mock_cpp, mock_flags):
        """Test reading multiple HDUs from a single path."""
        io.read("file.fits", hdu=[0, 1])
        mock_cpp.read_hdus_batch.assert_called_once_with("file.fits", [0, 1])

    def test_read_fp16(self, mock_cpp, mock_flags):
        """Test fp16 conversion."""
        res = io.read("file.fits", hdu=0, fp16=True, scale_on_device=False)
        assert res.dtype == torch.float16

    def test_read_bf16(self, mock_cpp, mock_flags):
        """Test bf16 conversion."""
        res = io.read("file.fits", hdu=0, bf16=True, scale_on_device=False)
        assert res.dtype == torch.bfloat16

    def test_read_scale_on_device_scaled(self, mock_cpp, mock_flags):
        """Test scale_on_device when cpp says scaled=True."""
        mock_cpp.read_full_raw_with_scale.return_value = (torch.ones((10, 10)), True, 2.0, 1.0)
        res = io.read("file.fits", hdu=0, scale_on_device=True, raw_scale=False)
        mock_cpp.read_full_raw_with_scale.assert_called_once_with("file.fits", 0, True)
        assert res.dtype == torch.float32
        # (1 * 2.0) + 1.0 = 3.0
        assert torch.all(res == 3.0)

    def test_read_scale_on_device_not_scaled(self, mock_cpp, mock_flags):
        """Test scale_on_device when cpp says scaled=False."""
        mock_cpp.read_full_raw_with_scale.return_value = (torch.zeros((10, 10)), False, 1.0, 0.0)
        io.read("file.fits", hdu=0, scale_on_device=True, raw_scale=False)
        mock_cpp.read_full_raw_with_scale.assert_called_once_with("file.fits", 0, True)

    def test_read_use_cache_raw_scale(self, mock_cpp, mock_flags):
        """Test use_cache=True and raw_scale=True."""
        io.read("file.fits", hdu=0, use_cache=True, raw_scale=True, scale_on_device=False)
        mock_cpp.read_full_raw.assert_called_once_with("file.fits", 0, True)

    def test_read_use_cache_no_raw_scale(self, mock_cpp, mock_flags):
        """Test use_cache=True and raw_scale=False."""
        io.read("file.fits", hdu=0, use_cache=True, raw_scale=False, scale_on_device=False)
        mock_cpp.read_full.assert_called_once_with("file.fits", 0, True)

    def test_read_no_cache_raw_scale_nommap(self, mock_cpp, mock_flags):
        """Test use_cache=False, raw_scale=True, mmap=False."""
        io.read("file.fits", hdu=0, use_cache=False, raw_scale=True, mmap=False, scale_on_device=False)
        mock_cpp.read_full_unmapped_raw.assert_called_once_with("file.fits", 0)

    def test_read_no_cache_raw_scale_mmap(self, mock_cpp, mock_flags):
        """Test use_cache=False, raw_scale=True, mmap=True."""
        io.read("file.fits", hdu=0, use_cache=False, raw_scale=True, mmap=True, scale_on_device=False)
        mock_cpp.read_full_raw.assert_called_once_with("file.fits", 0, True)

    def test_read_no_cache_no_raw_scale_nommap(self, mock_cpp, mock_flags):
        """Test use_cache=False, raw_scale=False, mmap=False."""
        io.read("file.fits", hdu=0, use_cache=False, raw_scale=False, mmap=False, scale_on_device=False)
        mock_cpp.read_full_unmapped.assert_called_once_with("file.fits", 0)

    def test_read_no_cache_no_raw_scale_mmap(self, mock_cpp, mock_flags):
        """Test use_cache=False, raw_scale=False, mmap=True."""
        io.read("file.fits", hdu=0, use_cache=False, raw_scale=False, mmap=True, scale_on_device=False)
        mock_cpp.read_full_nocache.assert_called_once_with("file.fits", 0, True)

    def test_read_target_device_conversion(self, mock_cpp, mock_flags):
        """Test moving data to different device if target_dtype is specified."""
        # Note: testing actual device move without CUDA is hard, we can just test dtype
        res = io.read("file.fits", hdu=0, bf16=True, scale_on_device=False)
        assert res.dtype == torch.bfloat16
