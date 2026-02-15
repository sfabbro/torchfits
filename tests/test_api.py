"""
Test the main torchfits API functions.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torchfits


class TestMainAPI:
    """Test main API functions."""

    def create_test_fits(self, shape=(100, 100), dtype=np.float32):
        """Create a test FITS file."""
        data = np.random.normal(0, 1, shape).astype(dtype)

        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            from astropy.io import fits

            hdu = fits.PrimaryHDU(data)
            hdu.writeto(f.name, overwrite=True)
            return f.name, data

    def create_test_fits_with_named_ext(self, shape=(64, 64), dtype=np.float32, extname="SCI"):
        """Create a FITS file with a named image extension."""
        data = np.random.normal(0, 1, shape).astype(dtype)
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            from astropy.io import fits

            phdu = fits.PrimaryHDU()
            ehdu = fits.ImageHDU(data, name=extname)
            fits.HDUList([phdu, ehdu]).writeto(f.name, overwrite=True)
            return f.name, data, extname

    def create_test_mef(self):
        """Create a FITS file with multiple named image extensions."""
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            from astropy.io import fits

            d1 = np.random.normal(0, 1, (32, 32)).astype(np.float32)
            d2 = np.random.normal(0, 1, (32, 32)).astype(np.float32)
            d3 = np.random.normal(0, 1, (32, 32)).astype(np.float32)
            hdus = [
                fits.PrimaryHDU(d1),
                fits.ImageHDU(d2, name="SCI"),
                fits.ImageHDU(d3, name="ERR"),
            ]
            fits.HDUList(hdus).writeto(f.name, overwrite=True)
            return f.name, d1, d2, d3

    def test_read_basic(self):
        """Test basic read functionality."""
        filepath, expected_data = self.create_test_fits()

        try:
            result, header = torchfits.read(filepath, return_header=True)
            assert isinstance(result, torch.Tensor)
            assert result.shape == expected_data.shape
            np.testing.assert_allclose(result.numpy(), expected_data, rtol=1e-5)
        finally:
            os.unlink(filepath)

    def test_read_image_basic(self):
        """read_image should expose deterministic low-level image reads."""
        filepath, expected_data = self.create_test_fits()

        try:
            result, header = torchfits.read_image(filepath, return_header=True, mmap=False)
            assert isinstance(result, torch.Tensor)
            assert result.shape == expected_data.shape
            assert result.device.type == "cpu"
            assert header.get("SIMPLE") in (True, "T")
            np.testing.assert_allclose(result.numpy(), expected_data, rtol=1e-5)
        finally:
            os.unlink(filepath)

    def test_read_image_matches_read(self):
        """read_image and default read should agree numerically."""
        filepath, _ = self.create_test_fits(shape=(128, 128), dtype=np.float64)

        try:
            default = torchfits.read(filepath, hdu=0, mmap=False)
            image = torchfits.read_image(filepath, hdu=0, mmap=False)
            torch.testing.assert_close(default, image)
        finally:
            os.unlink(filepath)

    def test_read_image_rejects_auto_hdu(self):
        """Specialized image API should require explicit HDU index."""
        filepath, _ = self.create_test_fits()
        try:
            with pytest.raises(ValueError, match="non-negative integer"):
                torchfits.read_image(filepath, hdu="auto", mmap=False)  # type: ignore[arg-type]
        finally:
            os.unlink(filepath)

    def test_read_hdus_direct(self):
        """read_hdus should use one-handle image reads for int/name HDUs."""
        filepath, d0, d1, d2 = self.create_test_mef()
        try:
            out = torchfits.read_hdus(filepath, [0, "SCI", 2], mmap=False)
            assert len(out) == 3
            np.testing.assert_allclose(out[0].numpy(), d0, rtol=1e-5)
            np.testing.assert_allclose(out[1].numpy(), d1, rtol=1e-5)
            np.testing.assert_allclose(out[2].numpy(), d2, rtol=1e-5)

            out2, headers = torchfits.read_hdus(
                filepath, ["SCI", "ERR"], mmap=False, return_header=True
            )
            assert len(out2) == 2
            assert headers[0].get("EXTNAME") == "SCI"
            assert headers[1].get("EXTNAME") == "ERR"
        finally:
            os.unlink(filepath)

    def test_read_image_handle_cache_flag(self):
        """read_image should accept explicit handle-cache control."""
        filepath, _ = self.create_test_fits(shape=(96, 96), dtype=np.float32)
        try:
            a = torchfits.read_image(filepath, hdu=0, mmap=False, handle_cache=True)
            b = torchfits.read_image(filepath, hdu=0, mmap=False, handle_cache=False)
            torch.testing.assert_close(a, b)
            with pytest.raises(ValueError, match="handle_cache must be bool"):
                torchfits.read_image(filepath, hdu=0, mmap=False, handle_cache="yes")  # type: ignore[arg-type]
        finally:
            os.unlink(filepath)

    def test_read_device(self):
        """Test reading to different devices."""
        filepath, expected_data = self.create_test_fits()

        try:
            # CPU
            result_cpu, _ = torchfits.read(filepath, device="cpu", return_header=True)
            assert result_cpu.device.type == "cpu"

            # GPU if available
            if torch.cuda.is_available():
                result_gpu, _ = torchfits.read(
                    filepath, device="cuda", return_header=True
                )
                assert result_gpu.device.type == "cuda"
                torch.testing.assert_close(result_cpu, result_gpu.cpu())
        finally:
            os.unlink(filepath)

    def test_read_mode_image(self):
        """read(mode='image') should force image semantics."""
        filepath, _ = self.create_test_fits(shape=(64, 64), dtype=np.float32)

        try:
            a = torchfits.read(filepath, hdu=0, mode="image", mmap=False)
            b = torchfits.read_image(filepath, hdu=0, mmap=False)
            torch.testing.assert_close(a, b)
        finally:
            os.unlink(filepath)

    def test_read_mode_image_rejects_table_args(self):
        """mode='image' should reject table-specific options."""
        filepath, _ = self.create_test_fits(shape=(64, 64), dtype=np.float32)
        try:
            with pytest.raises(ValueError, match="mode='image'"):
                torchfits.read(filepath, hdu=0, mode="image", columns=["A"])
        finally:
            os.unlink(filepath)

    def test_read_by_extname(self):
        """read should resolve named HDUs via EXTNAME."""
        filepath, expected_data, extname = self.create_test_fits_with_named_ext()
        try:
            result = torchfits.read(filepath, hdu=extname, mmap=False)
            assert isinstance(result, torch.Tensor)
            np.testing.assert_allclose(result.numpy(), expected_data, rtol=1e-5)
        finally:
            os.unlink(filepath)

    def test_read_by_extname_without_handle_cache(self):
        """Named HDU reads should work even when handle caching is disabled."""
        filepath, expected_data, extname = self.create_test_fits_with_named_ext()
        try:
            result = torchfits.read(
                filepath,
                hdu=extname,
                mmap=False,
                handle_cache_capacity=0,
            )
            assert isinstance(result, torch.Tensor)
            np.testing.assert_allclose(result.numpy(), expected_data, rtol=1e-5)
        finally:
            os.unlink(filepath)

    def test_get_header_by_extname(self):
        """get_header should resolve named HDUs via EXTNAME."""
        filepath, _expected_data, extname = self.create_test_fits_with_named_ext(extname="SCI_EXT")
        try:
            hdr = torchfits.get_header(filepath, hdu=extname)
            assert hdr.get("EXTNAME") == extname
        finally:
            os.unlink(filepath)

    def test_read_precision(self):
        """Test mixed precision reading."""
        filepath, expected_data = self.create_test_fits()

        try:
            # FP16
            result_fp16, _ = torchfits.read(filepath, fp16=True, return_header=True)
            assert result_fp16.dtype == torch.float16

            # BF16
            result_bf16, _ = torchfits.read(filepath, bf16=True, return_header=True)
            assert result_bf16.dtype == torch.bfloat16
        finally:
            os.unlink(filepath)

    def test_write_basic(self):
        """Test basic write functionality."""
        data = torch.randn(50, 50)

        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            try:
                torchfits.write(f.name, data, overwrite=True)

                # Read back and verify
                result, _ = torchfits.read(f.name, return_header=True)
                torch.testing.assert_close(result, data)
            finally:
                os.unlink(f.name)

    def test_subset_reader_matches_read_subset(self):
        """SubsetReader should match read_subset for repeated cutouts."""
        filepath, _ = self.create_test_fits(shape=(128, 128), dtype=np.float32)
        try:
            with torchfits.open_subset_reader(filepath, hdu=0, device="cpu") as reader:
                assert reader.hdu == 0
                assert reader.shape == (128, 128)
                a = reader.read_subset(10, 12, 30, 40)
                b = torchfits.read_subset(filepath, 0, 10, 12, 30, 40)
                torch.testing.assert_close(a, b)

                c = reader(0, 0, 16, 16)
                d = torchfits.read_subset(filepath, 0, 0, 0, 16, 16)
                torch.testing.assert_close(c, d)
        finally:
            os.unlink(filepath)

    def test_open_context_manager(self):
        """Test open() as context manager."""
        filepath, expected_data = self.create_test_fits()

        try:
            with torchfits.open(filepath) as hdul:
                assert len(hdul) >= 1
                result = hdul[0].to_tensor()
                assert isinstance(result, torch.Tensor)
        finally:
            os.unlink(filepath)

    def test_batch_operations(self):
        """Test batch reading functions."""
        files = []
        expected_shapes = []

        try:
            # Create multiple test files
            for i in range(3):
                shape = (50 + i * 10, 50 + i * 10)
                filepath, _ = self.create_test_fits(shape)
                files.append(filepath)
                expected_shapes.append(shape)

            # Test batch reading
            results = torchfits.read_batch(files)
            assert len(results) == 3
            for i, result in enumerate(results):
                assert result.shape == expected_shapes[i]

            # Test batch info
            info = torchfits.get_batch_info(files)
            assert info["num_files"] == 3
            assert info["valid_files"] == 3

        finally:
            for f in files:
                if os.path.exists(f):
                    os.unlink(f)

    def test_cache_functions(self):
        """Test cache management functions."""
        # Test cache stats
        stats = torchfits.get_cache_performance()
        assert isinstance(stats, dict)

        # Test cache clearing
        torchfits.clear_file_cache()  # Should not raise

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Non-existent file
        with pytest.raises((FileNotFoundError, RuntimeError)):
            torchfits.read("nonexistent.fits")

        # Invalid HDU
        filepath, _ = self.create_test_fits()
        try:
            with pytest.raises((ValueError, RuntimeError)):
                torchfits.read(filepath, hdu=999)
        finally:
            os.unlink(filepath)

        # Invalid device
        filepath, _ = self.create_test_fits()
        try:
            with pytest.raises(ValueError):
                torchfits.read(filepath, device="invalid")
        finally:
            os.unlink(filepath)

        # Invalid mode
        filepath, _ = self.create_test_fits()
        try:
            with pytest.raises(ValueError):
                torchfits.read(filepath, mode="invalid")
        finally:
            os.unlink(filepath)

    def test_cpp_unmapped_raw_matches_raw_no_mmap(self):
        """read_full_unmapped_raw should match stable raw no-mmap behavior."""
        filepath, expected_data = self.create_test_fits(shape=(64, 64), dtype=np.float64)

        try:
            baseline = torchfits.cpp.read_full_raw(filepath, 0, False)
            # Run multiple times to catch crashy regressions in this code path.
            for _ in range(20):
                result = torchfits.cpp.read_full_unmapped_raw(filepath, 0)
                torch.testing.assert_close(result, baseline)
            np.testing.assert_allclose(baseline.numpy(), expected_data, rtol=1e-12, atol=0.0)
        finally:
            os.unlink(filepath)


class TestTableAPI:
    """Test table-specific API functions."""

    def create_test_table(self, nrows=1000, extname: str | None = None):
        """Create a test FITS table."""
        from astropy.table import Table

        data = {
            "ID": np.arange(nrows),
            "RA": np.random.uniform(0, 360, nrows),
            "DEC": np.random.uniform(-90, 90, nrows),
            "MAG": np.random.normal(20, 2, nrows),
            "FLAG": np.random.choice([0, 1], nrows),
        }

        table = Table(data)
        if extname:
            table.meta["EXTNAME"] = extname

        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            table.write(f.name, format="fits", overwrite=True)
            return f.name

    def test_large_table_reading(self):
        """Test large table reading function."""
        filepath = self.create_test_table(10000)

        try:
            # Test streaming read
            result = torchfits.read_large_table(
                filepath, max_memory_mb=10, streaming=True
            )
            assert isinstance(result, dict)
            assert "RA" in result
            assert len(result["RA"]) == 10000

            # Test non-streaming read
            result2 = torchfits.read_large_table(filepath, streaming=False)
            assert isinstance(result2, dict)

        finally:
            os.unlink(filepath)

    def test_read_table_wrapper(self):
        """read_table should return table columns and reject image HDUs."""
        filepath = self.create_test_table(256)

        try:
            data = torchfits.read_table(filepath, hdu=1)
            assert isinstance(data, dict)
            assert "RA" in data
            assert len(data["RA"]) == 256
        finally:
            os.unlink(filepath)

    def test_read_mode_table(self):
        """read(mode='table') should force table semantics."""
        filepath = self.create_test_table(128)

        try:
            data = torchfits.read(filepath, hdu=1, mode="table")
            assert isinstance(data, dict)
            assert "MAG" in data
            assert len(data["MAG"]) == 128
        finally:
            os.unlink(filepath)

    def test_read_table_default_hdu(self):
        """read_table default HDU should read first table extension (index 1)."""
        filepath = self.create_test_table(64)

        try:
            data = torchfits.read_table(filepath)
            assert isinstance(data, dict)
            assert "ID" in data
            assert len(data["ID"]) == 64
        finally:
            os.unlink(filepath)

    def test_read_table_rows_wrapper(self):
        """read_table_rows should return the requested row slice."""
        filepath = self.create_test_table(512)

        try:
            data = torchfits.read_table_rows(filepath, hdu=1, start_row=11, num_rows=25)
            assert isinstance(data, dict)
            assert "ID" in data
            assert len(data["ID"]) == 25
            assert int(data["ID"][0].item()) == 10
        finally:
            os.unlink(filepath)

    def test_read_table_rows_default_hdu(self):
        """read_table_rows default HDU should read first table extension (index 1)."""
        filepath = self.create_test_table(128)

        try:
            data = torchfits.read_table_rows(filepath, start_row=6, num_rows=10)
            assert "ID" in data
            assert len(data["ID"]) == 10
            assert int(data["ID"][0].item()) == 5
        finally:
            os.unlink(filepath)

    def test_read_table_rejects_auto_hdu(self):
        """Specialized table API should require explicit HDU index."""
        filepath = self.create_test_table(128)

        try:
            with pytest.raises(ValueError, match="non-negative integer"):
                torchfits.read_table(filepath, hdu="auto")  # type: ignore[arg-type]
        finally:
            os.unlink(filepath)

    def test_read_table_rows_rejects_auto_hdu(self):
        """Specialized table row API should require explicit HDU index."""
        filepath = self.create_test_table(128)

        try:
            with pytest.raises(ValueError, match="non-negative integer"):
                torchfits.read_table_rows(  # type: ignore[arg-type]
                    filepath, hdu="auto", start_row=1, num_rows=5
                )
        finally:
            os.unlink(filepath)

    def test_read_table_by_extname(self):
        """read should resolve named table HDUs via EXTNAME."""
        filepath = self.create_test_table(128, extname="CATALOG")
        try:
            data = torchfits.read(filepath, hdu="CATALOG", mode="table")
            assert isinstance(data, dict)
            assert "ID" in data
            assert len(data["ID"]) == 128
        finally:
            os.unlink(filepath)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
