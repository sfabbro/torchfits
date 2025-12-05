"""
Test table reading functionality.
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


class TestTableReading:
    """Test FITS table reading functionality."""

    def create_test_table(self, nrows=1000, ncols=5):
        """Create a test FITS table with various data types."""
        from astropy.table import Table

        data = {}
        data["ID"] = np.arange(nrows, dtype=np.int32)
        data["RA"] = np.random.uniform(0, 360, nrows).astype(np.float64)
        data["DEC"] = np.random.uniform(-90, 90, nrows).astype(np.float64)
        data["MAG_G"] = np.random.normal(20, 2, nrows).astype(np.float32)
        data["FLAG"] = np.random.choice([0, 1], nrows).astype(np.uint8)

        # Add extra columns if requested
        for i in range(max(0, ncols - len(data))):
            data[f"EXTRA_{i}"] = np.random.normal(0, 1, nrows).astype(np.float32)

        table = Table(data)

        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            table.write(f.name, format="fits", overwrite=True)
            return f.name, data

    def test_basic_table_reading(self):
        """Test basic table reading."""
        filepath, expected_data = self.create_test_table(100)

        try:
            result, header = torchfits.read(
                filepath, hdu=1
            )  # Tables are usually in HDU 1

            assert isinstance(result, dict)
            assert "RA" in result
            assert "DEC" in result
            assert len(result["RA"]) == 100

        finally:
            os.unlink(filepath)

    def test_column_selection(self):
        """Test reading specific columns."""
        filepath, expected_data = self.create_test_table(100)

        try:
            result, header = torchfits.read(filepath, hdu=1, columns=["RA", "DEC"])

            assert isinstance(result, dict)
            assert "RA" in result
            assert "DEC" in result
            assert "MAG_G" not in result  # Should not be included

        finally:
            os.unlink(filepath)

    def test_row_range_reading(self):
        """Test reading specific row ranges."""
        filepath, expected_data = self.create_test_table(1000)

        try:
            result, header = torchfits.read(filepath, hdu=1, start_row=100, num_rows=50)

            assert isinstance(result, dict)
            assert len(result["RA"]) == 50

        finally:
            os.unlink(filepath)

    def test_large_table_streaming(self):
        """Test streaming read for large tables."""
        filepath, expected_data = self.create_test_table(10000)

        try:
            # Test with memory limit to force streaming
            result = torchfits.read_large_table(
                filepath, max_memory_mb=1, streaming=True
            )

            assert isinstance(result, dict)
            assert "RA" in result
            assert len(result["RA"]) == 10000

        finally:
            os.unlink(filepath)

    def test_data_types(self):
        """Test various FITS data types."""
        filepath, expected_data = self.create_test_table(100)

        try:
            result, header = torchfits.read(filepath, hdu=1)

            # Check data types are preserved appropriately
            assert result["ID"].dtype in [torch.int32, torch.int64]
            assert result["RA"].dtype in [torch.float32, torch.float64]
            assert result["FLAG"].dtype in [torch.uint8, torch.int16, torch.int32]

        finally:
            os.unlink(filepath)

    def test_empty_table(self):
        """Test handling of empty tables."""
        from astropy.table import Table

        # Create empty table
        empty_table = Table()
        empty_table["EMPTY_COL"] = []

        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            try:
                empty_table.write(f.name, format="fits", overwrite=True)

                # Should handle empty table gracefully
                result, header = torchfits.read(f.name, hdu=1)
                assert isinstance(result, dict)

            finally:
                os.unlink(f.name)


class TestTablePerformance:
    """Test table reading performance optimizations."""

    def create_large_table(self, nrows=100000):
        """Create a large test table."""
        from astropy.table import Table

        data = {
            "ID": np.arange(nrows, dtype=np.int64),
            "RA": np.random.uniform(0, 360, nrows).astype(np.float64),
            "DEC": np.random.uniform(-90, 90, nrows).astype(np.float64),
            "MAG_G": np.random.normal(20, 2, nrows).astype(np.float32),
            "MAG_R": np.random.normal(20, 2, nrows).astype(np.float32),
            "MAG_I": np.random.normal(20, 2, nrows).astype(np.float32),
            "FLUX_G": np.random.exponential(1000, nrows).astype(np.float32),
            "FLUX_R": np.random.exponential(1000, nrows).astype(np.float32),
            "FLUX_I": np.random.exponential(1000, nrows).astype(np.float32),
            "Z": np.random.exponential(0.5, nrows).astype(np.float32),
        }

        table = Table(data)

        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            table.write(f.name, format="fits", overwrite=True)
            return f.name

    def test_chunked_reading_performance(self):
        """Test performance of chunked reading."""
        import time

        filepath = self.create_large_table(50000)

        try:
            # Test streaming read
            start_time = time.time()
            result = torchfits.read_large_table(
                filepath, max_memory_mb=10, streaming=True
            )
            streaming_time = time.time() - start_time

            assert isinstance(result, dict)
            assert len(result["RA"]) == 50000

            # Streaming should complete in reasonable time
            assert streaming_time < 30.0  # Should complete within 30 seconds

        finally:
            os.unlink(filepath)

    def test_memory_efficiency(self):
        """Test memory efficiency of table reading."""
        import gc

        import psutil

        filepath = self.create_large_table(20000)

        try:
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024  # MB

            # Read table
            result = torchfits.read_large_table(filepath, max_memory_mb=50)

            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = mem_after - mem_before

            # Should not use excessive memory
            file_size = os.path.getsize(filepath) / 1024 / 1024  # MB
            assert memory_increase < 5 * file_size  # Allow 5x file size for processing

            # Cleanup
            del result
            gc.collect()

        finally:
            os.unlink(filepath)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
