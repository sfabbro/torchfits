import os
import tempfile

import pytest
import torch

import torchfits


@pytest.mark.skipif(
    os.environ.get("TORCHFITS_ENABLE_REMOTE_TESTS") != "1",
    reason="Remote smoke tests disabled by default; set TORCHFITS_ENABLE_REMOTE_TESTS=1 to enable.",
)
def test_remote_read_smoke_and_cache_roundtrip():
    # Allow overriding the test URL via env; default to a small file in this repo
    url = os.environ.get(
        "TORCHFITS_REMOTE_TEST_URL",
        "https://raw.githubusercontent.com/sfabbro/torchfits/main/examples/basic_example.fits",
    )

    with tempfile.TemporaryDirectory() as td:
        # Isolate cache for the test
        torchfits.configure_cache(cache_dir=td)

        # First read should download and cache the file
        data, header = torchfits.read(url, use_cache=True)
        assert isinstance(data, torch.Tensor)
        assert data.numel() > 0
        assert isinstance(header, dict)

        # Second read should hit the cache
        data2, header2 = torchfits.read(url, use_cache=True)
        if isinstance(data, torch.Tensor) and isinstance(data2, torch.Tensor):
            assert torch.allclose(data, data2)
        # Basic cache stats should reflect at least one cached file and multiple accesses
        stats = torchfits.cache_stats()
        if stats:  # stats may be empty if disabled; don't fail in that case
            assert stats.get("total_files", 0) >= 1
            assert stats.get("total_accesses", 0) >= 2
