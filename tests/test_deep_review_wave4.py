"""Wave-4 deep-review P2 performance regressions."""

from __future__ import annotations

import struct
from unittest import mock

import pytest
import torch


# --- P2-1: header read is skipped when no consumer needs it -----------------
def test_read_cpp_table_chunk_defers_header_when_read_fails():
    """A numeric row-slice read that fails before any header consumer runs must
    not touch read_header (header is lazy + unsigned dtypes deferred)."""
    from torchfits._table import read as read_mod

    fake_header = mock.Mock(name="read_header")

    def boom(*_a, **_k):
        raise RuntimeError("no reader")

    with mock.patch.object(read_mod, "_acquire_cpp_reader", side_effect=boom):
        with mock.patch("torchfits.read_header", fake_header, create=True):
            result = read_mod._read_cpp_table_chunk(
                "/does/not/exist.fits",
                1,
                None,  # columns
                (1, 5),  # row_slice -> num_rows != -1, short-circuits full path
                None,  # rows
                None,  # where
                False,  # mmap
                False,  # decode_bytes
                "utf-8",
                True,  # strip
                False,  # include_fits_metadata
                False,  # apply_fits_nulls
            )
    assert result is None
    fake_header.assert_not_called()


# --- P2-2: PNG export uses the fast numpy path and stays valid --------------
def test_write_rgb_image_produces_valid_png(tmp_path):
    from torchfits.transforms.rgb import write_rgb_image

    rgb = torch.rand(3, 4, 3)  # H=3, W=4, 3 channels
    out = tmp_path / "preview.png"
    write_rgb_image(str(out), rgb)

    data = out.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    # IHDR width/height live at bytes 16..24 (big-endian).
    width, height = struct.unpack(">II", data[16:24])
    assert (width, height) == (4, 3)


# --- P2-3: SigmaClip still clips correctly without the per-iter restore -----
def test_sigmaclip_clips_outlier():
    from torchfits.transforms.clip import SigmaClip

    x = torch.tensor([[1.0] * 10 + [100.0]])
    clipped = SigmaClip(n_sigma=3.0, max_iter=5, dim=(-1,))(x)
    # The 100.0 outlier is replaced by the mean of the retained ones (~1.0).
    assert torch.allclose(clipped[0, :10], torch.ones(10))
    assert clipped[0, 10].item() < 5.0
    # Deterministic across calls (no leftover buffer state).
    again = SigmaClip(n_sigma=3.0, max_iter=5, dim=(-1,))(x)
    assert torch.equal(clipped, again)


# --- P2-5: stream_table skips get_header when total_rows is supplied --------
def test_stream_table_skips_header_when_total_rows_given(tmp_path):
    from torchfits._io_engine.table_streaming import stream_table

    f = tmp_path / "empty.fits"
    f.write_bytes(b"\x00")  # only needs to exist

    header_fn = mock.Mock(side_effect=AssertionError("header should not be read"))
    # total_rows=0 returns immediately, after the existence + capability checks.
    chunks = list(stream_table(header_fn, str(f), total_rows=0))
    assert chunks == []
    header_fn.assert_not_called()


def test_stream_table_missing_path_raises():
    from torchfits._io_engine.table_streaming import stream_table

    with pytest.raises(FileNotFoundError):
        list(stream_table(lambda *_: None, "/no/such/torchfits_stream.fits"))


def test_default_table_column_rejects_bad_tnull():
    from torchfits._table.mutation import _default_table_column_values

    with pytest.raises(ValueError, match="TNULL"):
        _default_table_column_values("QUAL", "I", 3, tnull="not-an-int")


# --- P2-6: for_environment is memoised but stays correct under mocking ------
def test_for_environment_memoised_and_mock_safe():
    from torchfits.cache import CacheConfig

    CacheConfig._ENV_CACHE.clear()
    with mock.patch.object(CacheConfig, "_is_hpc_environment", return_value=True):
        with mock.patch.object(CacheConfig, "_is_gpu_environment", return_value=False):
            a = CacheConfig.for_environment()
            b = CacheConfig.for_environment()
            assert a is b  # identical env signature -> cached object reused
            assert a.max_files == 1000  # HPC config

    # Changing a detector changes the signature -> fresh, correct result.
    with mock.patch.object(CacheConfig, "_is_hpc_environment", return_value=False):
        with mock.patch.object(
            CacheConfig, "_is_cloud_environment", return_value=False
        ):
            with mock.patch.object(
                CacheConfig, "_is_gpu_environment", return_value=False
            ):
                c = CacheConfig.for_environment()
                assert c.max_files == 100  # default/local config


def test_cache_manager_owns_private_config_copy():
    from torchfits.cache import CacheConfig, CacheManager

    CacheConfig._ENV_CACHE.clear()
    shared = CacheConfig.for_environment()
    mgr = CacheManager()
    mgr.config.max_files = 123456
    # Mutating the manager's config must not corrupt the shared cached one.
    assert shared.max_files != 123456
    assert CacheConfig.for_environment().max_files != 123456


# --- P2-7: the C++ capability cache can be cleared for test reloads ---------
def test_clear_cpp_attr_cache():
    from torchfits._io_engine import _read_pipeline as rp

    dummy = mock.Mock(spec=["present"])
    assert rp._cpp_has(dummy, "present") is True
    assert rp._cpp_has(dummy, "absent") is False
    assert rp._CPP_ATTR_CACHE  # populated
    rp._clear_cpp_attr_cache()
    assert rp._CPP_ATTR_CACHE == {}


# --- P2-10: capability probes swallow only expected I/O errors --------------
def test_probe_returns_empty_on_missing_file():
    from torchfits._table import read as read_mod

    # read_header on a missing path raises OSError, which the probe catches.
    assert read_mod._unsigned_column_dtypes("/does/not/exist.fits", 1, None) == {}
    assert read_mod._column_tforms_for_decode("/does/not/exist.fits", 1, None) == {}
    assert (
        read_mod._can_use_full_read_path(
            "/does/not/exist.fits", 1, None, reject_scaled=False
        )
        is False
    )
