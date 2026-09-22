"""Read-pipeline contract tests: batch HDU attribution, fallback error typing.

Covers the R4 slice-A contracts (r4a-01, r4a-04): batch C++ results can never
silently misalign against HDU lists, and the fallback decode paths re-raise IO
errors typed instead of swallowing them into slower retries.
"""

from __future__ import annotations

import warnings
from unittest import mock

import pytest
import torch

import torchfits
from torchfits import io
from torchfits._io_engine import _read_pipeline, _read_pipeline_fallback


def test_read_hdus_short_batch_result_raises_naming_path(tmp_path):
    """read_hdus must never pair tensors with the wrong headers (r4a-01)."""
    path = str(tmp_path / "m.fits")
    torchfits.write(path, torch.zeros(4, 4), overwrite=True)

    cpp = mock.Mock()
    cpp.read_hdus_batch.return_value = [torch.zeros(2, 2)]  # 2 hdus, 1 result
    with mock.patch("torchfits._io_engine.image._cpp", cpp):
        with pytest.raises(RuntimeError, match="m.fits"):
            torchfits.read_hdus(path, [0, 0], mmap=True)


def test_read_batch_hdus_short_batch_result_falls_back_per_hdu():
    """read(path, hdu=[...]) must not return fewer tensors than hdus (r4a-01)."""
    cpp = mock.Mock()
    cpp.read_hdus_batch.return_value = [torch.zeros(2, 2)]  # 2 hdus, 1 result
    logger = mock.Mock()
    calls = []

    def fake_unified(**kwargs):
        calls.append(kwargs["hdu"])
        return torch.full((2, 2), float(len(calls)))

    with mock.patch.object(_read_pipeline, "read_unified", side_effect=fake_unified):
        out = _read_pipeline._read_batch_hdus(
            cpp_module=cpp,
            path="a.fits",
            hdu=[0, 1],
            device="cpu",
            mmap=True,
            fp16=False,
            bf16=False,
            raw_scale=False,
            scale_on_device=True,
            columns=None,
            start_row=1,
            num_rows=-1,
            cache_capacity=10,
            handle_cache_capacity=16,
            fast_header=True,
            return_header=False,
            batch_to_device=lambda xs, d: xs,
            autodetect_hdu=lambda p, c: 0,
            resolve_image_mmap=lambda *a, **k: True,
            read_check_cache=lambda *a, **k: (False, None, None),
            read_header=lambda *a, **k: {},
            debug_scale=False,
            cold_nocache=False,
            read_exc_types=io._READ_EXC_TYPES,
            logger=logger,
        )
    assert len(out) == 2
    assert calls == [0, 1]


def _fallback_kwargs(cpp, handle, read_header, **overrides):
    kwargs = dict(
        cpp_module=cpp,
        path="a.fits",
        hdu=1,
        device="cpu",
        mmap=True,
        fp16=False,
        bf16=False,
        cache_capacity=0,
        handle_cache_capacity=16,
        fast_header=True,
        return_header=False,
        force_image=False,
        force_table=False,
        hdu_type_hint=None,
        columns=None,
        start_row=1,
        num_rows=-1,
        read_check_cache=lambda *a, **k: (False, None, None),
        resolve_image_mmap=lambda *a, **k: True,
        read_header=read_header,
    )
    kwargs.update(overrides)
    return kwargs


def test_fallback_table_typed_dispatch_ignores_message():
    """A decode error whose message contains 'truncated' must still fall
    through to the next reader: dispatch is by exception type, never by
    message (r4a-05, Main/slice-C directive)."""
    cpp = mock.Mock()
    cpp.get_hdu_type.return_value = "BINARY_TABLE"
    cpp.read_fits_table.side_effect = TypeError("table data truncated beyond EOF")
    cpp.read_fits_table_rows.return_value = {"A": torch.tensor([1.0, 2.0])}
    handle = mock.Mock()

    with mock.patch.object(
        _read_pipeline_fallback, "get_cached_handle", lambda p, c: (handle, False)
    ):
        out = _read_pipeline_fallback.read_fallback(
            **_fallback_kwargs(cpp, handle, lambda *a, **k: {}, force_table=True)
        )
    assert torch.equal(out["A"], torch.tensor([1.0, 2.0]))
    cpp.read_fits_table_rows.assert_called_once()


def test_fallback_table_io_error_same_message_propagates_typed():
    """An OSError whose message contains 'truncated' must propagate as the
    original typed error (r4a-05): message matching previously dispatched it
    into the decode fall-through and re-wrapped it as RuntimeError."""
    cpp = mock.Mock()
    cpp.get_hdu_type.return_value = "BINARY_TABLE"
    cpp.read_fits_table.side_effect = OSError("table data truncated at byte 42")
    handle = mock.Mock()

    with mock.patch.object(
        _read_pipeline_fallback, "get_cached_handle", lambda p, c: (handle, False)
    ):
        with pytest.raises(OSError, match="truncated at byte 42"):
            _read_pipeline_fallback.read_fallback(
                **_fallback_kwargs(cpp, handle, lambda *a, **k: {}, force_table=True)
            )
    cpp.read_fits_table_rows.assert_not_called()
    cpp.read_fits_table_from_handle.assert_not_called()


def test_fallback_table_all_readers_fail_keeps_first_error():
    """When every reader fails with a decode error, the first (primary
    reader) diagnosis is what surfaces — truncation identity is preserved
    without any message sniffing (r4a-05)."""
    cpp = mock.Mock()
    cpp.get_hdu_type.return_value = "BINARY_TABLE"
    cpp.read_fits_table.side_effect = RuntimeError("primary: data truncated at row 5")
    cpp.read_fits_table_rows.side_effect = RuntimeError("secondary: eof")
    cpp.read_fits_table_from_handle.side_effect = RuntimeError("tertiary: bad heap")
    handle = mock.Mock()

    with mock.patch.object(
        _read_pipeline_fallback, "get_cached_handle", lambda p, c: (handle, False)
    ):
        with pytest.raises(RuntimeError, match="primary: data truncated at row 5"):
            _read_pipeline_fallback.read_fallback(
                **_fallback_kwargs(cpp, handle, lambda *a, **k: {}, force_table=True)
            )


def test_fallback_rethrows_io_errors_typed(tmp_path, monkeypatch):
    """IO errors must surface typed (FileNotFoundError), not wrapped away
    into RuntimeError by the fallback's blanket except (r4a-05)."""
    path = str(tmp_path / "io_err.fits")
    torchfits.write(path, torch.zeros(4, 4), overwrite=True)

    def boom(p, cap):
        raise FileNotFoundError(p)

    monkeypatch.setattr(_read_pipeline_fallback, "get_cached_handle", boom)
    with pytest.raises(FileNotFoundError):
        torchfits.read(path, return_header=True)


def test_image_meta_rethrows_io_errors_typed():
    """Meta probes are best-effort for decode failures but must re-raise IO
    errors typed (r4a-05)."""
    from torchfits._io_engine import image_meta

    cpp = mock.Mock()
    cpp.read_shape.side_effect = FileNotFoundError("/gone.fits")
    with pytest.raises(FileNotFoundError):
        image_meta.get_image_meta("/gone.fits", 0, cpp_module=cpp)

    def boom_read_header(fh, hdu, fast):
        raise FileNotFoundError("/gone2.fits")

    with pytest.raises(FileNotFoundError):
        image_meta.get_image_meta_from_handle(
            mock.Mock(), "/gone2.fits", 0, read_header=boom_read_header
        )

    # Decode failures stay best-effort: meta=None, no raise.
    cpp3 = mock.Mock()
    cpp3.read_shape.side_effect = RuntimeError("bad header")
    cpp3.read_header_dict.side_effect = RuntimeError("bad header")
    assert image_meta.get_image_meta("/gone3.fits", 0, cpp_module=cpp3) is None


def test_image_meta_invalidated_on_file_replacement(tmp_path):
    """Policy meta must not outlive the file it describes (r4a-10): replacing
    the file must invalidate the cached meta."""
    from torchfits._io_engine import image_meta

    path = str(tmp_path / "swap.fits")
    torchfits.write(path, torch.zeros(4, 4), overwrite=True)  # BITPIX=-32
    meta1 = image_meta.get_image_meta(path, 0)
    assert meta1 is not None and meta1[0] == -32

    torchfits.write(path, torch.zeros(4, 4, dtype=torch.int16), overwrite=True)
    meta2 = image_meta.get_image_meta(path, 0)
    assert meta2 is not None and meta2[0] == 16


def test_fallback_table_dtype_parity_bit_and_unsigned(tmp_path):
    """Fallback dict reads keep the dtype/unsigned conventions: BIT->bool,
    TZERO=32768->uint16 (r4a-11, output-parity oracle conventions)."""
    import numpy as np
    from astropy.io import fits as afits

    n = 8
    cols = [
        afits.Column(name="FLAGS", format="X", array=np.array([0b10101010] * n)),
        afits.Column(name="U16", format="I", bzero=32768, array=np.arange(n) + 32768),
        afits.Column(name="I32", format="J", array=np.arange(n, dtype=np.int32)),
    ]
    path = str(tmp_path / "conv.fits")
    afits.BinTableHDU.from_columns(cols).writeto(path, overwrite=True)

    full = torchfits.read(path, hdu=1)
    assert full["FLAGS"].dtype == torch.bool
    assert full["U16"].dtype == torch.uint16
    assert full["I32"].dtype == torch.int32
    windowed = torchfits.read(path, hdu=1, start_row=2, num_rows=4)
    assert windowed["FLAGS"].dtype == torch.bool
    assert windowed["U16"].dtype == torch.uint16
    assert torch.equal(windowed["U16"], full["U16"][1:5])


def test_fallback_table_io_error_not_retried():
    """An OSError from one table reader must re-raise instead of falling
    through to slower readers that will fail again (r4a-05)."""
    cpp = mock.Mock()
    cpp.get_hdu_type.return_value = "BINARY_TABLE"
    cpp.read_fits_table_rows.side_effect = OSError("disk gone")
    handle = mock.Mock()

    with mock.patch.object(
        _read_pipeline_fallback, "get_cached_handle", lambda p, c: (handle, False)
    ):
        with pytest.raises(OSError, match="disk gone"):
            _read_pipeline_fallback.read_fallback(
                **_fallback_kwargs(
                    cpp,
                    handle,
                    lambda *a, **k: {},
                    force_table=True,
                    start_row=2,
                    num_rows=5,
                )
            )
    cpp.read_fits_table_rows.assert_called_once()
    cpp.read_fits_table_from_handle.assert_not_called()
    cpp.read_fits_table.assert_not_called()


def test_fallback_image_skips_header_read_without_return_header(monkeypatch):
    """The fallback image path must not read+parse the header it will
    discard: zero read_header calls without return_header=True (r4a-06)."""
    cpp = mock.Mock()
    cpp.get_hdu_type.return_value = "IMAGE"
    cpp.read_full.return_value = torch.zeros(2, 2)
    handle = mock.Mock()
    monkeypatch.setattr(
        _read_pipeline_fallback, "get_cached_handle", lambda p, c: (handle, False)
    )
    reads = []

    def counting_read_header(fh, hdu, fast):
        reads.append(hdu)
        return {}

    data = _read_pipeline_fallback.read_fallback(
        **_fallback_kwargs(cpp, handle, counting_read_header, return_header=False)
    )
    assert torch.equal(data, torch.zeros(2, 2))
    assert reads == []

    reads.clear()
    data, header = _read_pipeline_fallback.read_fallback(
        **_fallback_kwargs(cpp, handle, counting_read_header, return_header=True)
    )
    assert reads == [1]


def test_read_kwargs_handle_cache_capacity_warns_literal_text(tmp_path):
    """The read() kwarg site must emit the same pinned literal as
    tests/test_deprecated_knobs.py (r4a-09)."""
    path = str(tmp_path / "knob.fits")
    torchfits.write(path, torch.zeros(2, 2), overwrite=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = torchfits.read(path, handle_cache_capacity=8)
    assert out.shape == (2, 2)
    msgs = [
        str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert msgs == [
        "ReadOptions.handle_cache_capacity is ignored since the handle cache was "
        "removed; it will be removed in 2.0"
    ]
