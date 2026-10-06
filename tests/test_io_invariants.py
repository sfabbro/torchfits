"""IO invariants: TNULL/BLANK/NaN, binary copy, checksums, and where= agreement."""

from __future__ import annotations

import filecmp
import gzip
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

import torchfits
from torchfits.hdu import Header, TableHDURef


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "torchfits.cli", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_read_torch_where_drops_tnull_like_arrow(tmp_path: Path) -> None:
    from astropy.io import fits as afits

    path = tmp_path / "tnull_where.fits"
    m = np.array([10, 25, 30], dtype=np.int32)
    afits.BinTableHDU.from_columns(
        [afits.Column(name="M", format="J", array=m, null=25)]
    ).writeto(path.as_posix(), overwrite=True)

    arrow = torchfits.table.read(path.as_posix(), hdu=1, where="M > 20")
    torch_cols = torchfits.table.read_torch(path.as_posix(), hdu=1, where="M > 20")
    got_arrow = [v for v in arrow.column("M").to_pylist() if v is not None]
    got_torch = [int(v) for v in torch_cols["M"].tolist()]
    assert got_arrow == got_torch == [30]


def test_read_torch_where_out_of_range_int16_literal(tmp_path: Path) -> None:
    path = tmp_path / "i16_where.fits"
    vals = torch.tensor([1, 2, 3], dtype=torch.int16)
    torchfits.write(path.as_posix(), {"V": vals}, overwrite=True)

    arrow = torchfits.table.read(path.as_posix(), hdu=1, where="V > 40000")
    torch_cols = torchfits.table.read_torch(path.as_posix(), hdu=1, where="V > 40000")
    assert len(arrow.column("V")) == 0
    assert torch_cols["V"].numel() == 0


def test_copy_is_binary_identical_for_compimage(tmp_path: Path) -> None:
    src = tmp_path / "rice.fits"
    dst = tmp_path / "rice_copy.fits"
    torchfits.write(
        src.as_posix(),
        torch.randn(32, 32),
        compress="RICE_1",
        overwrite=True,
    )
    result = _run_cli("copy", src.as_posix(), dst.as_posix())
    assert result.returncode == 0, result.stderr
    assert filecmp.cmp(src.as_posix(), dst.as_posix(), shallow=False)


def test_copy_same_path_fails_cleanly(tmp_path: Path) -> None:
    src = tmp_path / "same.fits"
    torchfits.write(src.as_posix(), torch.ones(4, 4), overwrite=True)
    result = _run_cli("copy", src.as_posix(), src.as_posix())
    assert result.returncode == 2
    assert "same-path" in result.stderr.lower() or "in place" in result.stderr.lower()


def test_quantize_nan_is_nan_on_torchfits_read(tmp_path: Path) -> None:
    x = torch.randn(16, 16) * 10 + 100
    x[0, 0] = float("nan")
    path = tmp_path / "nanq.fits"
    torchfits.write(path.as_posix(), x, quantize="robust", overwrite=True)

    got = torchfits.read(path.as_posix())
    assert bool(torch.isnan(got[0, 0]))
    tensor = torchfits.read_tensor(path.as_posix())
    assert bool(torch.isnan(tensor[0, 0]))


def test_table_hdu_ref_head_composes_existing_window(tmp_path: Path) -> None:
    path = tmp_path / "window_head.fits"
    vals = torch.arange(20, dtype=torch.int32)
    torchfits.write(path.as_posix(), {"V": vals}, overwrite=True)
    ref = TableHDURef(
        header=torchfits.read_header(path.as_posix(), 1),
        source_path=path.as_posix(),
        source_hdu=1,
        row_slice=slice(10, 20),
    )
    headed = ref.head(3)
    got = headed.read()
    assert got["V"].tolist() == [10, 11, 12]


def test_return_header_cache_clones_header(tmp_path: Path) -> None:
    path = tmp_path / "hdr_cache.fits"
    torchfits.write(path.as_posix(), torch.ones(8, 8), overwrite=True)
    data1, hdr1 = torchfits.read(path.as_posix(), return_header=True)
    data2, hdr2 = torchfits.read(path.as_posix(), return_header=True)
    assert isinstance(hdr1, Header)
    assert isinstance(hdr2, Header)
    hdr1["POISON"] = 1
    data3, hdr3 = torchfits.read(path.as_posix(), return_header=True)
    assert "POISON" not in hdr3
    assert torch.equal(data1, data2)
    assert torch.equal(data2, data3)


def test_raw_scale_honored_on_return_header_fallback(tmp_path: Path) -> None:
    path = tmp_path / "u16.fits"
    raw = torch.tensor(list(range(16)), dtype=torch.uint16).reshape(4, 4)
    torchfits.write(path.as_posix(), raw, overwrite=True)
    tensor = torchfits.read_tensor(path.as_posix(), raw_scale=True)
    via_header, _hdr = torchfits.read(
        path.as_posix(), raw_scale=True, return_header=True
    )
    assert via_header.dtype == tensor.dtype
    assert torch.equal(via_header.cpu(), tensor.cpu())


def test_cpp_getattr_rejects_undocumented_names() -> None:
    import torchfits._C as native
    import torchfits._cpp as cpp

    assert hasattr(native, "HAS_BZIP2")
    with pytest.raises(AttributeError):
        getattr(cpp, "HAS_BZIP2")
    assert "HAS_BZIP2" not in dir(cpp)


def test_complex_arrow_error_names_table_read_torch(tmp_path: Path) -> None:
    path = tmp_path / "cplx.fits"
    z = torch.tensor([1 + 2j, 3 + 4j], dtype=torch.complex64)
    torchfits.write(path.as_posix(), {"Z": z}, overwrite=True)
    with pytest.raises(NotImplementedError, match=r"torchfits\.table\.read_torch"):
        torchfits.table.read(path.as_posix(), hdu=1)
    got = torchfits.table.read_torch(path.as_posix(), hdu=1)
    assert got["Z"].dtype == torch.complex64


def test_diff_unsigned_uint16_succeeds(tmp_path: Path) -> None:
    a = tmp_path / "a.fits"
    b = tmp_path / "b.fits"
    img = torch.tensor(list(range(16)), dtype=torch.uint16).reshape(4, 4)
    torchfits.write(a.as_posix(), img, overwrite=True)
    torchfits.write(b.as_posix(), img, overwrite=True)
    result = _run_cli("diff", a.as_posix(), b.as_posix())
    assert result.returncode == 0, result.stderr


def test_keyboard_interrupt_is_not_usage_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    from torchfits.cli import main as cli_main
    from torchfits.cli.common import EXIT_INTERRUPT

    parser = MagicMock()
    parser.parse_args.side_effect = KeyboardInterrupt()
    monkeypatch.setattr(cli_main, "build_parser", lambda: parser)
    assert cli_main.main([]) == EXIT_INTERRUPT


def test_stats_json_is_valid_with_nan(tmp_path: Path) -> None:
    path = tmp_path / "nan_img.fits"
    img = torch.tensor([[float("nan"), 1.0], [2.0, 3.0]])
    torchfits.write(path.as_posix(), img, overwrite=True)
    result = _run_cli("stats", path.as_posix(), "-f", "json")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert isinstance(payload, list)


def test_cfitsio_filter_ignores_bracket_in_directory(tmp_path: Path) -> None:
    from torchfits._io_engine.paths import cfitsio_base_path, has_cfitsio_filter

    nested = tmp_path / "[data]"
    nested.mkdir()
    path = nested / "file.fits"
    src = tmp_path / "plain.fits"
    torchfits.write(src.as_posix(), torch.ones(2, 2), overwrite=True)
    shutil.copy2(src, path)
    assert not has_cfitsio_filter(path.as_posix())
    assert cfitsio_base_path(path.as_posix()) == path.as_posix()
    filtered = path.as_posix() + "[1:2,1:2]"
    assert has_cfitsio_filter(filtered)
    assert cfitsio_base_path(filtered) == path.as_posix()
    got = torchfits.read(path.as_posix())
    assert got.shape == (2, 2)


def test_empty_reader_keeps_header_schema(tmp_path: Path) -> None:
    path = tmp_path / "empty_window.fits"
    torchfits.write(
        path.as_posix(),
        {"RA": torch.tensor([1.0, 2.0]), "DEC": torch.tensor([3.0, 4.0])},
        overwrite=True,
    )
    reader = torchfits.table.reader(path.as_posix(), hdu=1, row_slice=slice(0, 0))
    assert list(reader.schema.names) == ["RA", "DEC"]


def test_gzip_table_read_matches_uncompressed(tmp_path: Path) -> None:
    raw = tmp_path / "tbl.fits"
    gz = tmp_path / "tbl.fits.gz"
    vals = torch.arange(8, dtype=torch.int32)
    torchfits.write(raw.as_posix(), {"V": vals}, overwrite=True)
    with raw.open("rb") as src, gzip.open(gz, "wb") as dst:
        shutil.copyfileobj(src, dst)
    plain = torchfits.table.read_torch(raw.as_posix(), hdu=1, mmap=True)
    zipped = torchfits.table.read_torch(gz.as_posix(), hdu=1, mmap=True)
    assert zipped["V"].tolist() == plain["V"].tolist() == vals.tolist()


def test_replace_hdu_strips_stale_zimage(tmp_path: Path) -> None:
    path = tmp_path / "rice_replace.fits"
    torchfits.write(
        path.as_posix(),
        torch.ones(16, 16),
        compress="RICE_1",
        overwrite=True,
    )
    torchfits.replace_hdu(path.as_posix(), 1, torch.ones(16, 16) * 7)
    hdr = torchfits.read_header(path.as_posix(), 1)
    assert "ZIMAGE" not in hdr
    assert "ZCMPTYPE" not in hdr
    assert "ZCHECKSUM" not in hdr
    assert "ZDATASUM" not in hdr
    got = torchfits.read(path.as_posix(), hdu=1)
    assert torch.allclose(got.cpu(), torch.ones(16, 16) * 7)


def test_replace_hdu_restamps_checksums(tmp_path: Path) -> None:
    path = tmp_path / "cksum.fits"
    torchfits.write(path.as_posix(), torch.ones(8, 8), overwrite=True, checksum=True)
    before = torchfits.verify_checksums(path.as_posix())
    assert before["present"] is True
    torchfits.replace_hdu(path.as_posix(), 0, torch.ones(8, 8) * 2)
    after = torchfits.verify_checksums(path.as_posix())
    assert after["present"] is True
    assert after["ok"] is True


def test_read_hdu_list_honors_mmap_false(tmp_path: Path) -> None:
    path = tmp_path / "mef.fits"
    from torchfits import HDUList, TensorHDU

    hdul = HDUList(
        [
            TensorHDU(torch.ones(4, 4), header=Header({"EXTNAME": "A"})),
            TensorHDU(torch.ones(4, 4) * 2, header=Header({"EXTNAME": "B"})),
        ]
    )
    torchfits.write(path.as_posix(), hdul, overwrite=True)
    got = torchfits.read(path.as_posix(), hdu=[0, 1], mmap=False)
    assert len(got) == 2
    assert torch.allclose(got[1].cpu(), torch.ones(4, 4) * 2)


def test_quantize_on_hdulist_compress_raises(tmp_path: Path) -> None:
    from torchfits import HDUList, TensorHDU

    path = tmp_path / "q.fits"
    hdul = HDUList([TensorHDU(torch.randn(8, 8))])
    with pytest.raises(ValueError, match="quantize"):
        torchfits.write(
            path.as_posix(),
            hdul,
            compress="RICE_1",
            quantize="robust",
            overwrite=True,
        )


def test_keep_zero_nan_uses_blank(tmp_path: Path) -> None:
    x = torch.tensor([[float("nan"), 1.0], [2.0, 3.0]])
    path = tmp_path / "keepzero.fits"
    torchfits.write(
        path.as_posix(),
        x,
        quantize={"keep_zero": True},
        overwrite=True,
    )
    hdr = torchfits.read_header(path.as_posix())
    assert int(hdr["BLANK"]) == -32767
    got = torchfits.read(path.as_posix())
    assert bool(torch.isnan(got[0, 0]))


def test_keep_zero_nonpositive_nan_roundtrip(tmp_path: Path) -> None:
    x = torch.tensor([[-1.0, 0.0], [float("nan"), -2.0]])
    path = tmp_path / "keepzero_nonpos.fits"
    torchfits.write(
        path.as_posix(),
        x,
        quantize={"keep_zero": True},
        overwrite=True,
    )
    hdr = torchfits.read_header(path.as_posix())
    assert int(hdr["BLANK"]) == -32767
    got = torchfits.read(path.as_posix())
    assert got.dtype.is_floating_point
    assert bool(torch.isnan(got[1, 0]))
    assert float(got[0, 0]) == 0.0
    assert float(got[0, 1]) == 0.0


def test_keep_zero_identity_scale_preserves_peak_and_nan(tmp_path: Path) -> None:
    """BSCALE=1 / BZERO=0 with BLANK must not rewrite the linear map."""
    x = torch.tensor([[0.0, 32765.0], [float("nan"), 0.0]])
    path = tmp_path / "keepzero_peak.fits"
    torchfits.write(
        path.as_posix(),
        x,
        quantize={"keep_zero": True, "hi_q": 100.0},
        overwrite=True,
    )
    hdr = torchfits.read_header(path.as_posix())
    assert int(hdr["BLANK"]) == -32767
    got = torchfits.read(path.as_posix())
    assert bool(torch.isnan(got[1, 0]))
    assert float(got[0, 0]) == pytest.approx(0.0, abs=1e-5)
    assert float(got[0, 1]) == pytest.approx(32765.0, rel=1e-5, abs=1.0)


def test_quantize_nan_is_nan_on_subset(tmp_path: Path) -> None:
    x = torch.randn(16, 16) * 10 + 100
    x[0, 0] = float("nan")
    path = tmp_path / "nanq_cut.fits"
    torchfits.write(path.as_posix(), x, quantize="robust", overwrite=True)
    cut = torchfits.read_subset(path.as_posix(), 0, 0, 0, 2, 2)
    assert bool(torch.isnan(cut[0, 0]))


def test_quantize_table_nan_is_nan_on_read_torch(tmp_path: Path) -> None:
    v = {"V": torch.cat([torch.randn(32) * 5 + 50, torch.tensor([float("nan")])])}
    path = tmp_path / "tblq_nan.fits"
    torchfits.write(path.as_posix(), v, quantize="robust", overwrite=True)
    got = torchfits.table.read_torch(path.as_posix(), hdu=1)
    assert bool(torch.isnan(got["V"][-1]))
    assert not bool(torch.isnan(got["V"][:-1]).any())


def test_integer_blank_is_nan_on_read(tmp_path: Path) -> None:
    from astropy.io import fits as afits

    path = tmp_path / "blank_i16.fits"
    data = np.array([[1, 2], [-32767, 4]], dtype=np.int16)
    hdu = afits.PrimaryHDU(data)
    hdu.header["BLANK"] = -32767
    hdu.writeto(path.as_posix(), overwrite=True)
    got = torchfits.read(path.as_posix())
    assert got.dtype.is_floating_point
    assert bool(torch.isnan(got[1, 0]))
    assert float(got[0, 0]) == 1.0
    cut = torchfits.read_subset(path.as_posix(), 0, 0, 0, 2, 2)
    assert bool(torch.isnan(cut[1, 0]))


def test_native_float32_roundtrip_stays_finite(tmp_path: Path) -> None:
    x = torch.linspace(0.1, 1.6, 16, dtype=torch.float32).reshape(4, 4)
    path = tmp_path / "f32.fits"
    torchfits.write(path.as_posix(), x, overwrite=True)
    got = torchfits.read(path.as_posix())
    assert not bool(torch.isnan(got).any())
    assert torch.allclose(got, x)


def test_header_parser_fortran_d_exponent() -> None:
    from torchfits.header_parser import FastHeaderParser

    card = "BSCALE  =              1.5D-3 / scale".ljust(80)
    _kw, value, _comment = FastHeaderParser._parse_card(card)
    assert value == pytest.approx(0.0015)


def test_to_arrow_keeps_vector_column_rows() -> None:
    from torchfits._tensor_buffer import tensor_to_arrow_array

    pa = pytest.importorskip("pyarrow")
    vec = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    arr = tensor_to_arrow_array(vec, pa)
    assert len(arr) == 3
    assert arr.type.list_size == 2


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_to_arrow_rejects_complex_tensors(dtype: torch.dtype) -> None:
    from torchfits._tensor_buffer import tensor_to_arrow_array

    pa = pytest.importorskip("pyarrow")
    value = torch.tensor([1 + 2j, 3 + 4j], dtype=dtype)
    with pytest.raises(ValueError, match="complex tensors"):
        tensor_to_arrow_array(value, pa)


@pytest.mark.parametrize("row_count", [0, 3])
def test_to_arrow_rejects_zero_width_2d_tensors(row_count: int) -> None:
    from torchfits._tensor_buffer import tensor_to_arrow_array

    pa = pytest.importorskip("pyarrow")
    value = torch.empty((row_count, 0), dtype=torch.float32)
    with pytest.raises(ValueError, match="zero width"):
        tensor_to_arrow_array(value, pa)


def test_to_arrow_preserves_empty_tensor_shapes() -> None:
    from torchfits._tensor_buffer import tensor_to_arrow_array

    pa = pytest.importorskip("pyarrow")
    assert len(tensor_to_arrow_array(torch.empty(0), pa)) == 0
    empty_rows = tensor_to_arrow_array(torch.empty((0, 2)), pa)
    assert len(empty_rows) == 0
    assert empty_rows.type.list_size == 2


# NOTE (R2-056): a ``test_kmp_duplicate_lib_ok_set_on_import`` used to live here
# and has been removed. It asserted only what the ambient environment already
# guaranteed -- pixi's activation env exports ``KMP_DUPLICATE_LIB_OK=TRUE`` for
# every task, and CI sets it explicitly on the release-gate job -- so deleting the
# ``setdefault`` in ``src/torchfits/__init__.py`` left this file fully green
# (51 passed). Its Linux branch could not fail either: the only values reachable
# under pixi were the ones it listed. The real guard is
# ``tests/test_package_isolation.py::test_import_sets_kmp_duplicate_lib_ok``,
# which pops the variable and imports in a subprocess; the same mutation fails
# there, and in ``test_duplicate_libomp_survives_after_import``.


def test_hdulist_write_same_path_does_not_corrupt(tmp_path: Path) -> None:
    path = tmp_path / "inplace.fits"
    torchfits.write(path.as_posix(), torch.arange(16.0).reshape(4, 4), overwrite=True)
    with torchfits.open(path.as_posix()) as hdul:
        hdul.write(path.as_posix(), overwrite=True)
    got = torchfits.read(path.as_posix())
    assert got.shape == (4, 4)
    assert not torch.isnan(got).any()


def test_http_is_scaled_treats_blank() -> None:
    from torchfits._io_engine.http_subset import _is_scaled

    assert _is_scaled({"BITPIX": 16, "BSCALE": 1.0, "BZERO": 0.0, "BLANK": -32767})
    assert not _is_scaled({"BITPIX": 16, "BSCALE": 1.0, "BZERO": 0.0})
    assert _is_scaled({"BITPIX": 16, "BSCALE": 0.01, "BZERO": 0.0})


def test_integer_blank_raw_scale_keeps_sentinel(tmp_path: Path) -> None:
    from astropy.io import fits as afits

    path = tmp_path / "blank_raw.fits"
    data = np.array([[1, 2], [-32767, 4]], dtype=np.int16)
    hdu = afits.PrimaryHDU(data)
    hdu.header["BLANK"] = -32767
    hdu.writeto(path.as_posix(), overwrite=True)
    raw = torchfits.read(path.as_posix(), raw_scale=True)
    assert raw.dtype == torch.int16
    assert int(raw[1, 0].item()) == -32767


def test_integer_blank_mmap_agrees(tmp_path: Path) -> None:
    from astropy.io import fits as afits

    path = tmp_path / "blank_mmap.fits"
    data = np.array([[1, 2], [-32767, 4]], dtype=np.int16)
    hdu = afits.PrimaryHDU(data)
    hdu.header["BLANK"] = -32767
    hdu.writeto(path.as_posix(), overwrite=True)
    a = torchfits.read(path.as_posix(), mmap=True)
    b = torchfits.read(path.as_posix(), mmap=False)
    assert a.dtype.is_floating_point and b.dtype.is_floating_point
    assert bool(torch.isnan(a[1, 0])) and bool(torch.isnan(b[1, 0]))
    assert float(a[0, 0]) == float(b[0, 0]) == 1.0


def test_write_decoded_float_drops_integer_bscale(tmp_path: Path) -> None:
    """Physical float + copied BITPIX=16 header must not replay BSCALE."""
    x = torch.tensor([[10.0, 20.0], [float("nan"), 40.0]])
    src = tmp_path / "qsrc.fits"
    dst = tmp_path / "qdst.fits"
    torchfits.write(src.as_posix(), x, quantize="robust", overwrite=True)
    cut = torchfits.read_subset(src.as_posix(), 0, 0, 0, 2, 2)
    hdr = Header(torchfits.read_header(src.as_posix()))
    hdr["OBJECT"] = "KEEPME"
    torchfits.write_tensor(dst.as_posix(), cut, header=hdr, overwrite=True)
    out_hdr = torchfits.read_header(dst.as_posix())
    assert "BSCALE" not in out_hdr
    assert "BZERO" not in out_hdr
    assert "BLANK" not in out_hdr
    assert out_hdr["OBJECT"] == "KEEPME"
    got = torchfits.read(dst.as_posix())
    assert bool(torch.isnan(got[1, 0]))
    assert float(got[0, 0]) == pytest.approx(float(cut[0, 0]), rel=1e-5, abs=1e-3)


def test_float_header_bscale_is_kept(tmp_path: Path) -> None:
    path = tmp_path / "fscale.fits"
    torchfits.write(
        path.as_posix(),
        torch.ones(4, 4) * 10,
        header={"BITPIX": -32, "BSCALE": 2.0},
        overwrite=True,
    )
    hdr = torchfits.read_header(path.as_posix())
    assert float(hdr["BSCALE"]) == pytest.approx(2.0)
    got = torchfits.read(path.as_posix())
    assert torch.allclose(got, torch.ones(4, 4) * 20)


def test_cli_cutout_quantize_matches_subset(tmp_path: Path) -> None:
    src = tmp_path / "qcut_src.fits"
    dst = tmp_path / "qcut_dst.fits"
    x = torch.tensor([[10.0, 20.0], [float("nan"), 40.0]])
    torchfits.write(src.as_posix(), x, quantize="robust", overwrite=True)
    expected = torchfits.read_subset(src.as_posix(), 0, 0, 0, 2, 2)
    result = _run_cli("cutout", src.as_posix(), dst.as_posix(), "--box", "0,0,2,2")
    assert result.returncode == 0, result.stderr
    got = torchfits.read(dst.as_posix())
    assert bool(torch.isnan(got[1, 0]))
    assert float(got[0, 0]) == pytest.approx(float(expected[0, 0]), rel=1e-5, abs=1e-3)


def test_replace_hdu_strips_blank(tmp_path: Path) -> None:
    from astropy.io import fits as afits

    path = tmp_path / "blank_replace.fits"
    data = np.array([[1, 2], [-32767, 4]], dtype=np.int16)
    hdu = afits.PrimaryHDU(data)
    hdu.header["BLANK"] = -32767
    hdu.writeto(path.as_posix(), overwrite=True)
    torchfits.replace_hdu(path.as_posix(), 0, torch.ones(2, 2) * 7)
    hdr = torchfits.read_header(path.as_posix())
    assert "BLANK" not in hdr
    got = torchfits.read(path.as_posix())
    assert torch.allclose(got.cpu(), torch.ones(2, 2) * 7)


def test_quantize_table_where_excludes_nan(tmp_path: Path) -> None:
    v = {"V": torch.tensor([10.0, 20.0, 30.0, float("nan")])}
    path = tmp_path / "tblq_where.fits"
    torchfits.write(path.as_posix(), v, quantize="robust", overwrite=True)
    for mmap in (True, False):
        arrow = torchfits.table.read(path.as_posix(), hdu=1, mmap=mmap, where="V > 15")
        torch_cols = torchfits.table.read_torch(
            path.as_posix(), hdu=1, mmap=mmap, where="V > 15"
        )
        arrow_n = len([x for x in arrow.column("V").to_pylist() if x is not None])
        assert arrow_n == 2
        assert torch_cols["V"].numel() == 2
        assert not bool(torch.isnan(torch_cols["V"]).any())


def test_quantize_table_isnull_is_arrow_null(tmp_path: Path) -> None:
    v = {"V": torch.tensor([10.0, 20.0, float("nan")])}
    path = tmp_path / "tblq_isnull.fits"
    torchfits.write(path.as_posix(), v, quantize="robust", overwrite=True)
    full = torchfits.table.read(path.as_posix(), hdu=1)
    assert full.column("V").null_count == 1
    nulls = torchfits.table.read(path.as_posix(), hdu=1, where="V IS NULL")
    assert len(nulls) == 1
    assert nulls.column("V").to_pylist() == [None]
    kept = torchfits.table.read(path.as_posix(), hdu=1, where="V IS NOT NULL")
    assert len(kept) == 2
    assert all(x is not None for x in kept.column("V").to_pylist())


def test_native_float_nan_is_not_fits_tnull(tmp_path: Path) -> None:
    """IEEE NaN without TNULLn stays a value; IS NULL is a FITS TNULL check."""
    path = tmp_path / "float_nan.fits"
    torchfits.write(
        path.as_posix(),
        {"V": torch.tensor([1.0, float("nan"), 3.0])},
        overwrite=True,
    )
    nulls = torchfits.table.read(path.as_posix(), hdu=1, where="V IS NULL")
    assert len(nulls) == 0
    full = torchfits.table.read(path.as_posix(), hdu=1)
    assert full.column("V").null_count == 0


def test_dataview_dtype_blank_is_float(tmp_path: Path) -> None:
    from astropy.io import fits as afits

    path = tmp_path / "blank_dv.fits"
    data = np.array([[1, 2], [-32767, 4]], dtype=np.int16)
    hdu = afits.PrimaryHDU(data)
    hdu.header["BLANK"] = -32767
    hdu.writeto(path.as_posix(), overwrite=True)
    with torchfits.open(path.as_posix()) as hdul:
        view = hdul[0].data
        assert view.dtype == torch.float64
        sl = view[:2, :2]
        assert sl.dtype.is_floating_point
        assert bool(torch.isnan(sl[1, 0]))


def test_native_float_inf_and_signed_zero_survive(tmp_path: Path) -> None:
    """CFITSIO fnan() must not run on uncompressed IEEE (nulval=nullptr)."""
    path = tmp_path / "ieee.fits"
    data = np.array([[0.0, -0.0], [np.inf, -np.inf]], dtype=np.float32)
    from astropy.io import fits as afits

    afits.PrimaryHDU(data).writeto(path.as_posix(), overwrite=True)
    got = torchfits.read_tensor(path.as_posix()).numpy()
    np.testing.assert_array_equal(got, data)
    assert not np.signbit(got[0, 0])
    assert np.signbit(got[0, 1])
    assert np.isposinf(got[1, 0])
    assert np.isneginf(got[1, 1])
    subset = torchfits.read_subset(path.as_posix(), 0, 0, 0, 2, 2).numpy()
    np.testing.assert_array_equal(subset, data)


def _multi_hdu_fixture(path: Path) -> list[tuple[int, ...]]:
    """Write a MEF whose HDUs are pairwise distinguishable, return their shapes.

    Built with astropy so the expectation does not depend on the writer under
    test. Every HDU has a unique shape, and the pair (BSCALE/BZERO, compression)
    differs too, so a metadata lookup that reads the wrong HDU cannot go unnoticed
    the way it would on a file of same-shaped images.
    """
    from astropy.io import fits

    shapes = [(4, 5), (7, 3), (9, 2, 6), (2, 11), (8, 8)]
    primary = fits.PrimaryHDU()
    hdus: list[fits.ImageHDU] = [primary]
    for i, shape in enumerate(shapes):
        data = np.full(shape, i + 1, dtype=np.float32)
        if i == 1:
            hdu = fits.ImageHDU(data)
            hdu.header["BSCALE"] = 4.0
            hdu.header["BZERO"] = 10.0
        elif i == 3:
            hdu = fits.CompImageHDU(data, compression_type="RICE_1")
        else:
            hdu = fits.ImageHDU(data)
        hdus.append(hdu)
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return shapes


def test_hdu_metadata_is_not_confused_by_a_moved_cursor(tmp_path: Path) -> None:
    """HDU-keyed metadata lookups must answer for the HDU they were asked about.

    Regression guard for the CFITSIO current-HDU cursor. ``FITSFile`` caches
    image info, scale and compression state in maps keyed by HDU number, but
    CFITSIO itself answers from whichever HDU the handle is currently positioned
    on. Reading a file's HDUs in a shuffled order exercises one shared handle
    moving between HDUs, which is the situation where a lookup that trusted the
    cursor position instead of its own key would report the neighbouring HDU's
    shape.
    """
    path = tmp_path / "mef.fits"
    shapes = _multi_hdu_fixture(path)

    with torchfits.open(path) as hdul:
        # Reverse order, and revisit early HDUs after later ones, so every read
        # follows a cursor move except the first. Index 0 is the primary HDU, so
        # image HDU k lives at hdul[k] and torchfits.read uses hdu=k as well.
        for k in (5, 1, 4, 2, 3, 1, 5):
            got = tuple(hdul[k].data.shape)
            expected = shapes[k - 1]
            assert got == expected, (
                f"image HDU {k} reported shape {got}, expected {expected}"
            )

    # The same file, read one HDU per call, must agree with the shuffled pass.
    for k, expected in enumerate(shapes, start=1):
        assert tuple(torchfits.read(path, hdu=k).shape) == expected


def test_hdu_keyed_caches_position_the_cursor_themselves() -> None:
    """Structural gate: the three HDU-keyed lookups must call ensure_hdu_checked.

    These three cache on ``hdu_num`` but read the CFITSIO current-HDU cursor on a
    miss. Positioning was left to the caller, which is a precondition nothing
    states and nothing enforces — every one of the twelve call sites happened to
    satisfy it, so the suite passed either way. That is precisely the kind of
    invariant a test should not rely on being true by convention, so the calls
    are pinned here.

    Structural rather than behavioural: ``FITSFile`` symbols are hidden in the
    extension, so a C++ probe cannot link against it the way the core-library
    probes link ``libtorchfits_core``. The behavioural test above guards the
    observable contract; this one guards the enforcement.
    """
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "torchfits"
        / "cpp_src"
        / "fits_file.cpp"
    ).read_text()

    for method in (
        "FITSFile::get_scale_info",
        "FITSFile::is_compressed_image_cached",
        "FITSFile::get_image_info",
    ):
        start = source.index(method)
        body = source[start : source.index("\n}", start)]
        assert "ensure_hdu_checked(" in body, (
            f"{method} caches on hdu_num but reads the CFITSIO cursor on a miss; "
            "it must position the cursor itself"
        )


def test_every_capsule_reader_binding_holds_the_reader_mutex() -> None:
    """Any binding handed a TableReader capsule must hold its mutex.

    ``TableReader::io_mutex_`` is declared in ``table_reader.h`` and acquired
    nowhere in that header -- the bindings that hand out a *shared* reader take it
    instead, because the only shared reader is the ``open_fits_mmap_reader``
    capsule. That split is easy to misread: the header comment described the
    serialization without saying where it happened, which is misleading enough
    that reading the header alone suggests the mutex is dead code.

    This gate makes the coupling explicit and keeps it true. A new binding that
    accepts a capsule without locking would otherwise be a silent data race on
    the CFITSIO cursor and on ``scratch_buffer_``, with the GIL released.
    """
    cpp_src = Path(__file__).resolve().parents[1] / "src" / "torchfits" / "cpp_src"
    table_bindings = (cpp_src / "table_bindings.cpp").read_text()
    table_reader = (cpp_src / "table_reader.h").read_text()

    assert "mutable std::mutex io_mutex_;" in table_reader, (
        "TableReader must keep a mutex for the shared-capsule path"
    )

    # Producer: open_fits_mmap_reader is the only thing that hands out a capsule.
    producers = table_bindings.count("-> nb::capsule")
    assert producers == 1, (
        f"expected one capsule producer, found {producers}; a second sharing path "
        "needs its own locking decision"
    )

    # Consumers: every lambda taking an nb::capsule parameter must lock. Matched
    # by scanning for the parameter rather than by counting, so a new consumer is
    # covered without editing this test.
    consumers = 0
    # A C++ lambda is [capture](params) -> ret: the capsule is a *parameter*, so
    # it appears inside the parentheses, not the brackets.
    pattern = r"\[\s*[^\]]*\]\s*\(([^)]*nb::capsule[^)]*)\)\s*->"
    for match in re.finditer(pattern, table_bindings):
        consumers += 1
        body_start = table_bindings.index("{", match.end())
        # The lambda body: brace-match from its opening brace.
        depth, idx = 0, body_start
        while idx < len(table_bindings):
            if table_bindings[idx] == "{":
                depth += 1
            elif table_bindings[idx] == "}":
                depth -= 1
                if depth == 0:
                    break
            idx += 1
        body = table_bindings[body_start : idx + 1]
        assert "io_mutex_" in body, (
            "a binding that accepts a TableReader capsule must hold io_mutex_ for "
            "the whole call (GIL released), or the CFITSIO cursor and "
            "scratch_buffer_ are raced"
        )
        assert "gil_scoped_release" in body, (
            "the lock is only useful if the GIL is released around the read"
        )
    assert consumers >= 1, "no capsule consumers found -- has the binding moved?"
