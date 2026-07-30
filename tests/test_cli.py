"""Smoke tests for the torchfits CLI."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
import torch

import torchfits


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "torchfits.cli", *args],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def image_fits(tmp_path):
    path = tmp_path / "image.fits"
    data = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    torchfits.write(str(path), data, header={"BITPIX": -32}, overwrite=True)
    return path


def test_help_lists_subcommands():
    result = _run_cli("--help")
    assert result.returncode == 0
    for name in (
        "info",
        "header",
        "verify",
        "stats",
        "table",
        "convert",
        "probe",
        "copy",
        "arith",
        "diff",
    ):
        assert name in result.stdout


def test_info_json(image_fits):
    result = _run_cli("info", str(image_fits), "--json")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload[0]["file"] == str(image_fits)
    assert payload[0]["hdu"] == 0
    assert payload[0]["type"] == "IMAGE"


def test_header_keyword(image_fits):
    result = _run_cli("header", str(image_fits), "--keyword", "BITPIX", "--json")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert any(row["keyword"] == "BITPIX" for row in payload)


def test_header_text_dumps_all_hdus_formatted(tmp_path):
    path = tmp_path / "mef.fits"
    a = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    b = torch.arange(4, dtype=torch.float32).reshape(2, 2) + 10
    torchfits.write(str(path), [a, b], overwrite=True)

    result = _run_cli("header", str(path))
    assert result.returncode == 0, result.stderr
    assert f"# HDU 0 (PRIMARY) in {path}:" in result.stdout
    assert f"# HDU 1 (HDU1) in {path}:" in result.stdout
    assert "SIMPLE  =" in result.stdout
    assert "XTENSION=" in result.stdout
    assert "SIMPLE  =                    T" in result.stdout
    # Text mode should not use the old file:hdu prefix / repr quoting style.
    assert f"{path}:0 " not in result.stdout
    assert "SIMPLE  = 'T'" not in result.stdout


def test_compress_jobs_help_and_multi_out_dir(image_fits, tmp_path):
    help_result = _run_cli("compress", "--help")
    assert help_result.returncode == 0, help_result.stderr
    assert "--jobs" in help_result.stdout
    assert "--file-jobs" in help_result.stdout
    assert "--split" in help_result.stdout
    assert "pytorch" in help_result.stdout.lower()
    assert "intra-op" in help_result.stdout.lower()

    other = tmp_path / "other.fits"
    torchfits.write(
        str(other),
        torch.arange(16, dtype=torch.float32).reshape(4, 4),
        overwrite=True,
    )
    out_dir = tmp_path / "fz"
    result = _run_cli(
        "compress",
        str(image_fits),
        str(other),
        "--out-dir",
        str(out_dir),
        "-j",
        "1",
    )
    assert result.returncode == 0, result.stderr
    assert (out_dir / image_fits.name).is_file()
    assert (out_dir / other.name).is_file()


def test_compress_split_hdu_mef(tmp_path):
    path = tmp_path / "mef.fits"
    a = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    b = torch.arange(4, dtype=torch.float32).reshape(2, 2) + 10
    torchfits.write(str(path), [a, b], overwrite=True)

    out_dir = tmp_path / "by_hdu"
    result = _run_cli(
        "compress",
        str(path),
        "--split",
        "hdu",
        "--out-dir",
        str(out_dir),
        "-j",
        "1",
    )
    assert result.returncode == 0, result.stderr
    out0 = out_dir / "mef_hdu00.fits"
    out1 = out_dir / "mef_hdu01.fits"
    assert out0.is_file()
    assert out1.is_file()
    # Compressed write uses empty primary + CompImage at HDU 1.
    t0 = torchfits.read_tensor(str(out0), hdu=1)
    t1 = torchfits.read_tensor(str(out1), hdu=1)
    assert t0.shape == (2, 2)
    assert t1.shape == (2, 2)

    dec_dir = tmp_path / "decoded"
    dec = _run_cli(
        "decompress",
        str(out0),
        str(out1),
        "--out-dir",
        str(dec_dir),
        "-j",
        "1",
    )
    assert dec.returncode == 0, dec.stderr
    assert (dec_dir / out0.name).is_file()
    assert (dec_dir / out1.name).is_file()
    assert torchfits.read_tensor(str(dec_dir / out0.name), hdu=1).shape == (2, 2)


def test_verify_without_checksums(image_fits):
    result = _run_cli("verify", str(image_fits), "--json")
    # No checksum keywords → ok=True, exit 0 (not corrupt, just unverified).
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload[0]["ok"] is True
    assert payload[0]["status"] == "no_checksums"


def test_verify_after_write_checksums(image_fits):
    torchfits.write_checksums(str(image_fits), hdu=0)
    result = _run_cli("verify", str(image_fits))
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_verify_text_output_messaging_contract(tmp_path):
    """Lock in the three verify text-output labels in text mode:

    - "OK (no checksum keywords)" — file has no DATASUM/CHECKSUM keywords
    - "OK"                       — checksums present and valid
    - "FAIL"                     — checksums present but corrupt (exit 4)
    """
    path = tmp_path / "verify.fits"
    data = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    torchfits.write(str(path), data, header={"FOO": 1}, overwrite=True)

    # 1. No checksum keywords → OK (no checksum keywords), exit 0.
    result = _run_cli("verify", str(path))
    assert result.returncode == 0, result.stderr
    assert "OK (no checksum keywords)" in result.stdout
    assert "FAIL" not in result.stdout

    # 2. Write checksums → OK, exit 0.
    torchfits.write_checksums(str(path), hdu=0)
    result = _run_cli("verify", str(path))
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
    assert "no checksum keywords" not in result.stdout
    assert "FAIL" not in result.stdout

    # 3. Corrupt a non-structural header keyword (FOO) → CHECKSUM fails, exit 4.
    # Changing a non-structural keyword alters header bytes without making
    # the file unopenable, so CHECKSUM recomputation diverges from the stored value.
    with open(path, "r+b") as f:
        raw = f.read()
        needle = b"FOO     ="
        idx = raw.find(needle)
        assert idx != -1, "FOO card not found"
        card = bytearray(raw[idx : idx + 80])
        one = card.find(b"1")
        assert one != -1, "digit to corrupt not found in FOO card"
        card[one : one + 1] = b"2"
        f.seek(idx)
        f.write(card)

    result = _run_cli("verify", str(path))
    assert result.returncode == 4, result.stderr
    assert "FAIL" in result.stdout


def test_copy_roundtrip(image_fits, tmp_path):
    out = tmp_path / "copy.fits"
    result = _run_cli("copy", str(image_fits), str(out))
    assert result.returncode == 0, result.stderr
    assert out.is_file()
    copied = torchfits.read_tensor(str(out), hdu=0)
    original = torchfits.read_tensor(str(image_fits), hdu=0)
    assert torch.equal(copied, original)


def test_arith_add(image_fits, tmp_path):
    out = tmp_path / "arith.fits"
    result = _run_cli(
        "arith",
        str(image_fits),
        "--op",
        "add",
        "--value",
        "1",
        "--out",
        str(out),
    )
    assert result.returncode == 0, result.stderr
    tensor = torchfits.read_tensor(str(out), hdu=0)
    expected = torchfits.read_tensor(str(image_fits), hdu=0) + 1.0
    assert torch.allclose(tensor, expected)


def test_arith_image_image_and_mef_stack(tmp_path):
    a_path = tmp_path / "a.fits"
    b_path = tmp_path / "b.fits"
    mef_path = tmp_path / "mef.fits"
    a = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    b = torch.ones(2, 2, dtype=torch.float32) * 3
    torchfits.write(str(a_path), a, overwrite=True)
    torchfits.write(str(b_path), b, overwrite=True)
    torchfits.write(
        str(mef_path),
        [
            {"data": a, "header": {"OBJECT": "A0"}},
            {"data": a + 1, "header": {"OBJECT": "A1"}},
        ],
        overwrite=True,
    )

    out = tmp_path / "mul.fits"
    result = _run_cli("arith", str(a_path), str(b_path), "--op", "mul", "-o", str(out))
    assert result.returncode == 0, result.stderr
    assert torch.allclose(torchfits.read_tensor(str(out), hdu=0), a * b)

    mef_out = tmp_path / "mef_mul.fits"
    result = _run_cli(
        "arith",
        str(mef_path),
        "--op",
        "mul",
        "--value",
        "2",
        "-e",
        "0,1",
        "-o",
        str(mef_out),
        "-j",
        "1",
    )
    assert result.returncode == 0, result.stderr
    assert torch.allclose(torchfits.read_tensor(str(mef_out), hdu=0), a * 2)
    assert torch.allclose(torchfits.read_tensor(str(mef_out), hdu=1), (a + 1) * 2)
    assert torchfits.read_header(str(mef_out), 0).get("OBJECT") == "A0"
    assert torchfits.read_header(str(mef_out), 1).get("OBJECT") == "A1"


def test_compress_rejects_duplicate_out_basenames(tmp_path):
    d1 = tmp_path / "d1"
    d2 = tmp_path / "d2"
    d1.mkdir()
    d2.mkdir()
    a = d1 / "same.fits"
    b = d2 / "same.fits"
    data = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    torchfits.write(str(a), data, overwrite=True)
    torchfits.write(str(b), data, overwrite=True)
    result = _run_cli(
        "compress", str(a), str(b), "--out-dir", str(tmp_path / "out"), "-J", "1"
    )
    assert result.returncode == 2, result.stderr
    assert "duplicate basename" in result.stderr.lower()


def test_arith_multi_file_out_dir(image_fits, tmp_path):
    other = tmp_path / "other.fits"
    torchfits.write(
        str(other),
        torch.arange(16, dtype=torch.float32).reshape(4, 4) + 1,
        overwrite=True,
    )
    out_dir = tmp_path / "arith_out"
    result = _run_cli(
        "arith",
        str(image_fits),
        str(other),
        "--op",
        "add",
        "--value",
        "1",
        "--out-dir",
        str(out_dir),
        "-J",
        "2",
    )
    assert result.returncode == 0, result.stderr
    assert (out_dir / image_fits.name).is_file()
    assert (out_dir / other.name).is_file()


def test_verify_file_jobs_help(image_fits):
    help_result = _run_cli("verify", "--help")
    assert help_result.returncode == 0, help_result.stderr
    assert "--file-jobs" in help_result.stdout
    result = _run_cli("verify", str(image_fits), "-J", "1", "--json")
    assert result.returncode == 0, result.stderr


def test_diff_same_file(image_fits):
    result = _run_cli("diff", str(image_fits), str(image_fits))
    assert result.returncode == 0, result.stderr


def test_diff_detects_change(image_fits, tmp_path):
    other = tmp_path / "other.fits"
    data = torchfits.read_tensor(str(image_fits), hdu=0) + 10.0
    torchfits.write(str(other), data, overwrite=True)
    result = _run_cli("diff", str(image_fits), str(other))
    assert result.returncode == 1, result.stderr
    assert result.stderr.strip()


def test_convert_png(image_fits, tmp_path):
    out = tmp_path / "rgb.png"
    result = _run_cli(
        "convert",
        str(image_fits),
        str(out),
        "--bands",
        "0,0,0",
    )
    assert result.returncode == 0, result.stderr
    assert out.read_bytes()[:4] == b"\x89PNG"


def test_lupton_rgb_zero_size_input():
    """lupton_rgb must not crash on a zero-size (e.g. degenerate cutout) band."""
    from torchfits.transforms.rgb import lupton_rgb

    r = g = b = torch.zeros(0, 5)
    out = lupton_rgb(r, g, b)
    assert out.shape == (0, 5, 3)


def test_lupton_rgb_preserves_midtones_with_bright_star():
    """Field-wide /peak normalisation crushed midtones to near-black.

    One saturated star must not force the rest of the field toward zero;
    Astropy-style mapping uses per-pixel peak clip only when max(R,G,B) > 1.
    """
    from torchfits.transforms.rgb import lupton_rgb

    h, w = 32, 32
    r = g = b = torch.full((h, w), 5.0)
    r = r.clone()
    r[0, 0] = 1.0e6  # saturated star
    out = lupton_rgb(r, g, b, Q=8.0, stretch=0.5)
    mid = float(out[16, 16].mean())
    star = float(out[0, 0].max())
    assert star == pytest.approx(1.0, abs=1e-6)
    assert mid > 0.05, f"midtones crushed: {mid}"


def test_lupton_rgb_astropy_parity():
    """Match Astropy ``make_lupton_rgb`` (LuptonAsinhStretch)."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("astropy")
    from astropy.visualization import make_lupton_rgb
    from torchfits.transforms.rgb import lupton_rgb

    torch.manual_seed(0)
    r = torch.rand(24, 24) * 40.0
    g = torch.rand(24, 24) * 40.0
    b = torch.rand(24, 24) * 40.0
    ours = lupton_rgb(r, g, b, Q=8.0, stretch=0.5).numpy()
    # Astropy <5.x has no output_dtype=; compare via uint8 / 255.
    try:
        ref = make_lupton_rgb(
            r.numpy(),
            g.numpy(),
            b.numpy(),
            Q=8.0,
            stretch=0.5,
            output_dtype=np.float64,
        )
        err = float(np.abs(ours - ref).max())
        assert err < 1e-6, f"max abs err vs Astropy float={err}"
    except TypeError:
        ref_u8 = (
            make_lupton_rgb(r.numpy(), g.numpy(), b.numpy(), Q=8.0, stretch=0.5).astype(
                np.float64
            )
            / 255.0
        )
        err = float(np.abs(ours - ref_u8).mean())
        assert err < 0.01, f"mean abs err vs Astropy uint8={err}"


def test_convert_infers_format_from_extension(table_fits, tmp_path):
    out = tmp_path / "table.parquet"
    result = _run_cli("convert", str(table_fits), str(out), "--hdu", "1")
    assert result.returncode == 0, result.stderr
    assert out.is_file()


def test_convert_unknown_extension_needs_to(table_fits, tmp_path):
    out = tmp_path / "table.dat"
    result = _run_cli("convert", str(table_fits), str(out), "--hdu", "1")
    assert result.returncode == 2, result.stderr
    assert "--to" in result.stderr


def test_info_jsonl_format(image_fits):
    result = _run_cli("info", str(image_fits), "--format", "jsonl")
    assert result.returncode == 0, result.stderr
    line = result.stdout.strip().splitlines()[0]
    payload = json.loads(line)
    assert payload["type"] == "IMAGE"


def test_cutout_box(image_fits, tmp_path):
    out = tmp_path / "box.fits"
    result = _run_cli(
        "cutout", str(image_fits), str(out), "--box", "0,0,2,2", "--hdu", "0"
    )
    assert result.returncode == 0, result.stderr
    cut = torchfits.read_tensor(str(out), hdu=0)
    assert cut.shape == (2, 2)


def test_cutout_cfitsio_section(image_fits, tmp_path):
    out = tmp_path / "section.fits"
    # CFITSIO 1-based inclusive [1:2,1:2] ≈ torchfits --box 0,0,2,2
    sectioned = f"{image_fits}[1:2,1:2]"
    result = _run_cli("cutout", sectioned, str(out), "--hdu", "0")
    assert result.returncode == 0, result.stderr
    cut = torchfits.read_tensor(str(out), hdu=0)
    assert cut.shape == (2, 2)
    expected = torchfits.read_subset(str(image_fits), 0, 0, 0, 2, 2)
    assert torch.allclose(cut, expected)


def test_cutout_rejects_section_and_box(image_fits, tmp_path):
    out = tmp_path / "bad.fits"
    result = _run_cli(
        "cutout",
        f"{image_fits}[1:2,1:2]",
        str(out),
        "--box",
        "0,0,2,2",
    )
    assert result.returncode == 2, result.stderr


def test_open_cfitsio_section_exists(image_fits):
    sectioned = f"{image_fits}[1:2,1:2]"
    with torchfits.open(sectioned) as hdul:
        assert len(hdul) >= 1
    tensor = torchfits.read_tensor(sectioned, hdu=0)
    assert tensor.shape == (2, 2)


def test_convert_parquet(table_fits, tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    out = tmp_path / "table.parquet"
    result = _run_cli(
        "convert",
        str(table_fits),
        str(out),
        "--to",
        "parquet",
        "--hdu",
        "1",
    )
    assert result.returncode == 0, result.stderr
    assert out.is_file()
    assert pq.read_table(str(out)).num_rows == 3


def test_convert_csv_tsv_arrow(table_fits, tmp_path):
    import pyarrow.csv as pacsv
    import pyarrow.feather as feather

    csv_out = tmp_path / "t.csv"
    tsv_out = tmp_path / "t.tsv"
    arrow_out = tmp_path / "t.arrow"
    for path, fmt in ((csv_out, "csv"), (tsv_out, "tsv"), (arrow_out, "arrow")):
        result = _run_cli(
            "convert", str(table_fits), str(path), "--to", fmt, "--hdu", "1"
        )
        assert result.returncode == 0, result.stderr
        assert path.is_file()
    assert pacsv.read_csv(csv_out).num_rows == 3
    assert (
        pacsv.read_csv(
            tsv_out, parse_options=pacsv.ParseOptions(delimiter="\t")
        ).num_rows
        == 3
    )
    assert feather.read_table(arrow_out).num_rows == 3


@pytest.fixture
def table_fits(tmp_path):
    import numpy as np
    from astropy.io import fits
    from astropy.table import Table

    path = tmp_path / "table.fits"
    data = {
        "ra": np.array([200.0, 201.0, 202.0], dtype=np.float64),
        "dec": np.array([45.0, 46.0, 47.0], dtype=np.float64),
        "flux": np.array([1.0, 2.0, 3.0], dtype=np.float32),
    }
    fits.BinTableHDU(Table(data), name="CAT").writeto(str(path), overwrite=True)
    return path


def test_header_keyword_table(image_fits):
    result = _run_cli(
        "header",
        str(image_fits),
        "--keyword-table",
        "-k",
        "BITPIX",
        "-k",
        "NAXIS",
        "-e",
        "0",
    )
    assert result.returncode == 0, result.stderr
    assert "BITPIX" in result.stdout
    assert "NAXIS" in result.stdout


def test_header_rejects_fitsort_alias(image_fits):
    result = _run_cli(
        "header",
        str(image_fits),
        "--fitsort",
        "--keyword",
        "BITPIX",
        "--hdu",
        "0",
    )
    assert result.returncode != 0
    assert "unrecognized" in result.stderr.lower() or "error" in result.stderr.lower()


def test_header_keyword_table_json(image_fits):
    result = _run_cli(
        "header",
        str(image_fits),
        "--keyword-table",
        "-k",
        "BITPIX",
        "-e",
        "0",
        "-f",
        "json",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert int(payload[0]["BITPIX"]) == -32


def test_info_short_format(image_fits):
    result = _run_cli("info", str(image_fits), "-f", "jsonl")
    assert result.returncode == 0, result.stderr
    assert "hdu" in result.stdout


def test_convert_where_filter(table_fits, tmp_path):
    out = tmp_path / "bright.parquet"
    result = _run_cli(
        "convert",
        str(table_fits),
        str(out),
        "-e",
        "1",
        "--where",
        "flux > 1.5",
        "--columns",
        "ra,dec,flux",
    )
    assert result.returncode == 0, result.stderr
    import pyarrow.parquet as pq

    table = pq.read_table(out)
    assert table.num_rows == 2
    assert table.column_names == ["ra", "dec", "flux"]


def test_convert_short_flags(table_fits, tmp_path):
    out = tmp_path / "short.parquet"
    result = _run_cli(
        "convert",
        str(table_fits),
        "-o",
        str(out),
        "-e",
        "1",
        "-w",
        "flux > 1.5",
        "-c",
        "ra,dec,flux",
    )
    assert result.returncode == 0, result.stderr
    import pyarrow.parquet as pq

    assert pq.read_table(out).num_rows == 2


def test_copy_dash_o(image_fits, tmp_path):
    out = tmp_path / "via_o.fits"
    result = _run_cli("copy", str(image_fits), "-o", str(out))
    assert result.returncode == 0, result.stderr
    assert out.is_file()


def test_probe_header_bytes_help():
    result = _run_cli("probe", "--help")
    assert result.returncode == 0
    assert "--header-bytes" in result.stdout


def test_invalid_hdu_is_usage_error(image_fits):
    result = _run_cli("info", str(image_fits), "-e", "not-an-int")
    assert result.returncode == 2, result.stderr


def test_vos_probe_missing_package_message(monkeypatch):
    import builtins

    from torchfits.cli import cmds_probe
    from torchfits.cli.common import UsageError

    real_import = builtins.__import__

    def _block_vos(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "vos" or name.startswith("vos."):
            raise ImportError("vos blocked for test")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _block_vos)
    with pytest.raises(UsageError, match="vos"):
        cmds_probe._probe_vos("vos:example/file.fits", header_bytes=5760)


def test_vos_probe_bad_uri_is_io_error_when_vos_present():
    # Placeholder user only — never a personal vault path in CI/git.
    result = _run_cli("probe", "vos:alice/example.fits", "--json")
    # vos may be installed (CANFAR lab) or missing; either way no traceback.
    assert result.returncode in (2, 3), result.stderr
    assert "vos" in result.stderr.lower() or "service" in result.stderr.lower()


def test_convert_png_invalid_bands_count(image_fits, tmp_path):
    out = tmp_path / "rgb.png"
    result = _run_cli(
        "convert", str(image_fits), str(out), "--to", "png", "--bands", "0,1"
    )
    assert result.returncode == 2, result.stderr
    assert "bands" in result.stderr.lower()


def test_convert_png_invalid_bands_non_integer(image_fits, tmp_path):
    out = tmp_path / "rgb.png"
    result = _run_cli(
        "convert", str(image_fits), str(out), "--to", "png", "--bands", "a,b,c"
    )
    assert result.returncode == 2, result.stderr


def test_keyword_table_with_invalid_hdu(image_fits):
    result = _run_cli(
        "header",
        str(image_fits),
        "--keyword-table",
        "-k",
        "BITPIX",
        "-e",
        "abc",
    )
    assert result.returncode == 2, result.stderr
    assert "hdu" in result.stderr.lower() or "invalid" in result.stderr.lower()


def test_setkey_hierarch_and_rename(image_fits, tmp_path):
    out = tmp_path / "keyed.fits"
    result = _run_cli("copy", str(image_fits), str(out))
    assert result.returncode == 0, result.stderr
    result = _run_cli(
        "setkey",
        str(out),
        "--key",
        "ESO DET CHIP1 ID",
        "--value",
        "42",
    )
    assert result.returncode == 0, result.stderr
    hdr = torchfits.read_header(str(out), 0)
    # HIERARCH cards may surface as key="HIERARCH" with the long name in the value.
    blob = " ".join(f"{k}={v}" for k, v in hdr.items()).upper()
    assert "CHIP1" in blob
    result = _run_cli(
        "setkey",
        str(out),
        "--key",
        "OBJECT",
        "--value",
        "DEMO",
    )
    assert result.returncode == 0, result.stderr
    result = _run_cli("setkey", str(out), "--rename", "OBJECT=TARGET")
    assert result.returncode == 0, result.stderr
    hdr = torchfits.read_header(str(out), 0)
    assert "TARGET" in hdr
    assert hdr["TARGET"] == "DEMO"
    assert "OBJECT" not in hdr


def test_copy_batch_out_dir(image_fits, tmp_path):
    other = tmp_path / "other.fits"
    torchfits.write(
        str(other),
        torch.arange(16, dtype=torch.float32).reshape(4, 4),
        overwrite=True,
    )
    out_dir = tmp_path / "copies"
    result = _run_cli(
        "copy",
        str(image_fits),
        str(other),
        "--out-dir",
        str(out_dir),
        "-J",
        "2",
    )
    assert result.returncode == 0, result.stderr
    assert (out_dir / image_fits.name).is_file()
    assert (out_dir / other.name).is_file()


def test_stats_std_median_json(image_fits):
    result = _run_cli("stats", str(image_fits), "--json")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert "std" in payload[0]
    assert "median" in payload[0]
    tensor = torchfits.read_tensor(str(image_fits), hdu=0).float().reshape(-1)
    assert payload[0]["std"] == pytest.approx(float(tensor.std(unbiased=False)))
    assert payload[0]["median"] == pytest.approx(float(tensor.median()))


def test_compress_algorithm_gzip(image_fits, tmp_path):
    out = tmp_path / "gzip.fits"
    result = _run_cli(
        "compress",
        str(image_fits),
        str(out),
        "--algorithm",
        "GZIP_1",
        "-j",
        "1",
    )
    assert result.returncode == 0, result.stderr
    assert out.is_file()
    hdr = torchfits.read_header(str(out), 1)
    zimage = str(hdr.get("ZIMAGE", "")).upper()
    zcmptype = str(hdr.get("ZCMPTYPE", "")).upper()
    assert zimage in {"T", "TRUE", "1"} or "GZIP" in zcmptype


def test_header_keyword_wildcard(image_fits):
    result = _run_cli("header", str(image_fits), "--keyword", "NAXIS*", "--json")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    keys = {row["keyword"] for row in payload}
    assert "NAXIS" in keys
    assert "NAXIS1" in keys
    assert "NAXIS2" in keys


def test_setkey_delete_and_at_list(image_fits, tmp_path):
    a = tmp_path / "a.fits"
    b = tmp_path / "b.fits"
    torchfits.write(
        str(a),
        torch.ones(2, 2),
        header={"OBJECT": "A", "FOO": 1},
        overwrite=True,
    )
    torchfits.write(
        str(b),
        torch.ones(2, 2),
        header={"OBJECT": "B", "FOO": 2},
        overwrite=True,
    )
    list_file = tmp_path / "paths.txt"
    list_file.write_text(f"{a}\n# comment\n{b}\n", encoding="utf-8")
    out_dir = tmp_path / "edited"
    result = _run_cli(
        "setkey",
        f"@{list_file}",
        "--delete",
        "FOO",
        "--out-dir",
        str(out_dir),
        "-J",
        "2",
    )
    assert result.returncode == 0, result.stderr
    for name in (a.name, b.name):
        hdr = torchfits.read_header(str(out_dir / name), 0)
        assert "FOO" not in hdr
        assert "OBJECT" in hdr


def test_setkey_delete_preserves_tile_compression(tmp_path):
    path = tmp_path / "rice.fits"
    torchfits.write(
        str(path),
        torch.arange(16, dtype=torch.float32).reshape(4, 4),
        header={"OBJECT": "KEEP", "FOO": 1},
        overwrite=True,
        compress="RICE_1",
    )
    before = torchfits.read_header(str(path), 1)
    assert str(before.get("ZCMPTYPE", "")).upper().startswith("RICE")
    result = _run_cli("setkey", str(path), "--delete", "FOO", "-e", "1")
    assert result.returncode == 0, result.stderr
    after = torchfits.read_header(str(path), 1)
    assert "FOO" not in after
    assert after.get("OBJECT") == "KEEP"
    assert str(after.get("ZCMPTYPE", "")).upper().startswith("RICE")
    assert str(after.get("XTENSION", "")).upper() == "BINTABLE"


def test_setkey_rejects_duplicate_inplace_paths(image_fits):
    result = _run_cli(
        "setkey",
        str(image_fits),
        str(image_fits),
        "--key",
        "OBJECT",
        "--value",
        "X",
        "-J",
        "2",
    )
    assert result.returncode == 2, result.stderr
    assert "duplicate" in result.stderr.lower()


def test_compress_split_hdu_rejects_stem_collision(tmp_path):
    a = tmp_path / "same.fits"
    b = tmp_path / "same.fit"
    data = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    torchfits.write(str(a), data, overwrite=True)
    torchfits.write(str(b), data + 1, overwrite=True)
    result = _run_cli(
        "compress",
        str(a),
        str(b),
        "--split",
        "hdu",
        "--out-dir",
        str(tmp_path / "out"),
        "-J",
        "1",
    )
    assert result.returncode == 2, result.stderr
    assert "stem" in result.stderr.lower()


def test_cutout_batch_out_dir_strips_section(image_fits, tmp_path):
    out_dir = tmp_path / "cuts"
    other = tmp_path / "other.fits"
    torchfits.write(
        str(other),
        torch.arange(16, dtype=torch.float32).reshape(4, 4),
        overwrite=True,
    )
    result = _run_cli(
        "cutout",
        f"{image_fits}[1:2,1:2]",
        f"{other}[1:2,1:2]",
        "--out-dir",
        str(out_dir),
        "-J",
        "1",
    )
    assert result.returncode == 0, result.stderr
    assert (out_dir / image_fits.name).is_file()
    assert (out_dir / other.name).is_file()
    assert not any("[" in p.name for p in out_dir.iterdir())


def test_setkey_rejects_negative_hdu_index(image_fits):
    result = _run_cli(
        "setkey",
        str(image_fits),
        "--key",
        "OBJECT",
        "--value",
        "DEMO",
        "--hdu",
        "-1",
    )
    assert result.returncode == 2, result.stderr
    assert "hdu" in result.stderr.lower()


def test_transform_rejects_private_name(image_fits, tmp_path):
    out = tmp_path / "transformed.fits"
    result = _run_cli(
        "transform",
        str(image_fits),
        "--name",
        "_fit_poly_continuum",
        "-o",
        str(out),
    )
    assert result.returncode != 0
    assert "unknown transform" in result.stderr.lower()


def test_transform_default_constructor(image_fits, tmp_path):
    out = tmp_path / "transformed.fits"
    result = _run_cli(
        "transform",
        str(image_fits),
        "--name",
        "ArcsinhStretch",
        "-o",
        str(out),
    )
    assert result.returncode == 0, result.stderr
    assert out.exists()


def test_transform_constructor_kwargs(image_fits, tmp_path):
    out = tmp_path / "transformed.fits"
    result = _run_cli(
        "transform",
        str(image_fits),
        "--name",
        "ArcsinhStretch:a=2.0",
        "-o",
        str(out),
    )
    assert result.returncode == 0, result.stderr
    assert out.exists()


def test_transform_rejects_unknown_kwarg(image_fits, tmp_path):
    out = tmp_path / "transformed.fits"
    result = _run_cli(
        "transform",
        str(image_fits),
        "--name",
        "ArcsinhStretch:bogus=1",
        "-o",
        str(out),
    )
    assert result.returncode == 2, result.stderr
    assert "unknown kwarg" in result.stderr.lower()


def test_http_probe_blocks_internal_ssrf():
    result = _run_cli("probe", "http://127.0.0.1:8000/latest/meta-data/")
    assert result.returncode == 3
    assert "access to internal or private networks is blocked" in result.stderr

    result = _run_cli("probe", "http://localhost:8080/")
    assert result.returncode == 3
    assert "access to internal or private networks is blocked" in result.stderr

    result = _run_cli("probe", "http://169.254.169.254/")
    assert result.returncode == 3
    assert "access to internal or private networks is blocked" in result.stderr


def test_cli_rejects_ftp_remote_paths():
    """CLI local-file commands must reject ftp:// (not only http/https/vos)."""
    result = _run_cli("info", "ftp://example.com/x.fits")
    assert result.returncode == 3
    assert "remote paths are not supported" in result.stderr
