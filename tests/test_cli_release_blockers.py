from __future__ import annotations

import subprocess
import sys
import time
from io import BytesIO
from types import SimpleNamespace

import pytest
import torch

import torchfits
import torchfits.cli.cmds_probe as probe_module
from torchfits.cli.cmds_probe import (
    _call_with_deadline,
    _complete_header_text,
    _probe_http_now,
    _probe_vos_now,
)
from torchfits.cli.cmds_transform import CLI_TRANSFORM_NAMES, _build_transform
from torchfits.cli.common import IoError
from torchfits.hdu import Header


def _run_cli(
    *args: str, input_text: str | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "torchfits.cli", *args],
        input=input_text,
        capture_output=True,
        text=True,
        check=False,
    )


def _fake_slow_http() -> dict[str, str]:
    while True:
        time.sleep(0.01)


def _fake_slow_vos() -> dict[str, str]:
    while True:
        time.sleep(0.01)


def test_root_version() -> None:
    result = _run_cli("--version")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"torchfits {torchfits.__version__}"


def test_transform_help_lists_registry() -> None:
    result = _run_cli("transform", "--help")
    assert result.returncode == 0, result.stderr
    for name in CLI_TRANSFORM_NAMES:
        assert name in result.stdout


@pytest.mark.parametrize("name", CLI_TRANSFORM_NAMES)
def test_every_registered_transform_constructs(name: str) -> None:
    built_name, transform = _build_transform(name)
    assert built_name == name
    assert callable(transform)


@pytest.mark.parametrize("name", ["safe_log", "rgb", "Compose"])
def test_non_registered_transform_is_usage_error(tmp_path, name: str) -> None:
    source = tmp_path / "source.fits"
    torchfits.write(source, torch.ones(2, 2), overwrite=True)
    result = _run_cli(
        "transform",
        str(source),
        "--name",
        name,
        "-o",
        str(tmp_path / "out.fits"),
    )
    assert result.returncode == 2
    assert "choose from" in result.stderr
    assert "Traceback" not in result.stderr


def test_transform_constructor_error_is_usage_error(tmp_path) -> None:
    source = tmp_path / "source.fits"
    torchfits.write(source, torch.ones(2, 2), overwrite=True)
    result = _run_cli(
        "transform",
        str(source),
        "--name",
        "ArcsinhStretch:a=0",
        "-o",
        str(tmp_path / "out.fits"),
    )
    assert result.returncode == 2
    assert "invalid ArcsinhStretch constructor" in result.stderr
    assert "Traceback" not in result.stderr


def test_default_float_compression_is_lossless(tmp_path) -> None:
    source = tmp_path / "source.fits"
    output = tmp_path / "compressed.fits"
    data = torch.randn(19, 23, generator=torch.Generator().manual_seed(17)) * 1.0e4
    torchfits.write(source, data, overwrite=True)
    result = _run_cli("compress", str(source), str(output))
    assert result.returncode == 0, result.stderr
    with torchfits.open(output) as hdul:
        image_indices = [
            i for i, hdu in enumerate(hdul) if int(hdu.header.get("NAXIS", 0))
        ]
    assert len(image_indices) == 1
    actual = torchfits.read_tensor(output, hdu=image_indices[0])
    assert torch.equal(actual, data)


def test_compression_codec_validation_and_help(tmp_path) -> None:
    source = tmp_path / "source.fits"
    torchfits.write(source, torch.ones(2, 2), overwrite=True)
    invalid = _run_cli(
        "compress", str(source), str(tmp_path / "bad.fits"), "--algorithm", "TYPO"
    )
    assert invalid.returncode == 2
    float_plio = _run_cli(
        "compress",
        str(source),
        str(tmp_path / "plio.fits"),
        "--algorithm",
        "PLIO_1",
    )
    assert float_plio.returncode == 2
    assert "does not support floating-point" in float_plio.stderr
    compressed = tmp_path / "gzip.fits"
    gzip = _run_cli("compress", str(source), str(compressed), "--algorithm", "GZIP_1")
    assert gzip.returncode == 0, gzip.stderr
    compressed_plio = _run_cli(
        "compress",
        str(compressed),
        str(tmp_path / "compressed-plio.fits"),
        "--algorithm",
        "PLIO_1",
    )
    assert compressed_plio.returncode == 2
    assert "does not support floating-point" in compressed_plio.stderr
    help_result = _run_cli("compress", "--help")
    assert help_result.returncode == 0
    for codec in ("GZIP_1", "GZIP_2", "RICE_1", "HCOMPRESS_1", "PLIO_1"):
        assert codec in help_result.stdout
    assert "lossless" in help_result.stdout
    assert "lossy" in help_result.stdout


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (("--bands", "0"), "--bands applies only to png convert"),
        (("--recipe", "auto"), "--recipe applies only to png convert"),
        (("--brightness", "0.2"), "--brightness applies only to png convert"),
        (("--saturation", "1"), "--saturation applies only to png convert"),
        (("--calibrated",), "--calibrated applies only to png convert"),
        (("--zeropoints", "25"), "--zeropoints applies only to png convert"),
        (("--q", "4"), "--q applies only to png convert"),
        (("--stretch", "0.5"), "--stretch applies only to png convert"),
    ],
)
def test_table_convert_rejects_png_options(tmp_path, extra, message: str) -> None:
    source = tmp_path / "table.fits"
    torchfits.table.write(source, {"x": [1, 2]}, overwrite=True)
    result = _run_cli(
        "convert",
        str(source),
        "-o",
        str(tmp_path / "out.csv"),
        "--to",
        "csv",
        *extra,
    )
    assert result.returncode == 2
    assert message in result.stderr


def test_other_inapplicable_options_are_rejected(tmp_path) -> None:
    source = tmp_path / "source.fits"
    torchfits.write(source, torch.ones(2, 2), overwrite=True)
    png_hdu = _run_cli(
        "convert",
        str(source),
        "-o",
        str(tmp_path / "out.png"),
        "--to",
        "png",
        "--hdu",
        "0",
    )
    assert png_hdu.returncode == 2
    assert "--hdu applies only to table convert" in png_hdu.stderr
    for command in ("compress", "decompress"):
        result = _run_cli(
            command,
            str(source),
            str(tmp_path / f"{command}.fits"),
            "--hdu",
            "0",
        )
        assert result.returncode == 2
        assert "--hdu applies only with --split hdu" in result.stderr
    arith = _run_cli(
        "arith",
        str(source),
        "--op",
        "add",
        "--value",
        "1",
        "--hdu2",
        "0",
        "-o",
        str(tmp_path / "arith.fits"),
    )
    assert arith.returncode == 2
    assert "--hdu2 applies only to image operand B" in arith.stderr
    for extra, message in (
        (("--header-bytes", "2880"), "--header-bytes applies only to remote"),
        (("--timeout", "1"), "--timeout applies only to remote"),
    ):
        probe = _run_cli("probe", str(source), *extra)
        assert probe.returncode == 2
        assert message in probe.stderr
    remote_hdu = _run_cli("probe", "https://example.invalid/a.fits", "--hdu", "0")
    assert remote_hdu.returncode == 2
    assert "--hdu applies only to local probe" in remote_hdu.stderr


def test_remote_header_parser_reads_through_second_block() -> None:
    cards = [f"KEY{i:05d}= {i:20d}".ljust(80) for i in range(40)]
    cards[0] = "SIMPLE  =                    T".ljust(80)
    cards.extend(["BITPIX  =                   16".ljust(80), "END".ljust(80)])
    chunk = "".join(cards).ljust(5760).encode("ascii")
    text = _complete_header_text(chunk, "remote.fits")
    assert len(text) > 2880
    assert "BITPIX" in text


def _multi_block_header() -> bytes:
    cards = [f"KEY{i:05d}= {i:20d}".ljust(80) for i in range(38)]
    cards[0] = "SIMPLE  =                    T".ljust(80)
    cards.extend(
        [
            "BITPIX  =                   16".ljust(80),
            "NAXIS   =                    2".ljust(80),
            "NAXIS1  =                   11".ljust(80),
            "NAXIS2  =                   13".ljust(80),
            "END".ljust(80),
        ]
    )
    return "".join(cards).ljust(5760).encode("ascii")


def test_http_probe_parses_multi_block_header(monkeypatch) -> None:
    chunk = _multi_block_header()
    monkeypatch.setattr(
        probe_module, "http_open", lambda *args, **kwargs: BytesIO(chunk)
    )
    record = _probe_http_now(
        "https://public.example/a.fits", header_bytes=5760, timeout=1
    )
    assert record["bitpix"] == 16
    assert record["naxis2"] == 13


def test_vos_probe_parses_multi_block_header(monkeypatch) -> None:
    chunk = _multi_block_header()

    class Client:
        def open(self, uri, mode):
            return BytesIO(chunk)

    real_import = probe_module.importlib.import_module
    monkeypatch.setattr(
        probe_module.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(Client=Client) if name == "vos" else real_import(name)
        ),
    )
    record = _probe_vos_now("vos:example/a.fits", header_bytes=5760)
    assert record["bitpix"] == 16
    assert record["naxis2"] == 13


@pytest.mark.parametrize("operation", [_fake_slow_http, _fake_slow_vos])
def test_remote_probe_deadline_stops_periodic_progress(operation) -> None:
    started = time.monotonic()
    with pytest.raises(IoError, match="timed out after 0.1 seconds"):
        _call_with_deadline(operation, deadline_seconds=0.1)
    assert time.monotonic() - started < 1.0


def test_remote_header_parser_reports_truncation() -> None:
    chunk = ("SIMPLE  =                    T".ljust(80) * 72).encode("ascii")
    with pytest.raises(IoError, match="truncated before END"):
        _complete_header_text(chunk, "remote.fits")


def test_diff_compares_repeated_cards_comments_and_order(tmp_path) -> None:
    cases = [
        (
            Header([("HISTORY", "first-a", ""), ("HISTORY", "same-last", "")]),
            Header([("HISTORY", "first-b", ""), ("HISTORY", "same-last", "")]),
        ),
        (Header([("A", 1, "left")]), Header([("A", 1, "right")])),
        (
            Header([("A", 1, ""), ("B", 2, "")]),
            Header([("B", 2, ""), ("A", 1, "")]),
        ),
    ]
    for index, (header_a, header_b) in enumerate(cases):
        path_a = tmp_path / f"a{index}.fits"
        path_b = tmp_path / f"b{index}.fits"
        torchfits.write(path_a, torch.ones(2, 2), header=header_a, overwrite=True)
        torchfits.write(path_b, torch.ones(2, 2), header=header_b, overwrite=True)
        result = _run_cli("diff", str(path_a), str(path_b))
        assert result.returncode == 1, (index, result.stderr)
        assert "card" in result.stderr


def test_diff_compares_duplicate_ordinary_cards(tmp_path) -> None:
    path_a = tmp_path / "a.fits"
    path_b = tmp_path / "b.fits"
    torchfits.write(path_a, torch.ones(2, 2), overwrite=True)
    torchfits.write(path_b, torch.ones(2, 2), overwrite=True)

    def inject(path, values: tuple[int, int]) -> None:
        data = path.read_bytes()
        header = data[:2880]
        end = header.index(b"END     ")
        cards = b"".join(
            f"DUP     = {value:20d}".ljust(80).encode("ascii") for value in values
        )
        shifted = header[:end] + cards + header[end : 2880 - len(cards)]
        path.write_bytes(shifted + data[2880:])

    inject(path_a, (1, 2))
    inject(path_b, (3, 2))
    result = _run_cli("diff", str(path_a), str(path_b))
    assert result.returncode == 1, result.stderr
    assert "DUP" in result.stderr


def test_setkey_stdin_only_and_empty_stdin(tmp_path) -> None:
    source = tmp_path / "source.fits"
    torchfits.write(source, torch.ones(2, 2), overwrite=True)
    result = _run_cli(
        "setkey", "--stdin", "--key", "TEST", "--value", "7", input_text=f"{source}\n"
    )
    assert result.returncode == 0, result.stderr
    assert torchfits.read_header(source, 0)["TEST"] == 7
    empty = _run_cli(
        "setkey", "--stdin", "--key", "TEST", "--value", "8", input_text=""
    )
    assert empty.returncode == 2
    assert "no input paths" in empty.stderr
