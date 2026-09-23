"""``setkey`` integrity: no destructive edits, no partial edits, no card churn.

Guards the ``setkey-no-rewrite`` contract (CFITSIO card update/delete only)
and its edge semantics: commentary cards survive an edit exactly once,
``--rename`` never destroys data, failed batches leave the file untouched, and
``--out``/``--out-dir`` destinations that resolve to the input file edit it in
place instead of crashing in ``shutil.copy2``.
"""

from __future__ import annotations

import subprocess
import sys

import torch

import torchfits
from torchfits.hdu import Header


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "torchfits.cli", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _make_header() -> Header:
    hdr = Header({"AKEY": 1, "BKEY": (2, "keep me"), "XKEY": 3})
    hdr.add_history("h1")
    hdr.add_history("h2")
    hdr.add_comment("c1")
    hdr.add_comment("c2")
    return hdr


def _commentary(path: str) -> tuple[list[str], list[str]]:
    header = torchfits.read_header(path, 0)
    history = [c.value for c in header.cards if c.key == "HISTORY"]
    comment = [c.value for c in header.cards if c.key == "COMMENT"]
    return history, comment


def test_setkey_edit_preserves_history_and_comment_exactly(tmp_path):
    img = tmp_path / "img.fits"
    torchfits.write(str(img), torch.zeros(2, 2), header=_make_header(), overwrite=True)
    before = _commentary(str(img))
    result = _run_cli("setkey", str(img), "-k", "NEWKEY", "--value", "1")
    assert result.returncode == 0, result.stderr
    assert _commentary(str(img)) == before


def test_setkey_rename_preserves_history_and_comment_exactly(tmp_path):
    img = tmp_path / "img.fits"
    torchfits.write(str(img), torch.zeros(2, 2), header=_make_header(), overwrite=True)
    before = _commentary(str(img))
    result = _run_cli("setkey", str(img), "--rename", "AKEY=RENAMED")
    assert result.returncode == 0, result.stderr
    assert _commentary(str(img)) == before


def test_setkey_rename_preserves_card_comment(tmp_path):
    img = tmp_path / "img.fits"
    torchfits.write(str(img), torch.zeros(2, 2), header=_make_header(), overwrite=True)
    result = _run_cli("setkey", str(img), "--rename", "BKEY=RENAMED")
    assert result.returncode == 0, result.stderr
    header = torchfits.read_header(str(img), 0)
    renamed = [c for c in header.cards if c.key == "RENAMED"]
    assert renamed and renamed[0].comment == "keep me"


def test_setkey_rename_same_name_refused(tmp_path):
    img = tmp_path / "img.fits"
    torchfits.write(str(img), torch.zeros(2, 2), header=_make_header(), overwrite=True)
    before = img.read_bytes()
    result = _run_cli("setkey", str(img), "--rename", "XKEY=XKEY")
    assert result.returncode == 2, result.stderr
    assert img.read_bytes() == before
    assert "XKEY" in torchfits.read_header(str(img), 0)


def test_setkey_rename_onto_existing_target_refused(tmp_path):
    img = tmp_path / "img.fits"
    torchfits.write(str(img), torch.zeros(2, 2), header=_make_header(), overwrite=True)
    before = img.read_bytes()
    result = _run_cli("setkey", str(img), "--rename", "AKEY=BKEY")
    assert result.returncode == 2, result.stderr
    assert img.read_bytes() == before
    header = torchfits.read_header(str(img), 0)
    assert header.get("BKEY") == 2
    assert header.get("AKEY") == 1


def test_setkey_non_integer_hdu_is_usage_error(tmp_path):
    img = tmp_path / "img.fits"
    torchfits.write(str(img), torch.zeros(2, 2), overwrite=True)
    result = _run_cli("setkey", str(img), "--hdu", "z", "-k", "N", "--value", "1")
    assert result.returncode == 2, result.stderr


def test_setkey_failed_batch_leaves_file_untouched(tmp_path):
    img = tmp_path / "img.fits"
    torchfits.write_tensor(str(img), torch.zeros(2, 2), overwrite=True)
    torchfits.insert_hdu(str(img), torch.ones(2, 2), index=1)
    torchfits.io._write_header_cards_if_supported(str(img), 0, {"AKEY": 1})
    torchfits.io._write_header_cards_if_supported(str(img), 1, {"AKEY": 2})
    before = img.read_bytes()
    # HDU 0's edits are valid; HDU 1 lacks MISSING -> the whole batch must roll
    # back (no partially edited file on error).
    result = _run_cli(
        "setkey", str(img), "-e", "all", "--delete", "AKEY", "--delete", "MISSING"
    )
    assert result.returncode != 0, result.stderr
    assert img.read_bytes() == before
    assert "AKEY" in torchfits.read_header(str(img), 0)
    assert "AKEY" in torchfits.read_header(str(img), 1)


def test_setkey_out_dir_resolving_to_input_edits_in_place(tmp_path):
    img = tmp_path / "img.fits"
    torchfits.write(str(img), torch.zeros(2, 2), header={"AKEY": 1}, overwrite=True)
    link = tmp_path / "link"
    link.symlink_to(tmp_path)
    result = _run_cli(
        "setkey", str(img), "--out-dir", str(link), "-k", "ZKEY", "--value", "7"
    )
    assert result.returncode == 0, result.stderr
    header = torchfits.read_header(str(img), 0)
    assert header.get("ZKEY") == 7
    assert header.get("AKEY") == 1


def test_setkey_value_requires_key(tmp_path):
    img = tmp_path / "img.fits"
    torchfits.write(str(img), torch.zeros(2, 2), header={"AKEY": 1}, overwrite=True)
    before = img.read_bytes()
    result = _run_cli("setkey", str(img), "--value", "5", "--delete", "AKEY")
    assert result.returncode == 2, result.stderr
    assert img.read_bytes() == before


def test_setkey_key_requires_value_even_with_rename(tmp_path):
    img = tmp_path / "img.fits"
    torchfits.write(str(img), torch.zeros(2, 2), header=_make_header(), overwrite=True)
    before = img.read_bytes()
    result = _run_cli("setkey", str(img), "-k", "NEWKEY", "--rename", "AKEY=RENAMED")
    assert result.returncode == 2, result.stderr
    assert img.read_bytes() == before
