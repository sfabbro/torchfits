import json
import os
import sys
from pathlib import Path

import pytest
import torch

import torchfits as tf

EX_FILE = Path("examples/basic_example.fits")


@pytest.mark.skipif(not EX_FILE.exists(), reason="example FITS not present")
def test_read_buffered_flag_roundtrip():
    # Default read
    data_def, hdr_def = tf.read(str(EX_FILE), hdu=0)
    # Buffered fast-path
    data_buf, hdr_buf = tf.read(str(EX_FILE), hdu=0, enable_buffered=True)

    assert isinstance(data_def, torch.Tensor)
    assert isinstance(data_buf, torch.Tensor)
    assert data_def.shape == data_buf.shape
    # Exact equality for example file
    assert torch.equal(data_def, data_buf)
    # Headers should be dict-like; compare a stable key if both present
    if isinstance(hdr_def, dict) and isinstance(hdr_buf, dict):
        assert hdr_def.get("SIMPLE") == hdr_buf.get("SIMPLE")


@pytest.mark.skipif(not EX_FILE.exists(), reason="example FITS not present")
def test_bench_buffered_script_runs(tmp_path, monkeypatch):
    # Run the micro-benchmark script and verify outputs contain both modes
    out = tmp_path / "bench_buffered.jsonl"
    # Build command to run the script with explicit --input for determinism
    script = Path("scripts/bench/bench_buffered_vs_default.py")
    assert script.exists(), "benchmark script missing"

    # Use same interpreter to avoid env mismatch
    rc = os.system(f"{sys.executable} {script} --input {EX_FILE} --output {out}")
    assert rc == 0, "benchmark script failed"

    assert out.exists()
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2
    modes = {json.loads(line)["mode"] for line in lines}
    assert modes == {"default", "buffered"}
