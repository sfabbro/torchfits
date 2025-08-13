import json
import os
import sys
from pathlib import Path

import pytest

EX_FILE = Path("examples/basic_example.fits")


@pytest.mark.skipif(not EX_FILE.exists(), reason="example FITS not present")
def test_bench_mmap_script_runs(tmp_path):
    out = tmp_path / "bench_mmap.jsonl"
    script = Path("scripts/bench/bench_mmap_vs_default.py")
    assert script.exists(), "benchmark script missing"

    rc = os.system(f"{sys.executable} {script} --input {EX_FILE} --output {out}")
    assert rc == 0, "benchmark script failed"

    assert out.exists()
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2
    modes = {json.loads(line)["mode"] for line in lines}
    assert modes == {"default", "mmap"}
