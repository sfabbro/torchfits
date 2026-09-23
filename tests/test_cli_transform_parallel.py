"""``transform -J`` determinism (decision 5).

``-J`` gives every file its own transform instance, so parallel runs must
produce byte-identical outputs to the serial run even for state-carrying
(``DataState``-bearing) transforms such as ``ZScaleNormalize`` (stores
``_last_state`` zscale limits per call) and ``RobustNormalize`` /
``MeshBackgroundSubtract`` (store ``_last_med``/``_last_std``/``_last_bg``).
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
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


@pytest.mark.parametrize(
    "name",
    ["ZScaleNormalize", "RobustNormalize", "MeshBackgroundSubtract"],
)
def test_transform_parallel_matches_serial(tmp_path, name):
    inputs = []
    rng = torch.Generator().manual_seed(7)
    for i in range(4):
        path = tmp_path / f"t{i}.fits"
        data = torch.randn(16, 16, generator=rng) * (i + 1) + 3 * i + 10
        torchfits.write(str(path), data, overwrite=True)
        inputs.append(str(path))

    out1 = tmp_path / "j1"
    out4 = tmp_path / "j4"
    out1.mkdir()
    out4.mkdir()
    r1 = _run_cli(
        "transform", *inputs, "--name", name, "--out-dir", str(out1), "-J", "1"
    )
    r4 = _run_cli(
        "transform", *inputs, "--name", name, "--out-dir", str(out4), "-J", "4"
    )
    assert r1.returncode == 0, r1.stderr
    assert r4.returncode == 0, r4.stderr

    for i in range(4):
        serial = torchfits.read_tensor(str(out1 / f"t{i}.fits"), hdu=0).numpy()
        parallel = torchfits.read_tensor(str(out4 / f"t{i}.fits"), hdu=0).numpy()
        np.testing.assert_array_equal(serial, parallel)
