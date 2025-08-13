import os
import random
import tempfile

import pytest
import torch

import torchfits as tf


@pytest.mark.parametrize(
    "rows,max_len,seed",
    [
        (5, 32, 1),
        (12, 64, 7),
    ],
)
def test_varlen_write_fuzz_astropy_verify(rows, max_len, seed):
    rng = random.Random(seed)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "varlen_fuzz.fits")
        arrays = []
        for _ in range(rows):
            n = rng.randint(1, max_len)
            arrays.append(torch.linspace(0, 1, n, dtype=torch.float32))
        tf.write_variable_length_array(
            path, arrays, header={"EXTNAME": "VLA"}, overwrite=True
        )

        # Verify header basics
        hdr = tf.get_header(path, hdu=1)
        assert int(hdr.get("TFIELDS", 0)) == 1
        assert hdr.get("TFORM1", "").startswith("1P")

        # Data verification via astropy fallback
        try:
            from astropy.io import fits
        except Exception:
            pytest.skip("astropy not available for varlen verification")
        with fits.open(path) as hdul:
            data = hdul[1].data
            assert len(data) == rows
