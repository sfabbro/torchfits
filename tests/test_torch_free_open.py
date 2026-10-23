"""``torchfits.open`` is a header-only path and must not import PyTorch.

The HDU list and every header come from ``torchfits._core``, which does not
link libtorch.  The tensor-backed extension (and therefore the Python ``torch``
runtime) is imported only when an HDU touches its data.  The check runs in a
subprocess because ``torch`` is already imported in the test process.
"""

import json
import subprocess
import sys
import textwrap

import torch

import torchfits


def test_open_headers_and_info_do_not_import_torch(tmp_path):
    image = tmp_path / "image.fits"
    table = tmp_path / "table.fits"
    torchfits.write(str(image), torch.zeros(4, 4))
    torchfits.table.write(str(table), {"A": torch.arange(5, dtype=torch.int32)})

    code = textwrap.dedent(
        f"""
        import io, json, sys
        import torchfits

        out = {{}}
        with torchfits.open({str(image)!r}) as hdul:
            hdul.info(io.StringIO())
            out["image_naxis"] = hdul[0].header["NAXIS"]
            out["image_shape"] = hdul[0].shape_str
            out["image_torch"] = "torch" in sys.modules

        with torchfits.open({str(table)!r}) as hdul:
            hdul.info(io.StringIO())
            out["table_columns"] = list(hdul[1].columns)
            out["table_rows"] = hdul[1].num_rows
            out["table_torch"] = "torch" in sys.modules

        out["data_before"] = "torch" in sys.modules
        with torchfits.open({str(image)!r}) as hdul:
            hdul[0].to_tensor()
        out["data_after"] = "torch" in sys.modules
        print(json.dumps(out))
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout.strip().splitlines()[-1])

    assert result["image_torch"] is False
    assert result["table_torch"] is False
    assert result["data_before"] is False
    assert result["data_after"] is True
    assert result["image_naxis"] == 2
    assert result["image_shape"] == "(4, 4)"
    assert result["table_columns"] == ["A"]
    assert result["table_rows"] == 5
