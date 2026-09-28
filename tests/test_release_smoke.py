from __future__ import annotations

from importlib import metadata
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

import torchfits
import torchfits._C
import torchfits._core as core


def _declared_version() -> str:
    try:
        return metadata.version("torchfits")
    except metadata.PackageNotFoundError:
        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib

        with pyproject.open("rb") as fh:
            data = tomllib.load(fh)
        return str(data["project"]["version"])


def test_runtime_version_matches_declared_version() -> None:
    assert torchfits.__version__ == _declared_version()


def test_release_smoke_image_read_write_roundtrip() -> None:
    image = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    header = {"OBJECT": "SMOKE"}

    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as fh:
        path = Path(fh.name)

    try:
        torchfits.write(path, image, header=header, overwrite=True)
        out, hdr = torchfits.read(str(path), return_header=True)

        assert isinstance(out, torch.Tensor)
        assert out.shape == (4, 4)
        assert torch.allclose(out.cpu(), image)
        assert str(hdr["OBJECT"]).strip() == "SMOKE"

        with torchfits.open(str(path)) as hdul:
            reopened = hdul[0].to_tensor()
            assert torch.allclose(reopened.cpu(), image)
    finally:
        Path(path).unlink(missing_ok=True)


def test_release_smoke_table_read() -> None:
    table = {
        "RA": np.array([10.1, 10.2, 10.3], dtype=np.float64),
        "ID": np.array([1, 2, 3], dtype=np.int64),
    }

    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as fh:
        path = fh.name

    try:
        torchfits.write(path, table, overwrite=True)
        data = torchfits.table.read_torch(path, hdu=1, columns=["RA", "ID"])
        hdu = torchfits.TableHDU.from_fits(path, hdu_index=1)

        assert set(data.keys()) == {"RA", "ID"}
        assert torch.allclose(
            data["RA"].cpu(), torch.tensor([10.1, 10.2, 10.3], dtype=torch.float64)
        )
        assert torch.equal(data["ID"].cpu(), torch.tensor([1, 2, 3], dtype=torch.int64))
        assert hdu.num_rows == 3
        assert set(hdu.col_names) == {"RA", "ID"}
    finally:
        Path(path).unlink(missing_ok=True)


def test_release_smoke_core_library_is_present_and_consistent() -> None:
    """The split must survive packaging, not just the source tree.

    A wheel can pass every functional test above and still be broken for
    metadata: ``libtorchfits_core`` missing, ``_core`` unable to dlopen it, or
    the extension and the library stamped from different builds. The first
    fails at import, the second at the first metadata call, and the third is a
    cross-version call into a C library whose symptom is a segfault -- so the
    build-id agreement is checked here rather than trusted.
    """
    assert core.TORCH_FREE is True
    assert core.core_library_build_id() == core.__build_id__
    assert torchfits._C.__core_build_id__ == core.__build_id__

    library = core.core_library_build_id()
    assert library, "the build id must not be empty"
    # _core must resolve against the library, not have it statically absorbed:
    # the whole point is one CFITSIO in the process, shared with _C.
    assert Path(core.__file__).name.startswith("_core")


def test_release_smoke_metadata_answers_from_the_core() -> None:
    """A real metadata round trip on an installed wheel, with torch unimported.

    The child blocks ``torch`` and ``numpy`` outright, so this fails if any
    part of the metadata path reaches for them -- and it also exercises the
    dlopen of ``libtorchfits_core`` from an installed wheel layout, which is
    where a missing sibling rpath shows up.
    """
    import json
    import subprocess

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "smoke.fits")
        torchfits.write(
            path,
            torch.arange(16, dtype=torch.float32).reshape(4, 4),
            header={"OBJECT": "COREMETA"},
            overwrite=True,
        )
        script = """
import importlib.abc, json, sys


class _Block(importlib.abc.MetaPathFinder):
    BLOCKED = ("torch", "numpy")

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root in self.BLOCKED:
            raise ImportError(root + " is blocked: " + fullname)
        return None


sys.meta_path.insert(0, _Block())

import torchfits
import torchfits._core as core

print(json.dumps({
    "object": torchfits.read_header(sys.argv[1], 0)["OBJECT"],
    "shape": list(torchfits.read_shape(sys.argv[1], 0)[1]),
    "core_shape": list(core.read_shape(sys.argv[1], 0)[1]),
    "torch_imported": "torch" in sys.modules,
    "build_ids_match": core.core_library_build_id() == core.__build_id__,
}))
"""
        result = subprocess.run(
            [sys.executable, "-c", script, path],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        assert payload["torch_imported"] is False
        assert payload["build_ids_match"] is True
        assert payload["shape"] == payload["core_shape"] == [4, 4]
        assert str(payload["object"]).strip() == "COREMETA"
