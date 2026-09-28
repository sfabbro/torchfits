"""Real CFHT observations through the torch-free core.

``tests/test_core_library.py`` proves the split is *correct*: ``_core`` and
``_C`` must never answer differently. It proves that on synthetic files,
because a synthetic file is the only kind whose expected values can be written
down by hand.

Real observations are the other half, and they are strictly harder input:

* the MegaPipe mosaics are single-HDU 1.6 GB files of 21404x20347 pixels, so
  every shape question is asked of a file that does not fit in cache and whose
  header repeats keywords hundreds of times (``HISTORY``);
* the MegaCam frames are Rice-compressed MEFs (``1PB(nnnn)`` tiles, 37-41 HDUs
  each) where CFITSIO reports *every* HDU as a compressed ``IMAGE`` even though
  ``read_table_info`` still answers for HDU 1 -- a distinction a synthetic
  fixture cannot accidentally get right;
* real primaries are ``NAXIS=0``, so ``NAXIS1`` is genuinely absent and
  ``read_keys`` must raise rather than invent it;
* real header values carry the type split the typed key path exists to
  preserve (``EXPTIME`` is a float in a MegaCam frame and an int in a mosaic,
  ``CRVAL1`` a float, ``CTYPE1`` a string).

The data is not in git (``.gitignore`` covers ``benchmarks_data/``). Fetch it
once::

    bash scripts/fetch_cfht_megacam_sample.sh    # ~2.5 GB, 10 Rice .fz frames
    bash scripts/fetch_cfht_megapipe_sample.sh  # ~5 GB, 3 1.6 GB mosaics

Without the samples every test here skips, so a clone without 9 GB of CFHT data
still gets a green suite. The point is the parity: if the core and the
torch-linked extension ever disagree about a real frame, this file fails.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

core = pytest.importorskip("torchfits._core")

C = pytest.importorskip("torchfits._C")

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "benchmarks_data"

#: Rice-compressed MegaCam MEFs and single-HDU MegaPipe mosaics, as fetched.
MEGACAM = sorted(DATA_ROOT.glob("cfht_megacam/*.fits.fz"))
MEGAPIPE = sorted(DATA_ROOT.glob("cfht_megapipe/*.fits"))
CORPUS = MEGACAM + MEGAPIPE

pytestmark = pytest.mark.skipif(
    not CORPUS,
    reason=(
        "real CFHT sample data absent; run scripts/fetch_cfht_megacam_sample.sh "
        "and scripts/fetch_cfht_megapipe_sample.sh"
    ),
)

#: Keywords present in both real instruments' primaries, and in every HDU of a
#: real header, so the typed key path is exercised on real value cards.
KEYS = ("DATE-OBS", "INSTRUME", "TELESCOP", "FILTER", "OBJECT", "NAXIS")

#: ``fits_hdr2str`` in the vendored CFITSIO concatenates 80-character cards with
#: no separator and appends one END card, so header text is exactly
#: ``80 * (cards + 1)`` bytes. That is an exactness property real headers with
#: 300+ cards and 70-character HISTORY values can break where a 4-card synthetic
#: header cannot.
CARD_WIDTH = 80


def _try(fn: Any, *args: Any) -> Any:
    """Return the probe's value, or its error text.

    Comparing the error text alongside the value is deliberate: "the core
    raised where the extension answered" is a disagreement, and so is "both
    raised but the core said something different about which keyword was
    missing".
    """
    try:
        return fn(*args)
    except (RuntimeError, ValueError) as exc:
        return f"{type(exc).__name__}: {exc}"


def _record(module: Any, path: str, hdu: int) -> dict[str, Any]:
    """Every path-level probe the core exposes, as one comparable record."""
    return {
        "num_hdus": _try(module.read_num_hdus, path),
        "hdu_type": _try(module.read_hdu_type, path, hdu),
        "shape": _try(module.read_shape, path, hdu),
        "nrows": _try(module.read_nrows, path, hdu),
        "colnames": _try(module.read_colnames, path, hdu),
        "table_info": _try(module.read_table_info, path, hdu),
        "header_dict": _try(module.read_header_dict, path, hdu),
        "keys": _try(lambda: module.read_keys(path, hdu, list(KEYS))),
    }


def test_every_hdu_of_every_real_frame_agrees_with_the_extension() -> None:
    """All 409 (frame, HDU) records of the corpus, both modules, byte for byte.

    This is the expensive one -- it opens each 300 MB Rice frame once per HDU --
    and it is the test that would catch a real split, because it compares every
    HDU of every real file rather than a representative handful.
    """
    checked = 0
    for path in CORPUS:
        name = path.name
        num_hdus = core.read_num_hdus(str(path))
        assert num_hdus >= 1
        for hdu in range(num_hdus):
            via_core = _record(core, str(path), hdu)
            via_extension = _record(C, str(path), hdu)
            assert via_core == via_extension, f"{name} HDU {hdu}: " + ", ".join(
                f"{key}: core={via_core[key]!r} extension={via_extension[key]!r}"
                for key in via_core
                if via_core[key] != via_extension[key]
            )
            checked += 1
    assert checked == sum(core.read_num_hdus(str(p)) for p in CORPUS)
    assert checked >= len(CORPUS)


def test_header_text_agrees_with_the_extension_on_real_frames() -> None:
    """Header text is handle-based in ``_C``; compare the three ends of each file."""
    for path in CORPUS:
        num_hdus = core.read_num_hdus(str(path))
        for hdu in sorted({0, min(1, num_hdus - 1), num_hdus - 1}):
            handle = C.open_fits_file(str(path), "r")
            try:
                via_extension = C.read_header_string(handle, hdu)
            finally:
                handle.close()
            via_core = core.read_header_string(str(path), hdu)
            assert via_core == via_extension, f"{path.name} HDU {hdu}"


def test_real_header_text_is_exactly_eighty_column_cards() -> None:
    """The exactness property, on every HDU of every real frame.

    The text is a header *file image* with the newlines dropped, so a truncated
    read shows up as a length that is not a multiple of 80, and a lost card as a
    count that disagrees with the parsed card list. Both are invisible on a
    four-card synthetic header and both are real risks on a 358-card frame.
    """
    for path in CORPUS:
        for hdu in range(core.read_num_hdus(str(path))):
            cards = core.read_header_dict(str(path), hdu)
            text = core.read_header_string(str(path), hdu)
            assert text.isascii(), f"{path.name} HDU {hdu}: non-ASCII in header text"
            assert len(text) == CARD_WIDTH * (len(cards) + 1), (
                f"{path.name} HDU {hdu}: {len(cards)} cards "
                f"should be {CARD_WIDTH * (len(cards) + 1)} bytes, got {len(text)}"
            )
            assert text.endswith("END" + " " * (CARD_WIDTH - 3))
            # Every value card must actually appear in the text. HISTORY and
            # COMMENT are free text with no "=" and are checked by the length
            # invariant above instead.
            for key, _value, _comment in cards:
                if key in {"HISTORY", "COMMENT", ""}:
                    continue
                assert f"{key:<8}=" in text, f"{path.name} HDU {hdu}: {key} missing"


def test_megapipe_mosaics_report_their_true_geometry() -> None:
    """A 1.6 GB mosaic, 21404x20347, answered from headers alone.

    This is the case the split is really for: a file far larger than any cache,
    where every question here is answered by reading a few hundred header cards
    and never touching a pixel.
    """
    assert MEGAPIPE, "no MegaPipe mosaics fetched"
    for path in MEGAPIPE:
        assert core.read_num_hdus(str(path)) == 1
        bitpix, shape = core.read_shape(str(path), 0)
        assert bitpix == -32
        assert shape == (21404, 20347)
        assert shape[0] * shape[1] > 4 * 10**8
        keys = core.read_keys(
            str(path), 0, ["NAXIS1", "NAXIS2", "BITPIX", "CRVAL1", "CTYPE1", "FILTER"]
        )
        assert keys["NAXIS1"] == shape[1] and isinstance(keys["NAXIS1"], int)
        assert keys["NAXIS2"] == shape[0] and isinstance(keys["NAXIS2"], int)
        assert keys["BITPIX"] == -32
        assert isinstance(keys["CRVAL1"], float)
        assert isinstance(keys["CTYPE1"], str)
        # A real WCS, typed: the value cards are what the split had to keep.
        assert keys["CTYPE1"].startswith("RA---")
        assert re.fullmatch(r"[a-zA-Z0-9]+\.MP\d+", str(keys["FILTER"]))


def test_megacam_frames_are_rice_compressed_image_mefs() -> None:
    """CFITSIO calls every HDU of a ``.fz`` an IMAGE; the table probes still answer.

    This is the real-data distinction that synthetic fixtures never exercise and
    that a reader has to get right: the HDU *is* a compressed image (so
    ``read_shape`` gives no axes) yet its header still describes a one-column
    binary table holding the compressed tiles, so ``read_nrows`` and
    ``read_table_info`` are meaningful and must agree with the ``NAXIS2`` and
    ``TFIELDS`` cards.
    """
    assert MEGACAM, "no MegaCam frames fetched"
    for path in MEGACAM:
        num_hdus = core.read_num_hdus(str(path))
        assert num_hdus > 1
        info = core.read_table_info(str(path), 1)
        assert info["colnames"] == ["COMPRESSED_DATA"]
        assert len(info["tforms"]) == 1
        assert info["tforms"][0].startswith("1PB(")
        keys = core.read_keys(str(path), 1, ["NAXIS2", "TFIELDS", "ZCMPTYPE"])
        assert core.read_nrows(str(path), 1) == keys["NAXIS2"]
        assert info["nrows"] == keys["NAXIS2"]
        assert keys["TFIELDS"] == 1
        assert keys["ZCMPTYPE"] == "RICE_1"
        # The primary carries the observation, the extensions the tiles.
        primary = core.read_keys(
            str(path), 0, ["INSTRUME", "TELESCOP", "DATE-OBS", "EXPTIME", "FILTER"]
        )
        assert primary["INSTRUME"] == "MegaPrime"
        assert primary["TELESCOP"] == "CFHT 3.6m"
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(primary["DATE-OBS"]))
        assert isinstance(primary["EXPTIME"], float)
        assert re.fullmatch(r"[a-zA-Z0-9]+\.MP\d+", str(primary["FILTER"]))
        # Every extension is a compressed image, not a table.
        assert {core.read_hdu_type(str(path), h) for h in range(num_hdus)} == {"IMAGE"}


def test_missing_keyword_still_raises_on_real_data() -> None:
    """A ``NAXIS=0`` real primary has no ``NAXIS1``; inventing one would be a lie.

    ``read_keys`` is strict by design (``tests/test_core_library.py`` pins that
    on a synthetic header). Real data is where strictness actually costs
    something, because a caller that asks a compressed primary for ``NAXIS1``
    gets an exception on real frames and must handle it.
    """
    for path in MEGACAM:
        assert core.read_keys(str(path), 0, ["NAXIS"])["NAXIS"] == 0
        with pytest.raises(RuntimeError, match="keyword not found: NAXIS1"):
            core.read_keys(str(path), 0, ["NAXIS1"])
    for path in CORPUS:
        with pytest.raises(RuntimeError, match="keyword not found: NO_SUCH_KEY"):
            core.read_keys(str(path), 0, ["NO_SUCH_KEY"])


def test_real_corpus_is_readable_with_neither_torch_nor_numpy() -> None:
    """The whole corpus, in a fresh interpreter where both imports are blocked.

    ``tests/test_package_isolation.py`` does this on a two-card synthetic file.
    Doing it on 9 GB of real observations is the version that proves the
    metadata path is genuinely independent of torch and numpy rather than
    incidentally so: a 1.6 GB mosaic and ten Rice frames, every HDU, through the
    public API, with ``torch`` and ``numpy`` both raising on import. The parent
    computed the same answers with the torch-linked extension before the child
    ran, so this is also a cross-process parity check.
    """
    expected: dict[str, Any] = {}
    for path in CORPUS:
        num_hdus = core.read_num_hdus(str(path))
        expected[path.name] = {
            "num_hdus": num_hdus,
            "types": [core.read_hdu_type(str(path), h) for h in range(num_hdus)],
            "nrows_hdu1": _try(core.read_nrows, str(path), 1),
            "shape_hdu0": _try(core.read_shape, str(path), 0),
            # The public read_header() returns a dict-like Header, so it has one
            # entry per *unique* keyword; the core returns the raw card list,
            # which on real data carries hundreds of repeated HISTORY cards.
            "entries_hdu0": len(
                {key for key, _v, _c in core.read_header_dict(str(path), 0)}
            ),
            "cards_hdu0": len(core.read_header_dict(str(path), 0)),
        }
    for path in CORPUS:
        assert C.read_num_hdus(str(path)) == expected[path.name]["num_hdus"]

    script = '''
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


def _safe(fn, *args):
    """A probe that must not raise on a real file still records its failure."""
    try:
        return fn(*args)
    except (RuntimeError, ValueError) as exc:
        return type(exc).__name__ + ": " + str(exc)

out = {}
for path in sys.argv[1:]:
    num_hdus = torchfits.read_num_hdus(path)
    out[path.rsplit("/", 1)[-1]] = {
        "num_hdus": num_hdus,
        "types": [torchfits.read_hdu_type(path, h) for h in range(num_hdus)],
        "nrows_hdu1": _safe(torchfits.read_nrows, path, 1),
        "shape_hdu0": _safe(torchfits.read_shape, path, 0),
        "entries_hdu0": len(torchfits.read_header(path, 0)),
        "colnames_hdu1": _safe(torchfits.read_colnames, path, 1),
        "table_info_hdu1": _safe(torchfits.read_table_info, path, 1),
        "core_num_hdus": core.read_num_hdus(path),
        "core_colnames_hdu1": _safe(core.read_colnames, path, 1),
    }
print(json.dumps({
    "torch_imported": "torch" in sys.modules,
    "numpy_imported": "numpy" in sys.modules,
    "extension_imported": "torchfits._C" in sys.modules,
    "build_ids_match": core.core_library_build_id() == core.__build_id__,
    "files": out,
}))
'''
    result = subprocess.run(
        [sys.executable, "-c", script, *[str(p) for p in CORPUS]],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["torch_imported"] is False
    assert payload["numpy_imported"] is False
    # Stronger than "torch is absent": the torch-linked extension module itself
    # was never imported, so nothing dlopened libtorch at all.
    assert payload["extension_imported"] is False
    assert payload["build_ids_match"] is True
    files = payload["files"]
    assert sorted(files) == sorted(expected)
    for name, record in expected.items():
        got = files[name]
        assert got["num_hdus"] == record["num_hdus"] > 0
        assert got["types"] == record["types"]
        assert got["nrows_hdu1"] == record["nrows_hdu1"]
        # Shapes cross a JSON boundary, so a tuple arrives as a list; normalise
        # the parent's value the same way instead of comparing representations.
        assert got["shape_hdu0"] == json.loads(json.dumps(record["shape_hdu0"]))
        assert got["entries_hdu0"] == record["entries_hdu0"]
        # The public API and the raw core module must agree in the child too.
        assert got["core_num_hdus"] == got["num_hdus"]
        assert got["core_colnames_hdu1"] == got["colnames_hdu1"]
    # Real values, not just parity: the child read real observations correctly.
    mega = files[MEGACAM[0].name]
    assert mega["colnames_hdu1"] == ["COMPRESSED_DATA"]
    assert mega["table_info_hdu1"]["nrows"] == mega["nrows_hdu1"]
    mosaic = files[MEGAPIPE[0].name]
    assert mosaic["num_hdus"] == 1
    # A real CFHT mosaic primary carries the full WCS and instrument block, so
    # dozens of distinct keywords, not the eight a synthetic primary has.
    assert mosaic["entries_hdu0"] >= 30
    # Every real frame has repeated keywords, so the dict-like public header and
    # the raw card list genuinely differ here -- if a fetch ever produced a
    # sample set without them, this would stop proving anything.
    assert any(
        record["cards_hdu0"] > record["entries_hdu0"] for record in expected.values()
    )
