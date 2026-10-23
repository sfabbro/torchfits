"""Example: inspect a multi-extension FITS header (astropy FITS-Header tutorial file).

Lists HDUs like ``example_image_mef.py``, then reads a header by EXTNAME when
the file has one, falling back to HDU index otherwise. ``torchfits.open`` and
header inspection run without importing PyTorch. Skips cleanly if the sample
isn't cached (fetch via ``bash scripts/fetch_example_samples.sh``).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples._sample_data import try_ensure_sample  # noqa: E402

import torchfits  # noqa: E402


def main() -> int:
    path = try_ensure_sample("fits_header_mef")
    if path is None:
        print(
            "SKIP: sample 'fits_header_mef' not cached. "
            "Fetch via: bash scripts/fetch_example_samples.sh"
        )
        return 0

    with torchfits.open(str(path)) as hdul:
        hdul.info()
        print(f"HDU count: {len(hdul)}")

        hdu_ref: int | str = 0
        for i, hdu in enumerate(hdul):
            extname = hdu.header.get("EXTNAME")
            shape = hdu.data.shape if hasattr(hdu.data, "shape") else "?"
            print(f"  [{i}] name={extname or 'PRIMARY'} shape={shape}")
            if extname and hdu_ref == 0 and i > 0:
                hdu_ref = extname

        bitpix, shape = torchfits.read_shape(str(path), hdu=hdu_ref)
        print(f"read_shape(hdu={hdu_ref!r}): bitpix={bitpix} shape={shape}")
        print(f"read_hdu_type: {torchfits.read_hdu_type(str(path), hdu=hdu_ref)}")
        header = torchfits.read_header(str(path), hdu=hdu_ref)
        print(f"read_header cards: {len(list(header.keys()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
